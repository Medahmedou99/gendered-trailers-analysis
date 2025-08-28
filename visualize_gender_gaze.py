import argparse
import json
import os
from glob import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OBJECTIFY_WEIGHTS = {
    'gaze_weight_down': 1.0,   
    'gaze_weight_center': 0.0, 
    'gaze_weight_up': 0.0,
    'gaze_weight_left': 0.0,
    'gaze_weight_right': 0.0,
    'closeup_weight': 1.0      
}

VALID_GAZE = ["left", "right", "up", "down", "center", "unknown"]
VALID_SHOTS = ["close-up", "medium", "wide"]


def load_results(input_dir):
    paths = sorted(glob(os.path.join(input_dir, "*_analysis.json")))
    if not paths:
        raise FileNotFoundError(f"No *_analysis.json files found in {input_dir}")

    rows = []
    for path in paths:
        category = os.path.basename(path).replace("_analysis.json", "")
        with open(path, 'r') as f:
            data = json.load(f)
        for r in data:
            rows.append({
                'category': category,
                'scene_video': r.get('scene_video'),
                'frame': r.get('frame'),
                'gender': r.get('gender'),         
                'shot_type': r.get('shot_type'),   
                'gaze': r.get('gaze'),             
                'x1': (r.get('bbox') or [None]*4)[0],
                'y1': (r.get('bbox') or [None]*4)[1],
                'x2': (r.get('bbox') or [None]*4)[2],
                'y2': (r.get('bbox') or [None]*4)[3],
            })
    df = pd.DataFrame(rows)

    # Clean categories
    df['gender'] = df['gender'].fillna('Unknown')
    df['shot_type'] = df['shot_type'].where(df['shot_type'].isin(VALID_SHOTS), 'other')
    df['gaze'] = df['gaze'].where(df['gaze'].isin(VALID_GAZE), 'unknown')
    return df


def compute_summaries(df):
    summaries = {}

    for category, cdf in df.groupby('category'):
        
        screen = (
            cdf.groupby('gender')['frame']
               .count()
               .rename('frames_with_faces')
               .reset_index()
        )
        total_frames = screen['frames_with_faces'].sum()
        if total_frames > 0:
            screen['share_percent'] = 100.0 * screen['frames_with_faces'] / total_frames
        else:
            screen['share_percent'] = 0.0

        # Shot-type distribution per gender
        shot_dist = (
            cdf.groupby(['gender', 'shot_type'])['frame']
               .count()
               .rename('count')
               .reset_index()
        )
        # Compute close-up rate P(close-up | gender)
        cu_rate = (
            shot_dist.pivot(index='gender', columns='shot_type', values='count')
                     .fillna(0)
        )
        for st in VALID_SHOTS + ['other']:
            if st not in cu_rate.columns:
                cu_rate[st] = 0
        cu_rate['total'] = cu_rate.sum(axis=1)
        cu_rate['closeup_rate'] = np.where(cu_rate['total'] > 0, cu_rate['close-up'] / cu_rate['total'], 0.0)
        cu_rate = cu_rate.reset_index()

        # Gaze distribution per gender
        gaze_dist = (
            cdf.groupby(['gender', 'gaze'])['frame']
               .count()
               .rename('count')
               .reset_index()
        )
        gaze_piv = gaze_dist.pivot(index='gender', columns='gaze', values='count').fillna(0)
        for g in VALID_GAZE:
            if g not in gaze_piv.columns:
                gaze_piv[g] = 0
        gaze_piv['total'] = gaze_piv.sum(axis=1)
        for g in VALID_GAZE:
            gaze_piv[g + '_pct'] = np.where(gaze_piv['total'] > 0, gaze_piv[g] / gaze_piv['total'], 0.0)
        gaze_piv = gaze_piv.reset_index()

        
        def compute_obj_index(row):
            # Match by gender row; need closeup_rate and gaze_pcts
            g = row['gender']
            cu = float(cu_rate.loc[cu_rate['gender'] == g, 'closeup_rate'].values[0]) if g in cu_rate['gender'].values else 0.0
            # gaze pcts row already
            down = float(row.get('down_pct', 0.0))
            center = float(row.get('center_pct', 0.0))
            up = float(row.get('up_pct', 0.0))
            left = float(row.get('left_pct', 0.0))
            right = float(row.get('right_pct', 0.0))
            score = (
                OBJECTIFY_WEIGHTS['closeup_weight'] * cu +
                OBJECTIFY_WEIGHTS['gaze_weight_down'] * down +
                OBJECTIFY_WEIGHTS['gaze_weight_center'] * center +
                OBJECTIFY_WEIGHTS['gaze_weight_up'] * up +
                OBJECTIFY_WEIGHTS['gaze_weight_left'] * left +
                OBJECTIFY_WEIGHTS['gaze_weight_right'] * right
            )
            return score

        obj_df = gaze_piv.copy()
        obj_df['objectify_index'] = obj_df.apply(compute_obj_index, axis=1)

        summaries[category] = {
            'screen_share': screen,
            'shot_distribution': shot_dist,
            'closeup_rate': cu_rate[['gender', 'closeup_rate']],
            'gaze_distribution': gaze_piv,
            'objectify_index': obj_df[['gender', 'objectify_index']]
        }

    return summaries



def plot_screen_share(summaries, out_dir):
    # One grouped bar chart: gender screen-time share by category
    cats = []
    males = []
    females = []

    for cat, s in summaries.items():
        cats.append(cat)
        scr = s['screen_share']
        m = float(scr.loc[scr['gender'] == 'Male', 'share_percent'].values[0]) if 'Male' in scr['gender'].values else 0.0
        f = float(scr.loc[scr['gender'] == 'Female', 'share_percent'].values[0]) if 'Female' in scr['gender'].values else 0.0
        males.append(m)
        females.append(f)

    x = np.arange(len(cats))
    width = 0.35
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, males, width, label='Male')
    plt.bar(x + width/2, females, width, label='Female')
    plt.xticks(x, cats, rotation=45, ha='right')
    plt.ylabel('Screen-time Share (%)')
    plt.title('Screen-time Share by Gender and Category (frame-count proxy)')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'screen_time_share_by_category.png')
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_closeup_rate(summaries, out_dir):
    cats = []
    males = []
    females = []

    for cat, s in summaries.items():
        cats.append(cat)
        cur = s['closeup_rate']
        m = float(cur.loc[cur['gender'] == 'Male', 'closeup_rate'].values[0]) if 'Male' in cur['gender'].values else 0.0
        f = float(cur.loc[cur['gender'] == 'Female', 'closeup_rate'].values[0]) if 'Female' in cur['gender'].values else 0.0
        males.append(m)
        females.append(f)

    x = np.arange(len(cats))
    width = 0.35
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, males, width, label='Male')
    plt.bar(x + width/2, females, width, label='Female')
    plt.xticks(x, cats, rotation=45, ha='right')
    plt.ylabel('Close-up Rate (proportion)')
    plt.title('Close-up Rate by Gender and Category')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'closeup_rate_by_category.png')
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_gaze_distribution(summaries, out_dir):
    # One figure per category: grouped bars of gaze distribution by gender
    for cat, s in summaries.items():
        gd = s['gaze_distribution']
        genders = list(gd['gender'].values)
        gaze_cols = [g + '_pct' for g in VALID_GAZE]
        x = np.arange(len(VALID_GAZE))
        width = 0.35
        fig = plt.figure(figsize=(10, 5))
        for i, g in enumerate(genders):
            vals = [float(gd.loc[gd['gender'] == g, col].values[0]) for col in gaze_cols]
            offset = (-width/2 if i == 0 else width/2)
            plt.bar(x + offset, vals, width, label=g)
        plt.xticks(x, VALID_GAZE)
        plt.ylim(0, 1)
        plt.ylabel('Proportion')
        plt.title(f'Gaze Direction Distribution by Gender — {cat}')
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f'gaze_distribution_{cat}.png')
        plt.savefig(path, dpi=200)
        plt.close(fig)


def plot_objectify_index(summaries, out_dir):
    # Optional index: higher means more close-ups and more "down" gaze.
    cats = []
    males = []
    females = []

    for cat, s in summaries.items():
        cats.append(cat)
        oi = s['objectify_index']
        m = float(oi.loc[oi['gender'] == 'Male', 'objectify_index'].values[0]) if 'Male' in oi['gender'].values else 0.0
        f = float(oi.loc[oi['gender'] == 'Female', 'objectify_index'].values[0]) if 'Female' in oi['gender'].values else 0.0
        males.append(m)
        females.append(f)

    x = np.arange(len(cats))
    width = 0.35
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, males, width, label='Male')
    plt.bar(x + width/2, females, width, label='Female')
    plt.xticks(x, cats, rotation=45, ha='right')
    plt.ylabel('Objectify Index (unitless, proxy)')
    plt.title('Proxy Objectification Index by Gender and Category')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'objectify_index_by_category.png')
    plt.savefig(path, dpi=200)
    plt.close(fig)



def write_report(df, summaries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append("# Gender vs Screen Time, Close-Ups, and Gaze\n")
    lines.append("This report summarizes results computed from your analysis JSONs.\n\n")

    for cat, s in sorted(summaries.items(), key=lambda kv: kv[0]):
        lines.append(f"## Category: {cat}\n")

        scr = s['screen_share']
        lines.append("**Screen-time share (%):**\n")
        for _, row in scr.iterrows():
            lines.append(f"- {row['gender']}: {row['share_percent']:.1f}% (n={int(row['frames_with_faces'])})\n")

        cur = s['closeup_rate']
        lines.append("\n**Close-up rate:**\n")
        for _, row in cur.iterrows():
            lines.append(f"- {row['gender']}: {row['closeup_rate']*100:.1f}% of frames with faces\n")

        lines.append("\n**Gaze distribution (proportion):**\n")
        gd = s['gaze_distribution']
        for _, row in gd.iterrows():
            g = row['gender']
            props = ", ".join([f"{k.replace('_pct','')}: {row[k]*100:.1f}%" for k in gd.columns if k.endswith('_pct')])
            lines.append(f"- {g}: {props}\n")

        oi = s['objectify_index']
        for _, row in oi.iterrows():
            lines.append(f"- Proxy objectify index ({row['gender']}): {row['objectify_index']:.3f}\n")

        lines.append("\n---\n\n")

    report_path = os.path.join(out_dir, 'summary_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[INFO] Report written to {report_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='final_combined', help='Folder with *_analysis.json files')
    parser.add_argument('--out_dir', type=str, default='viz_out', help='Output folder for charts and CSVs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.input_dir)
    summaries = compute_summaries(df)

    # Plots
    plot_screen_share(summaries, args.out_dir)
    plot_closeup_rate(summaries, args.out_dir)
    plot_gaze_distribution(summaries, args.out_dir)
    plot_objectify_index(summaries, args.out_dir)

    write_report(df, summaries, args.out_dir)

    print(f"[DONE] Wrote charts & report to: {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
