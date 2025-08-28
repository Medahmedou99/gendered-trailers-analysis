import os
import pandas as pd
import yt_dlp

genres = ['action', 'adventure', 'comedy', 'crime', 'drama', 'fantasy', 'horror', 'romance']
csv_folder = r"C:/Users/medma/Desktop/gender_trailersID"
download_root = r"C:/Users/medma/Desktop/gender_trailers"

os.makedirs(download_root, exist_ok=True)

for genre in genres:
    genre_folder = os.path.join(download_root, genre)
    os.makedirs(genre_folder, exist_ok=True)

    csv_path = os.path.join(csv_folder, f"{genre}_sampled.csv")
    df = pd.read_csv(csv_path)
    
    
    video_ids = df[genre.capitalize()].tolist()

    for video_id in video_ids:
        output_filepath = os.path.join(genre_folder, f"{video_id}.mp4")
        
        if os.path.exists(output_filepath):
            print(f"Already downloaded: {video_id}")
            continue
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'outtmpl': output_filepath,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'quiet': True,
        }
        
        try:
            print(f"Downloading: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Downloaded: {video_id}")
        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading {video_id}: {e}")
