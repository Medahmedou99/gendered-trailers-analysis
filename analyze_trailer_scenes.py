import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import insightface
import mediapipe as mp
import numpy as np
import json
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

GENDER_TRAILERS = "gender_trailers"
TEMP_SCENES_ROOT = "temp_scenes"
OUTPUT_DIR = "final_combined"

os.makedirs(TEMP_SCENES_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda'
SHOT_LABELS = ['close-up', 'medium', 'wide']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(SHOT_LABELS))
    model = model.to(DEVICE)
    model.eval()
    return model

def init_face_model():
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def detect_and_extract_scenes(video_path, output_dir):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    print(f"[INFO] Detected {len(scene_list)} scenes in {base_name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(scene_list):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
        out_path = os.path.join(output_dir, f"{base_name}_scene_{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for frame_num in range(start.get_frames(), end.get_frames()):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()

def predict_shot_type(model, frame):
    with torch.no_grad():
        img = transform(frame).unsqueeze(0).to(DEVICE)
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
    return SHOT_LABELS[pred]

def get_eye_center(landmarks, indexes, w, h):
    coords = []
    for i in indexes:
        lm = landmarks[i]
        if hasattr(lm, 'x') and hasattr(lm, 'y'):
            x = int(lm.x * w)
            y = int(lm.y * h)
        elif isinstance(lm, (tuple, list)) and len(lm) == 2:
            x = int(lm[0] * w)
            y = int(lm[1] * h)
        else:
            print(f"[WARNING] Unexpected landmark format at index {i}: {lm}")
            continue
        coords.append((x, y))
    if not coords:
        return (w // 2, h // 2)
    x_avg = int(np.mean([pt[0] for pt in coords]))
    y_avg = int(np.mean([pt[1] for pt in coords]))
    return (x_avg, y_avg)

def estimate_gaze(image, landmarks):
    h, w, _ = image.shape
    left_eye = get_eye_center(landmarks, LEFT_EYE, w, h)
    right_eye = get_eye_center(landmarks, RIGHT_EYE, w, h)
    left_iris = get_eye_center(landmarks, LEFT_IRIS, w, h)
    right_iris = get_eye_center(landmarks, RIGHT_IRIS, w, h)

    print(f"left_eye: {left_eye}, right_eye: {right_eye}, "
          f"left_iris: {left_iris}, right_iris: {right_iris}")

    try:
        dx = ((left_iris[0] - left_eye[0]) + (right_iris[0] - right_eye[0])) / 2
        dy = ((left_iris[1] - left_eye[1]) + (right_iris[1] - right_eye[1])) / 2
    except Exception as e:
        print(f"[ERROR] Gaze calculation failure: {e}")
        return "unknown"

    direction = "center"
    if abs(dx) > abs(dy):
        if dx > 5:
            direction = "right"
        elif dx < -5:
            direction = "left"
    else:
        if dy > 5:
            direction = "down"
        elif dy < -5:
            direction = "up"
    return direction

def analyze_scenes_folder(shot_model, face_model, scenes_folder):
    combined_results = []
    for filename in os.listdir(scenes_folder):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(scenes_folder, filename)
        cap = cv2.VideoCapture(video_path)
        base_name = os.path.splitext(filename)[0]
        if not cap.isOpened():
            print(f"[ERROR] Cannot open scene {video_path}")
            continue
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % 10 == 0:
                try:
                    shot_type = predict_shot_type(shot_model, frame)
                    faces = face_model.get(frame)
                    if not faces:
                        frame_id += 1
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mesh_result = face_mesh.process(rgb)
                    for face in faces:
                        if face.gender == -1:
                            continue
                        gender = 'Male' if face.gender == 1 else 'Female'
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        gaze = "unknown"
                        if mesh_result.multi_face_landmarks:
                            gaze = estimate_gaze(frame, mesh_result.multi_face_landmarks[0].landmark)
                        combined_results.append({
                            "scene_video": base_name,
                            "frame": frame_id,
                            "gender": gender,
                            "shot_type": shot_type,
                            "gaze": gaze,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                except Exception as e:
                    print(f"[WARN] Failed at frame {frame_id} in {filename}: {e}")
            frame_id += 1
        cap.release()
    return combined_results

def main():
    shot_model = load_model()
    face_model = init_face_model()
    for genre_folder in os.listdir(GENDER_TRAILERS):
        genre_path = os.path.join(GENDER_TRAILERS, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        temp_scene_folder = os.path.join(TEMP_SCENES_ROOT, genre_folder)
        os.makedirs(temp_scene_folder, exist_ok=True)
        print(f"[INFO] Extracting scenes for genre: {genre_folder}")
        for video_file in os.listdir(genre_path):
            if not video_file.endswith('.mp4'):
                continue
            video_full_path = os.path.join(genre_path, video_file)
            detect_and_extract_scenes(video_full_path, temp_scene_folder)
        print(f"[INFO] Analyzing scenes for genre: {genre_folder}")
        combined_results = analyze_scenes_folder(shot_model, face_model, temp_scene_folder)
        output_json_path = os.path.join(OUTPUT_DIR, f"{genre_folder}_analysis.json")
        with open(output_json_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"[DONE] Saved combined analysis JSON: {output_json_path}")

if __name__ == "__main__":
    main()
