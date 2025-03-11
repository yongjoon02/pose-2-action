import os
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import time
import threading
import glob
import tomli  # pip install tomli

# 프로젝트 루트(현재 스크립트의 상위 디렉토리)를 설정 및 작업 디렉토리 변경
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
import sys
sys.path.append(project_root)
from src import skeleton
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def get_next_result_folder(result_folder_base):
    existing_folders = glob.glob(os.path.join(result_folder_base, "result*"))
    if not existing_folders:
        return os.path.join(result_folder_base, "result")
    max_num = 0
    for folder in existing_folders:
        base = os.path.join(result_folder_base, "result")
        if folder == base:
            max_num = max(max_num, 1)
        else:
            try:
                num = int(folder.split("result")[-1])
                max_num = max(max_num, num)
            except ValueError:
                continue
    return os.path.join(result_folder_base, f"result{max_num + 1}")

def process_video(video_path, model_path, cam_name, result_folder, device):
    model = YOLO(model_path)
    model = model.to(device)
    model.conf = 0.2
    model.imgsz = (960, 960)
    
    save_dir = os.path.join(result_folder, cam_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Processing {cam_name}: {video_path}")
    print(f"Results will be saved to {save_dir}")
    
    all_keypoints = []
    
    results = model(video_path,
                    task='pose',
                    stream=True,
                    save=True,
                    project=save_dir,
                    name="",
                    device=device,
                    verbose=False)
    
    for frame_idx, r in enumerate(results):
        print(f"\r{cam_name} - Processing frame: {frame_idx + 1}", end="")
        frame_dict = {'frame': frame_idx}
        try:
            if (r.keypoints is not None and 
                len(r.keypoints.data) > 0 and 
                r.keypoints.data[0].shape[0] > 0):
                
                keypoints = r.keypoints.data[0].cpu().numpy()
                frame_dict['person_detected'] = True
                
                for i, name in enumerate(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                           'left_knee', 'right_knee', 'left_ankle', 'right_ankle']):
                    if i < len(keypoints):
                        frame_dict[f'{name}_x'] = float(keypoints[i][0])
                        frame_dict[f'{name}_y'] = float(keypoints[i][1])
                        frame_dict[f'{name}_conf'] = float(keypoints[i][2])
                    else:
                        frame_dict[f'{name}_x'] = 0.0
                        frame_dict[f'{name}_y'] = 0.0
                        frame_dict[f'{name}_conf'] = 0.0
            else:
                frame_dict['person_detected'] = False
                for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                    frame_dict[f'{name}_x'] = 0.0
                    frame_dict[f'{name}_y'] = 0.0
                    frame_dict[f'{name}_conf'] = 0.0
        except Exception as e:
            print(f"\n{cam_name} - Error processing frame {frame_idx}: {str(e)}")
            frame_dict['person_detected'] = False
            for name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
                frame_dict[f'{name}_x'] = 0.0
                frame_dict[f'{name}_y'] = 0.0
                frame_dict[f'{name}_conf'] = 0.0
        
        all_keypoints.append(frame_dict)
    
    if all_keypoints:
        df = pd.DataFrame(all_keypoints)
        df.to_csv(os.path.join(save_dir, "keypoints.csv"), index=False)
        print(f"\n{cam_name} - Keypoints saved to {save_dir}/keypoints.csv")
    
    print(f"\n{cam_name} - Total frames processed: {len(all_keypoints)}")

def main():
    # config.toml 파일 경로 (프로젝트 루트 기준)
    config_path = os.path.join(project_root, "config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config 파일 {config_path}을(를) 찾을 수 없습니다.")
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    # 기존 캘리브레이션 및 triangulation 관련 config는 그대로 사용
    calib_config = config["calibration"]
    triang_config = config["triangulation"]
    
    intrinsic_folder_cam01 = calib_config["intrinsic_folder_cam01"]
    intrinsic_folder_cam02 = calib_config["intrinsic_folder_cam02"]
    extrinsic_folder_cam01 = calib_config["extrinsic_folder_cam01"]
    extrinsic_folder_cam02 = calib_config["extrinsic_folder_cam02"]
    output_dir = calib_config["output_dir"]
    
    json_folder_cam1 = triang_config["json_folder_cam1"]
    json_folder_cam2 = triang_config["json_folder_cam2"]
    
    # [pose_estimation] 섹션에서 비디오, 모델, 결과 폴더 기본 경로 읽기
    pose_config = config["pose_estimation"]
    video1 = pose_config["video1"]
    video2 = pose_config["video2"]
    model_path = pose_config["model_path"]
    result_folder_base = pose_config["result_folder_base"]
    
    result_folder = get_next_result_folder(result_folder_base)
    print(f"Using result folder: {result_folder}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("Using device: cuda")
    else:
        print("CUDA is not available, using CPU")
    
    thread1 = threading.Thread(
        target=process_video, 
        args=(video1, model_path, "cam1", result_folder, device)
    )
    
    thread2 = threading.Thread(
        target=process_video, 
        args=(video2, model_path, "cam2", result_folder, device)
    )
    
    thread1.start()
    time.sleep(1)
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    print(f"\nAll videos processed. Results saved to {result_folder}")

if __name__ == "__main__":
    main()
