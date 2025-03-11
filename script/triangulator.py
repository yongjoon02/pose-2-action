import os
import glob
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import tomli  # pip install tomli
import argparse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src import skeleton
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# config.toml 파일 경로 (현재 스크립트 위치 기준 상위 디렉토리의 config.toml)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.toml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"[ERROR] config 파일 {config_path}을(를) 찾을 수 없습니다.")
with open(config_path, "rb") as f:
    config = tomli.load(f)

# config 섹션에서 값 읽기
calib_config = config["calibration"]
triang_config = config["triangulation"]

intrinsic_folder_cam01 = calib_config["intrinsic_folder_cam01"]
intrinsic_folder_cam02 = calib_config["intrinsic_folder_cam02"]
extrinsic_folder_cam01 = calib_config["extrinsic_folder_cam01"]
extrinsic_folder_cam02 = calib_config["extrinsic_folder_cam02"]
output_dir = calib_config["output_dir"]

json_folder_cam1 = triang_config["json_folder_cam1"]
json_folder_cam2 = triang_config["json_folder_cam2"]

# 캘리브레이션 결과 파일 경로 (output 폴더 내에 저장된 파일)
calib_toml = os.path.join(output_dir, "calib.toml")

# 체커보드 관련 설정 (intrinsics와 extrinsics)
rows_intrinsic = calib_config["checkerboard_rows"]
columns_intrinsic = calib_config["checkerboard_columns"]
size_intrinsic = calib_config["checkerboard_size"]
rows_extrinsic = calib_config["stereo_checkerboard_rows"]
columns_extrinsic = calib_config["stereo_checkerboard_columns"]
size_extrinsic = calib_config["stereo_checkerboard_size"]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sys.path.append('..')
from src import skeleton

############################################
# 1. OpenPose 좌표 불러오기 함수
############################################
def load_coordinates_openpose(file_path, confidence_threshold=0.5):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data.get('people') or len(data['people']) == 0:
        raise ValueError(f"[ERROR] 파일 {file_path} 내에 유효한 사람 데이터가 없습니다.")
    keypoints = data['people'][0].get('pose_keypoints_2d', [])
    if not keypoints:
        raise ValueError(f"[ERROR] 파일 {file_path} 내 키포인트 데이터가 없습니다.")
    coords = []
    for i in range(0, len(keypoints), 3):
        x, y, conf_val = keypoints[i], keypoints[i+1], keypoints[i+2]
        if conf_val >= confidence_threshold:
            coords.append([x, y])
        else:
            coords.append([None, None])
    return np.array(coords, dtype=np.float32)

############################################
# 2. 스켈레톤 연결 정보 로드 함수
############################################
def load_connections(model_name='HALPE_26'):
    try:
        from anytree import PreOrderIter
        model_upper = model_name.upper()
        if hasattr(skeleton, model_upper):
            model = getattr(skeleton, model_upper)
            connections = []
            for node in PreOrderIter(model):
                if node.id is not None:
                    for child in node.children:
                        if child.id is not None:
                            connections.append([node.id, child.id])
            return connections
        else:
            print(f"[WARN] skeleton.py에서 모델 '{model_name}'을 찾을 수 없습니다.")
            available_models = [name for name in dir(skeleton) if not name.startswith('_') and name.isupper()]
            for name in available_models:
                print(f" - {name}")
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
    except ImportError as e:
        print(f"[ERROR] skeleton.py 또는 anytree 라이브러리를 가져올 수 없습니다: {e}")
        raise

############################################
# 3. 벡터화된 삼각측량 (기본 DLT, cv.triangulatePoints 사용) 및 reprojection error 기반 필터링
############################################
def triangulate_points(P1, P2, pts1, pts2):
    """
    pts1, pts2: (N,2) 배열 (각 행: [x, y])
    P1, P2: (3,4) 투영행렬
    """
    pts1 = np.array(pts1, dtype=np.float32).T  # 2 x N
    pts2 = np.array(pts2, dtype=np.float32).T  # 2 x N
    pts4d = cv.triangulatePoints(P1, P2, pts1, pts2)  # 4 x N
    pts4d /= pts4d[3, :]  # 동차 좌표 정규화
    return pts4d[:3, :].T  # (N, 3)

def reprojection_error(P, point3d, point2d):
    """
    P: (3,4) 투영행렬, point3d: (3,) 3D 좌표, point2d: (2,) 2D 좌표
    """
    point3d_h = np.append(point3d, 1)
    projected = P @ point3d_h
    projected /= projected[2]
    return np.linalg.norm(projected[:2] - point2d)

############################################
# 4. Intrinsic 및 Extrinsic 캘리브레이션 함수 (기존과 동일)
############################################
def calibrate_camera(images_folder, rows, columns, checker_size):
    images_name = sorted(glob.glob(images_folder))
    if len(images_name) == 0:
        raise ValueError(f"[ERROR] {images_folder}에서 이미지를 찾지 못했습니다.")
    images = [cv.imread(im, 1) for im in images_name]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp = objp * checker_size
    imgpoints = []
    objpoints = []
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (columns, rows), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (columns, rows), corners, ret)
            cv.imshow('Calibration', frame)
            cv.waitKey(500)
            objpoints.append(objp)
            imgpoints.append(corners)
    cv.destroyWindow('Calibration')
    ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Intrinsic Calibration RMSE:', ret)
    print('Camera Matrix:\n', mtx)
    print('Distortion Coeffs:', dist)
    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, cam1_folder, cam2_folder):
    # 각 카메라 폴더에서 이미지 가져오기
    c1_images_names = sorted(glob.glob(cam1_folder))
    c2_images_names = sorted(glob.glob(cam2_folder))
    
    if len(c1_images_names) == 0 or len(c2_images_names) == 0:
        raise ValueError(f"이미지를 찾을 수 없습니다. cam1: {cam1_folder}, cam2: {cam2_folder}")
    
    print(f"[INFO] 카메라1 이미지: {len(c1_images_names)}개, 카메라2 이미지: {len(c2_images_names)}개")
    
    if len(c1_images_names) != len(c2_images_names):
        print(f"[WARN] 카메라1과 카메라2의 이미지 개수가 다릅니다. 동일한 개수의 파일 사용.")
        min_len = min(len(c1_images_names), len(c2_images_names))
        c1_images_names = c1_images_names[:min_len]
        c2_images_names = c2_images_names[:min_len]

    # 체커보드 설정 - config.toml 값과 일치해야 함
    rows = 6  # stereo_checkerboard_rows 
    columns = 8  # stereo_checkerboard_columns
    world_scaling = 60  # stereo_checkerboard_size (mm)
    
    print(f"[INFO] 체커보드 설정: {columns}x{rows}, 크기: {world_scaling}mm")
    
    # 다양한 체커보드 감지 옵션 시도
    chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    # 보다 정확한 코너 감지가 필요하면 CALIB_CB_ACCURACY 추가
    # 보다 빠른 감지가 필요하면 CALIB_CB_FAST_CHECK 추가
    
    # 체커보드 3D 점 준비
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp = objp * world_scaling
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    imgpoints_left = []
    imgpoints_right = []
    objpoints = []
    
    detection_success = 0
    
    for i, (c1_path, c2_path) in enumerate(zip(c1_images_names, c2_images_names)):
        print(f"[INFO] 이미지 쌍 {i+1}/{len(c1_images_names)} 처리 중...")
        
        frame1 = cv.imread(c1_path)
        frame2 = cv.imread(c2_path)
        
        if frame1 is None or frame2 is None:
            print(f"[WARN] 이미지 로드 실패: {c1_path} 또는 {c2_path}")
            continue
        
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        print(f"[INFO] 체커보드 감지 시도: {c1_path}")
        c_ret1, c_corners1 = cv.findChessboardCorners(
            gray1, (columns, rows), flags=chessboard_flags
        )
        
        print(f"[INFO] 체커보드 감지 시도: {c2_path}")
        c_ret2, c_corners2 = cv.findChessboardCorners(
            gray2, (columns, rows), flags=chessboard_flags
        )
        
        if c_ret1 and c_ret2:
            detection_success += 1
            print(f"[INFO] 체커보드 감지 성공 ({detection_success}번째)")
            
            corners1 = cv.cornerSubPix(gray1, c_corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, c_corners2, (11, 11), (-1, -1), criteria)
            
            vis1 = frame1.copy()
            vis2 = frame2.copy()
            cv.drawChessboardCorners(vis1, (columns, rows), corners1, c_ret1)
            cv.drawChessboardCorners(vis2, (columns, rows), corners2, c_ret2)
            
            scale = 0.5
            vis1_resized = cv.resize(vis1, (0, 0), fx=scale, fy=scale)
            vis2_resized = cv.resize(vis2, (0, 0), fx=scale, fy=scale)
            
            cv.imshow('카메라1 - 체커보드', vis1_resized)
            cv.imshow('카메라2 - 체커보드', vis2_resized)
            
            key = cv.waitKey(200)
            if key == 27:
                cv.destroyAllWindows()
                break
            
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            
        else:
            print(f"[WARN] 체커보드 감지 실패: cam1={c_ret1}, cam2={c_ret2}")
            if False:
                cv.imshow('카메라1 - 실패', cv.resize(frame1, (0, 0), fx=0.5, fy=0.5))
                cv.imshow('카메라2 - 실패', cv.resize(frame2, (0, 0), fx=0.5, fy=0.5))
                cv.waitKey(100)
    
    cv.destroyAllWindows()
    print(f"[INFO] 총 {len(c1_images_names)}개 이미지 중 {detection_success}개에서 체커보드 감지됨")
    
    if len(objpoints) == 0:
        raise ValueError("[ERROR] 모든 이미지에서 체커보드 감지 실패. 다음을 확인하세요:\n"
                       "1. 체커보드 크기 설정 (행, 열 개수)\n"
                       "2. 이미지에 체커보드가 완전히 보이는지\n"
                       "3. 이미지 품질 및 조명 상태")
    
    img_shape = gray1.shape[::-1]
    
    print(f"[INFO] 스테레오 캘리브레이션 수행 중 ({len(objpoints)}개 이미지 사용)...")
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2,
        img_shape, criteria=criteria, flags=stereocalibration_flags
    )
    
    print(f"[INFO] 스테레오 캘리브레이션 완료. RMSE: {ret}")
    return R, T

############################################
# 5. 캘리브레이션 결과를 불러와 투영행렬 계산
############################################
if not os.path.exists(calib_toml):
    raise FileNotFoundError("[ERROR] 캘리브레이션 결과 TOML 파일을 찾을 수 없습니다.")
with open(calib_toml, "rb") as f:
    calib_data = tomli.load(f)
print("[INFO] 캘리브레이션 TOML 파일 로드 완료")
K1 = np.array(calib_data["camera1"]["matrix"])
K2 = np.array(calib_data["camera2"]["matrix"])
R_extrinsic = np.array(calib_data["stereo"]["rotation"])
T_extrinsic = np.array(calib_data["stereo"]["translation"])
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R_extrinsic, T_extrinsic))
print(f"[INFO] 투영 행렬 계산 완료")
print(f"[INFO] P1 형태: {P1.shape}")
print(f"[INFO] P2 형태: {P2.shape}")

############################################
# 6. 2D Pose JSON 파일에서 좌표 불러오기 및 삼각측량 + 3D 시각화
############################################
cam1_files = sorted(glob.glob(json_folder_cam1))
cam2_files = sorted(glob.glob(json_folder_cam2))
num_frames = min(len(cam1_files), len(cam2_files))
if num_frames == 0:
    raise ValueError("[ERROR] JSON 파일을 찾지 못했습니다.")

# 스켈레톤 연결 정보
connections = load_connections('HALPE_26')

# 3D 플롯 초기 설정
plt.ion()
fig = plt.figure(figsize=(6, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')

scatter_obj = None
line_objs = []

# reprojection error 임계값
reproj_error_thresh = 1000.0

# ------------------------------
# 3D 좌표를 저장할 리스트
# ------------------------------
all_frames_3d = []

for frame_idx in range(num_frames):
    try:
        uvs1 = load_coordinates_openpose(cam1_files[frame_idx], confidence_threshold=0.1)
        uvs2 = load_coordinates_openpose(cam2_files[frame_idx], confidence_threshold=0.1)
    except Exception as e:
        print(f"[WARN] Frame {frame_idx} 로드 실패: {e}")
        continue
    
    # 유효한 2D 점 인덱스
    valid_indices = [j for j in range(len(uvs1)) if None not in uvs1[j] and None not in uvs2[j]]
    if len(valid_indices) == 0:
        continue
    
    pts1 = uvs1[valid_indices]
    pts2 = uvs2[valid_indices]
    
    # 삼각측량 (기본 DLT)
    pts3d = triangulate_points(P1, P2, pts1, pts2)
    
    # reprojection error로 필터링
    inlier_mask = np.array([
        (reprojection_error(P1, pts3d[k], pts1[k]) < reproj_error_thresh) and
        (reprojection_error(P2, pts3d[k], pts2[k]) < reproj_error_thresh)
        for k in range(len(pts3d))
    ])
    
    # inlier된 점들을 dict로 관리 (키: 2D조인트 인덱스, 값: 3D좌표)
    inlier_points = {
        valid_indices[k]: pts3d[k]
        for k in range(len(pts3d))
        if inlier_mask[k]
    }
    
    # ------------------------------
    # 이 프레임의 3D 관절 좌표를 JSON으로 저장하기 위해 준비
    # (예: 26개 관절이면, 인덱스 순서대로 배열로 만든다)
    # ------------------------------
    n_joints = len(uvs1)  # 보통 26 또는 18 등
    joints_3d = [None] * n_joints
    
    for j_idx, coord_3d in inlier_points.items():
        # coord_3d는 (x, y, z) numpy 배열
        joints_3d[j_idx] = coord_3d.tolist()  # 리스트로 변환
    
    # 프레임 정보를 저장할 딕셔너리
    frame_3d_data = {
        "frame_index": frame_idx,
        "3d_joints": joints_3d  # 인덱스별 3D 좌표
    }
    all_frames_3d.append(frame_3d_data)
    
    # ----------- 시각화 -----------
    # 기존 플롯 제거
    if scatter_obj is not None:
        scatter_obj.remove()
    for line in line_objs:
        line.remove()
    line_objs.clear()
    
    # 3D 점 시각화
    if inlier_points:
        pts3d_filtered = np.array(list(inlier_points.values()))
        scatter_obj = ax.scatter(
            pts3d_filtered[:, 0],
            pts3d_filtered[:, 1],
            pts3d_filtered[:, 2],
            c='b',
            marker='o'
        )
        
        # 스켈레톤 연결
        for conn in connections:
            idx1, idx2 = conn
            if idx1 in inlier_points and idx2 in inlier_points:
                p1 = inlier_points[idx1]
                p2 = inlier_points[idx2]
                line, = ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    c='r'
                )
                line_objs.append(line)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_idx}')
    ax.set_box_aspect([1, 1, 1])
    
    plt.draw()
    plt.pause(0.1)

print("[INFO] 모든 프레임 시각화 완료. 창을 닫으면 프로그램이 종료됩니다.")
plt.ioff()
plt.show()

# ------------------------------
# 최종적으로 all_frames_3d를 JSON에 저장
# ------------------------------
output_3d_json_path = triang_config["output_3d_json"]  # config.toml에서 읽어온 경로
os.makedirs(os.path.dirname(output_3d_json_path), exist_ok=True)

with open(output_3d_json_path, "w", encoding="utf-8") as f:
    json.dump(all_frames_3d, f, indent=4, ensure_ascii=False)

print(f"[INFO] 3D 관절 좌표가 JSON으로 저장되었습니다: {output_3d_json_path}")