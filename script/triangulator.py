import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import click
import toml
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def load_coordinates_openpose(file_path, confidence_threshold=0.0):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not data.get('people') or len(data['people']) == 0:
        raise ValueError("유효한 사람 데이터가 없습니다.")
    keypoints = data['people'][0].get('pose_keypoints_2d', [])
    if not keypoints:
        raise ValueError("키포인트 데이터가 없습니다.")
    coords = []
    for i in range(0, len(keypoints), 3):
        if i+2 < len(keypoints):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf >= confidence_threshold:
                coords.append([x, y])
    return np.array(coords)

def DLT(P1, P2, point1, point2):
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :]
    ]
    A = np.array(A).reshape((4, 4))
    from scipy import linalg
    U, s, Vh = linalg.svd(A.transpose() @ A, full_matrices=False)
    point_3d = Vh[3, 0:3] / Vh[3, 3]
    print('Triangulated point:', point_3d)
    return point_3d

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
            print(f"경고: skeleton.py에서 모델 '{model_name}'을 찾을 수 없습니다.")
            available_models = [name for name in dir(skeleton) if not name.startswith('_') and name.isupper()]
            for name in available_models:
                print(f" - {name}")
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
    except ImportError as e:
        print(f"오류: skeleton.py 또는 anytree 라이브러리를 가져올 수 없습니다: {e}")
        raise

@click.command()
@click.option('--model', default='HALPE_26', help='Pose model from skeleton.py')
def main(model):
    """여러 프레임에 대해 카메라 캘리브레이션, 3D 포즈 추정 및 결과를 실시간 시각화 (시점 유지)"""
    # TOML 파일에서 카메라 파라미터 로드
    
    # 스크립트의 위치를 기준으로 상대 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(script_dir, '..', 'output', 'calibration', 'calib.toml')
    
    try:
        calib_data = toml.load(calib_path)
        print(f"캘리브레이션 파일을 성공적으로 로드했습니다: {calib_path}")
    except FileNotFoundError:
        print(f"오류: 캘리브레이션 파일을 찾을 수 없습니다: {calib_path}")
        raise
    
    # 카메라 1 파라미터
    mtx1 = np.array(calib_data['camera1']['matrix'])
    dist1 = np.array(calib_data['camera1']['distortions'])
    
    # 카메라 2 파라미터
    mtx2 = np.array(calib_data['camera2']['matrix'])
    dist2 = np.array(calib_data['camera2']['distortions'])
    
    # 외부 파라미터(스테레오 변환) 로드
    R = np.array(calib_data['stereo']['rotation'])
    T = np.array(calib_data['stereo']['translation'])
    
    # 프로젝션 행렬 계산
    RT1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)
    P1 = mtx1 @ RT1
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2

    # JSON 파일 목록 불러오기 (여러 프레임)
    cam1_files = sorted(glob.glob(r'F:\calibration\coordinates\cam1_json\*.json'))
    cam2_files = sorted(glob.glob(r'F:\calibration\coordinates\cam2_json\*.json'))
    num_frames = min(len(cam1_files), len(cam2_files))
    if num_frames == 0:
        raise ValueError("JSON 파일을 찾을 수 없습니다.")

    # skeleton 연결 정보 로드
    connections = load_connections(model)

    # matplotlib figure 생성 (interactive mode on)
    plt.ion()
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # scatter와 선 객체를 저장할 변수 (초기에는 None)
    scatter_obj = None
    line_objs = []

    for i in range(num_frames):
        try:
            uvs1 = load_coordinates_openpose(cam1_files[i], confidence_threshold=0.5)
            uvs2 = load_coordinates_openpose(cam2_files[i], confidence_threshold=0.5)
        except Exception as e:
            print(f"Frame {i}: 좌표 로드 실패 - {e}")
            continue

        if uvs1.size == 0 or uvs2.size == 0:
            print(f"Frame {i}: 유효한 키포인트가 없습니다.")
            continue

        num_joints = min(len(uvs1), len(uvs2))
        p3ds = []
        for j in range(num_joints):
            p3d = DLT(P1, P2, uvs1[j], uvs2[j])
            p3ds.append(p3d)
        p3ds = np.array(p3ds)

        # scatter 객체 업데이트 또는 생성
        if scatter_obj is None:
            scatter_obj = ax.scatter(p3ds[:, 0], p3ds[:, 1], p3ds[:, 2], c=None, marker='o')
        else:
            scatter_obj._offsets3d = (p3ds[:, 0], p3ds[:, 1], p3ds[:, 2])

        # 기존의 선 객체 제거
        for line in line_objs:
            line.remove()
        line_objs = []

        # 스켈레톤 연결 선 그리기
        for connection in connections:
            if connection[0] < num_joints and connection[1] < num_joints:
                line, = ax.plot([p3ds[connection[0], 0], p3ds[connection[1], 0]],
                                [p3ds[connection[0], 1], p3ds[connection[1], 1]],
                                [p3ds[connection[0], 2], p3ds[connection[1], 2]], c='red')
                line_objs.append(line)

        # 축 레이블과 제목 업데이트 (시점은 유지됨)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (up)')
        ax.set_title(f'Frame {i}')

        plt.draw()
        plt.pause(0.1)  # 프레임당 0.1초 대기

    print("실시간 시각화 완료. 창을 닫으면 프로그램이 종료됩니다.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
