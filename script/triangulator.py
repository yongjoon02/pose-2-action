def main():
    import os
    import glob
    import json
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    import sys
    import tomli  # pip install tomli

    from scipy import linalg
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (unused import for 3D plotting)

    # 부모 디렉토리를 경로에 추가하여 skeleton 모듈 불러오기
    sys.path.append('..')
    from src import skeleton

    ############################
    # 1. OpenPose 좌표 불러오기
    ############################
    def load_coordinates_openpose(file_path, confidence_threshold=0.5):
        """
        OpenPose 결과 JSON에서 첫 번째 사람의 pose_keypoints_2d를 불러와서
        (x, y) 좌표 리스트로 반환한다.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data.get('people') or len(data['people']) == 0:
            raise ValueError(f"[ERROR] 파일 {file_path} 내에 유효한 사람 데이터가 없습니다.")

        keypoints = data['people'][0].get('pose_keypoints_2d', [])
        if not keypoints:
            raise ValueError(f"[ERROR] 파일 {file_path} 내 키포인트 데이터가 없습니다.")

        coords = []
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf >= confidence_threshold:
                coords.append([x, y])
            else:
                coords.append([None, None])  # 신뢰도 낮으면 None 처리
        return np.array(coords, dtype=np.float32)

    ##########################
    # 2. 스켈레톤 연결 정보 로드
    ##########################
    def load_connections(model_name='HALPE_26'):
        """
        skeleton.py 내부에 정의된 모델 구조를 불러와서
        연결 정보(부위 간 선분)를 리스트로 반환한다.
        """
        global skeleton  # 전역 변수 skeleton 참조
        
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
                available_models = [
                    name for name in dir(skeleton)
                    if not name.startswith('_') and name.isupper()
                ]
                for name in available_models:
                    print(f" - {name}")
                raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
        except ImportError as e:
            print(f"[ERROR] skeleton.py 또는 anytree 라이브러리를 가져올 수 없습니다: {e}")
            raise

    ###############################
    # 3. DLT를 이용한 3D 좌표 추정
    ###############################
    def DLT(P1, P2, point1, point2):
        """
        두 카메라의 투영행렬(P1, P2)과 각각의 이미지 좌표(point1, point2)를 사용해
        선형(DLT) 방식으로 3D 점을 추정한다.
        """
        # 입력값 유효성 검사
        if None in point1 or None in point2:
            print(f"[WARN] 유효하지 않은 2D 좌표: point1={point1}, point2={point2}")
            return np.array([0, 0, 0])  # 기본값 반환
            
        # NaN/Inf 확인
        if np.isnan(point1).any() or np.isnan(point2).any() or np.isinf(point1).any() or np.isinf(point2).any():
            print(f"[WARN] NaN 또는 Inf 값 발견: point1={point1}, point2={point2}")
            return np.array([0, 0, 0])  # 기본값 반환

        # point1 = (x1, y1), point2 = (x2, y2)
        A = [
            point1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - point1[0] * P1[2, :],
            point2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - point2[0] * P2[2, :]
        ]
        A = np.array(A).reshape((4, 4))
        
        # 행렬에 NaN/Inf가 있는지 확인
        if np.isnan(A).any() or np.isinf(A).any():
            print(f"[WARN] A 행렬에 NaN 또는 Inf 값 발견")
            print(f"A = \n{A}")
            return np.array([0, 0, 0])  # 기본값 반환

        # SVD를 이용한 해 계산
        try:
            _, _, Vh = linalg.svd(A, full_matrices=False)
            # 마지막 행(또는 열)에서 X, Y, Z, W
            X, Y, Z, W = Vh[-1, :]
            
            # W가 0에 가까우면 불안정
            if abs(W) < 1e-8:
                print(f"[WARN] W가 너무 작음: {W}")
                return np.array([0, 0, 0])
                
            point_3d = np.array([X / W, Y / W, Z / W])
            
            # 결과가 합리적인지 확인
            if np.isnan(point_3d).any() or np.isinf(point_3d).any():
                print(f"[WARN] 결과에 NaN 또는 Inf 값: {point_3d}")
                return np.array([0, 0, 0])
                
            return point_3d
            
        except Exception as e:
            print(f"[ERROR] DLT 계산 중 오류 발생: {e}")
            return np.array([0, 0, 0])  # 기본값 반환

    #################################################
    # 4. DLT 삼각 측량 및 3D 스켈레톤 시각화 실행
    #################################################
    # (1) 투영행렬(P1, P2) 불러오기 (TOML 파일 경로 하드코딩)
    calib_path = r"F:\pose-2-action\output\calibration\calib.toml"
    if not os.path.exists(calib_path):
        raise FileNotFoundError("[ERROR] 캘리브레이션 파일을 찾을 수 없습니다.")
    
    # TOML 파일 로드
    with open(calib_path, "rb") as f:
        calib_data = tomli.load(f)
    
    print("[INFO] TOML 파일 로드 완료")
    
    # 내부 파라미터 (카메라 행렬)
    K1 = np.array(calib_data["camera1"]["matrix"])
    K2 = np.array(calib_data["camera2"]["matrix"])
    
    # 왜곡 계수 (필요시 사용)
    dist1 = np.array(calib_data["camera1"]["distortions"][0])
    dist2 = np.array(calib_data["camera2"]["distortions"][0])
    
    # 외부 파라미터 (스테레오 회전 및 변환)
    R = np.array(calib_data["stereo"]["rotation"])
    T = np.array(calib_data["stereo"]["translation"])
    
    # 투영 행렬 계산
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    P1 = K1 @ np.hstack((R1, T1))  # 카메라1: P1 = K1[I|0]
    
    R2 = R
    T2 = T
    P2 = K2 @ np.hstack((R2, T2))  # 카메라2: P2 = K2[R|T]
    
    print(f"[INFO] 투영 행렬 계산 완료")
    print(f"[INFO] P1 형태: {P1.shape}")
    print(f"[INFO] P2 형태: {P2.shape}")

    # (2) JSON 파일 목록 (하드코딩 예시)
    cam1_json_pattern = "F:/calibration/coordinates/cam1_json/*.json"
    cam2_json_pattern = "F:/calibration/coordinates/cam2_json/*.json"
    cam1_files = sorted(glob.glob(cam1_json_pattern))
    cam2_files = sorted(glob.glob(cam2_json_pattern))
    num_frames = min(len(cam1_files), len(cam2_files))
    if num_frames == 0:
        raise ValueError("[ERROR] JSON 파일을 찾을 수 없습니다.")

    # (3) 스켈레톤 연결 정보 불러오기 (model 옵션은 하드코딩 또는 기본값 사용)
    connections = load_connections('HALPE_26')

    # (4) Matplotlib 초기화
    plt.ion()
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    scatter_obj = None
    line_objs = []

    # (5) 모든 프레임 순회하며 3D 삼각 측량 및 시각화
    for i in range(num_frames):
        try:
            uvs1 = load_coordinates_openpose(cam1_files[i], confidence_threshold=0.5)
            uvs2 = load_coordinates_openpose(cam2_files[i], confidence_threshold=0.5)
        except Exception as e:
            print(f"[WARN] Frame {i} 로드 실패: {e}")
            continue

        # 유효한 키포인트 인덱스만 추출
        valid_indices = []
        for j in range(len(uvs1)):
            if not (None in uvs1[j] or None in uvs2[j]):
                valid_indices.append(j)

        # 3D 좌표 계산
        p3ds = []
        for j in valid_indices:
            p3d = DLT(P1, P2, uvs1[j], uvs2[j])
            p3ds.append(p3d)
        p3ds = np.array(p3ds)

        # 시각화 업데이트
        if p3ds.shape[0] > 0:
            # scatter (관절점)
            if scatter_obj is None:
                scatter_obj = ax.scatter(
                    p3ds[:, 0], p3ds[:, 1], p3ds[:, 2],
                    c='b', marker='o'
                )
            else:
                scatter_obj._offsets3d = (p3ds[:, 0], p3ds[:, 1], p3ds[:, 2])

            # 기존 선 제거
            for line in line_objs:
                line.remove()
            line_objs.clear()

            # 스켈레톤 연결 그리기
            for c in connections:
                idx1, idx2 = c
                # 인덱스 유효성 및 삼각 측량된 관절인지 확인
                if idx1 in valid_indices and idx2 in valid_indices:
                    i1 = valid_indices.index(idx1)
                    i2 = valid_indices.index(idx2)
                    line, = ax.plot(
                        [p3ds[i1, 0], p3ds[i2, 0]],
                        [p3ds[i1, 1], p3ds[i2, 1]],
                        [p3ds[i1, 2], p3ds[i2, 2]],
                        c='r'
                    )
                    line_objs.append(line)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')

        plt.draw()
        plt.pause(0.1)

    print("[INFO] 모든 프레임 시각화 완료. 창을 닫으면 프로그램이 종료됩니다.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
