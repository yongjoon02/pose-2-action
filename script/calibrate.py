import cv2 as cv
import glob
import numpy as np
import os
import toml  # pip install toml
import matplotlib.pyplot as plt

def calibrate_camera(images_folder):
    images_name = sorted(glob.glob(images_folder))
    images = []
    for imname in images_name:
        im = cv.imread(imname, 1)
        images.append(im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria

    row = 6
    columns = 8
    world_scaling = 60  # 60mm 단위

    objp = np.zeros((row * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:row].T.reshape(-1, 2)
    objp = objp * world_scaling

    width = images[0].shape[1]
    height = images[0].shape[0]

    imgpoints = []  # 2D points in image plane
    objpoints = []  # 3D points in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (6, 8), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (columns, row), corners, ret)
            cv.imshow('img', frame)
            cv.waitKey(500)
            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyWindow('img')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, cam1_folder, cam2_folder):
    # 각 카메라 폴더에서 이미지 가져오기
    c1_images_names = sorted(glob.glob(cam1_folder))
    c2_images_names = sorted(glob.glob(cam2_folder))

    c1_images = [cv.imread(im, 1) for im in c1_images_names]
    c2_images = [cv.imread(im, 1) for im in c2_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 3
    columns = 5
    world_scaling = 140

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp = objp * world_scaling

    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, c_corners1 = cv.findChessboardCorners(gray1, (3, 5), None)
        c_ret2, c_corners2 = cv.findChessboardCorners(gray2, (3, 5), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, c_corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, c_corners2, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame1, (3, 5), corners1, c_ret1)
            cv.imshow('cam01_img', frame1)
            cv.drawChessboardCorners(frame2, (3, 5), corners2, c_ret2)
            cv.imshow('cam02_img', frame2)
            cv.waitKey(0)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2, (width, height),
        criteria=criteria, flags=stereocalibration_flags
    )
    print('Stereo Calibration RMSE:', ret)
    return R, T

# 캘리브레이션 결과를 저장할 출력 디렉토리 지정
output_dir = r"F:\pose-2-action\output\calibration"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 각 카메라의 intrinsic 캘리브레이션 수행
mtx1, dist1 = calibrate_camera(r'F:\pose-2-action\data\calibration\intrinsics\cam01\*.jpg')
mtx2, dist2 = calibrate_camera(r'F:\pose-2-action\data\calibration\intrinsics\cam02\*.jpg')

# 스테레오 캘리브레이션 수행
R, T = stereo_calibrate(
    mtx1, dist1, mtx2, dist2,
    r'F:\pose-2-action\data\calibration\extrinsics\cam01\*.jpg',
    r'F:\pose-2-action\data\calibration\extrinsics\cam02\*.jpg'
)

# TOML 형식으로 저장하기 위한 데이터 구성
calib_data = {
    "camera1": {
        "name": "camera1",
        "size": [1920.0, 1080.0],
        "matrix": mtx1.tolist(),
        "distortions": dist1.tolist()
    },
    "camera2": {
        "name": "camera2",
        "size": [1920.0, 1080.0],
        "matrix": mtx2.tolist(),
        "distortions": dist2.tolist()
    },
    "stereo": {
        "rotation": R.tolist(),
        "translation": T.tolist()
    }
}

# calib.toml 파일로 저장
toml_file = os.path.join(output_dir, "calib.toml")
with open(toml_file, "w") as f:
    toml.dump(calib_data, f)

print("캘리브레이션 결과가 TOML 파일로 저장되었습니다:")
print(toml_file)
