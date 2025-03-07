import cv2 as cv
import glob
import numpy as np
import os
import toml  # pip install toml
import matplotlib.pyplot as plt

def main():
    # config.toml 파일 로드
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config 파일 {config_path}을(를) 찾을 수 없습니다.")
    config = toml.load(config_path)
    calib_config = config["calibration"]

    # config에서 필요한 값들을 읽어옵니다.
    intrinsic_folder_cam01 = calib_config["intrinsic_folder_cam01"]
    intrinsic_folder_cam02 = calib_config["intrinsic_folder_cam02"]
    extrinsic_folder_cam01 = calib_config["extrinsic_folder_cam01"]
    extrinsic_folder_cam02 = calib_config["extrinsic_folder_cam02"]
    checkerboard_rows = calib_config["checkerboard_rows"]
    checkerboard_columns = calib_config["checkerboard_columns"]
    checkerboard_size = calib_config["checkerboard_size"]
    stereo_checkerboard_rows = calib_config["stereo_checkerboard_rows"]
    stereo_checkerboard_columns = calib_config["stereo_checkerboard_columns"]
    stereo_checkerboard_size = calib_config["stereo_checkerboard_size"]
    output_dir = calib_config["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ############################################
    # 1. 캘리브레이션 함수들 (내부 함수)
    ############################################
    def calibrate_camera(images_folder, rows, columns, checker_size):
        images_name = sorted(glob.glob(images_folder))
        images = []
        for imname in images_name:
            im = cv.imread(imname, 1)
            images.append(im)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 체커보드의 월드 좌표 계산 (mm 단위)
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        objp = objp * checker_size

        width = images[0].shape[1]
        height = images[0].shape[0]

        imgpoints = []  # 이미지상의 2D 좌표
        objpoints = []  # 실제 월드의 3D 좌표

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
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print('Intrinsic Calibration RMSE:', ret)
        print('Camera Matrix:\n', mtx)
        print('Distortion Coeffs:', dist)
        return mtx, dist

    def stereo_calibrate(mtx1, dist1, mtx2, dist2, cam1_folder, cam2_folder, rows, columns, checker_size):
        c1_images_names = sorted(glob.glob(cam1_folder))
        c2_images_names = sorted(glob.glob(cam2_folder))

        c1_images = [cv.imread(im, 1) for im in c1_images_names]
        c2_images = [cv.imread(im, 1) for im in c2_images_names]

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        objp = objp * checker_size

        width = c1_images[0].shape[1]
        height = c1_images[0].shape[0]

        imgpoints_left = []
        imgpoints_right = []
        objpoints = []

        for frame1, frame2 in zip(c1_images, c2_images):
            gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            ret1, corners1 = cv.findChessboardCorners(gray1, (columns, rows), None)
            ret2, corners2 = cv.findChessboardCorners(gray2, (columns, rows), None)

            if ret1 and ret2:
                corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                cv.drawChessboardCorners(frame1, (columns, rows), corners1, ret1)
                cv.imshow('Cam01 Calibration', frame1)
                cv.drawChessboardCorners(frame2, (columns, rows), corners2, ret2)
                cv.imshow('Cam02 Calibration', frame2)
                cv.waitKey(0)
                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

        cv.destroyAllWindows()

        flags = cv.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx1, dist1, mtx2, dist2, (width, height),
            criteria=criteria, flags=flags
        )
        print('Stereo Calibration RMSE:', ret)
        return R, T

    ###############################################
    # 2. 캘리브레이션 실행 및 결과 저장
    ###############################################
    # Intrinsic 캘리브레이션 수행
    mtx1, dist1 = calibrate_camera(intrinsic_folder_cam01, checkerboard_rows, checkerboard_columns, checkerboard_size)
    mtx2, dist2 = calibrate_camera(intrinsic_folder_cam02, checkerboard_rows, checkerboard_columns, checkerboard_size)

    # Extrinsic (스테레오) 캘리브레이션 수행
    R, T = stereo_calibrate(
        mtx1, dist1, mtx2, dist2,
        extrinsic_folder_cam01, extrinsic_folder_cam02,
        stereo_checkerboard_rows, stereo_checkerboard_columns, stereo_checkerboard_size
    )

    # 캘리브레이션 결과 데이터를 TOML 파일로 저장
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

    toml_file = os.path.join(output_dir, "calib.toml")
    with open(toml_file, "w") as f:
        toml.dump(calib_data, f)

    print("캘리브레이션 결과가 TOML 파일로 저장되었습니다:")
    print(toml_file)

if __name__ == "__main__":
    main()
