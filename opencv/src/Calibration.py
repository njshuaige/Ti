import cv2
import numpy as np
import glob

# 1. 设置棋盘格参数
chessboard_size = (9, 6)  # 9x6 的内部角点数（注意：不是格子数！）
square_size = 25  # 每个格子的边长，单位 mm，可自定义

# 2. 生成世界坐标（假设棋盘在 Z=0 平面）
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 世界坐标（每张图）
imgpoints = []  # 图像坐标（每张图）

# 3. 加载所有图像
images = glob.glob('./calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找棋盘角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # 可视化检查
        img_vis = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Corners', img_vis)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 4. 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 5. 输出结果
print("\n=== 相机内参矩阵 ===\n", camera_matrix)
print("\n=== 畸变系数 ===\n", dist_coeffs.ravel())

# 6. 可选：保存参数
np.savez("calibration_result.npz", 
         camera_matrix=camera_matrix, 
         dist_coeffs=dist_coeffs, 
         rvecs=rvecs, 
         tvecs=tvecs)
