import numpy as np
import rerun as rr

# 初始化 Rerun
rr.init("Multi-Camera Pose Example",spawn=True)

# 假设你有两个摄像头的位姿和内参
# 摄像头1的位姿和内参
pose1 = np.array([
    [0.99, -0.10, 0.10, 1.0],
    [0.10, 0.99, -0.10, 2.0],
    [-0.10, 0.10, 0.99, 3.0],
    [0.0, 0.0, 0.0, 1.0]
])
intrinsic1 = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

# 摄像头2的位姿和内参
pose2 = np.array([
    [0.99, 0.10, -0.10, -1.0],
    [-0.10, 0.99, 0.10, -2.0],
    [0.10, -0.10, 0.99, -3.0],
    [0.0, 0.0, 0.0, 1.0]
])
intrinsic2 = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

# 展示摄像头1
rr.log_camera("camera1", pose=pose1, intrinsic=intrinsic1)

# 展示摄像头2
rr.log_camera("camera2", pose=pose2, intrinsic=intrinsic2)

