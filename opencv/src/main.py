import cv2
import time
from send import ServoController
from broad import Triangle_Detector

# 摄像头图像中心点，默认1920x1080，中心为(960, 540)
SCREEN_CENTER = (960, 540)

# 转换屏幕坐标为舵机角度，按比例缩放（你可以根据实际标定修改比例）
def screen_to_servo(x, y):
    dx = x - SCREEN_CENTER[0]
    dy = y - SCREEN_CENTER[1]
    
    yaw = dx * 0.05  # 每像素对应的yaw角度
    pitch = -dy * 0.05  # pitch正方向朝下
    return yaw, pitch

# 插值
def interpolate_points(p1, p2, steps):
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) * p1[0] + t * p2[0]
        y = (1 - t) * p1[1] + t * p2[1]
        points.append((x, y))
    return points

def main():
    # 初始化
    detector = Triangle_Detector()
    servo = ServoController(port="COM9", baudrate=9600)

    while True:
        triangle = detector.get_triangle()
        if triangle is None or len(triangle) != 3:
            print("未检测到三角形")
            continue

        # 选择两个顶点（比如点1和点2）
        p1, p2 = triangle[0], triangle[1]

        # 插值点（控制步数）
        steps = 10
        interp_points = interpolate_points(p1, p2, steps)

        for (x, y) in interp_points:
            yaw, pitch = screen_to_servo(x, y)
            servo.send(yaw, pitch)
            time.sleep(0.1)  # 控制速度（调整越小越快）

        break  # 只跑一遍就结束，可以改成循环或加入键盘控制退出

if __name__ == "__main__":
    main()
