import cv2
import numpy as np

class PointDetector:
    def __init__(self, min_area=5, max_area=2000):
        self.red_point = None

    def is_valid_contour(self, cnt):
        area = cv2.contourArea(cnt)
        if area < 5 or area > 2000:
            return False
        return True  # 不再限制圆度、亮度等

    def find_max_contours(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = [cnt for cnt in contours if self.is_valid_contour(cnt)]

        if filtered:
            largest = max(filtered, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = int(x + w / 2), int(y + h / 2)
            return [x, y, w, h, cx, cy]
        else:
            return [0, 0, 0, 0, 0, 0]

    def find_red(self, hsv):
        # 放宽阈值，适应更多红色激光点情况
        lower1 = np.array([0, 70, 150])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        lower2 = np.array([156, 70, 150])
        upper2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        return mask1 | mask2

    def find_point(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = self.find_red(hsv)

        # 调试用：显示掩码图
        cv2.imshow("Red Mask", red_mask)

        return self.find_max_contours(red_mask)

    def detect(self, frame):
        self.red_point = self.find_point(frame)
        return self.red_point

    def draw(self, frame):
        x, y, w, h, cx, cy = self.red_point
        if w > 0 and h > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"Red: ({cx}, {cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

def main():
    cap = cv2.VideoCapture(0)

    # 可选：降低曝光值（若支持）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # 试试 -4, -6, -8

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = PointDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector.detect(frame)
        output = detector.draw(frame)

        cv2.imshow("Laser Detection", output)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
