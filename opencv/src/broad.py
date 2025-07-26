import cv2
import numpy as np

class Detector:
    def __init__(self):
        pass

    def find_rectangle_and_draw_corners(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)   # 浮点坐标
            box = box.astype(int)       # 转整数坐标

            corners = box.tolist()
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            for i, (x, y) in enumerate(box):
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(img, f"{i}:({x},{y})", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            break

        return img, corners


if __name__ == "__main__":
    detector = Detector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, corners = detector.find_rectangle_and_draw_corners(frame)

        if corners:
            print("检测到角点坐标：")
            for i, (x, y) in enumerate(corners):
                print(f"角点{i}: x={x}, y={y}")

        cv2.imshow("Rectangle Detection", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

