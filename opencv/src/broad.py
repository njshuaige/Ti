import cv2
import numpy as np
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

class TriangleDetector:
    def __init__(self, max_perimeter=99999, min_perimeter=100, min_angle=20):
        self.img = None
        self.max_perimeter = max_perimeter
        self.min_perimeter = min_perimeter
        self.min_angle = min_angle
        self.vertices = None
        self.center = None

    def preprocess_image(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(mask, 50, 150)
        self.pre_img = edges

    def find_max_triangle_vertices(self):
        contours, _ = cv2.findContours(self.pre_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_perimeter_now = 0
        self.vertices = None
        self.center = None
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                perimeter = cv2.arcLength(approx, True)
                if self.min_perimeter <= perimeter <= self.max_perimeter and perimeter > max_perimeter_now:
                    angles = []
                    for i in range(3):
                        p0 = approx[i][0]
                        p1 = approx[(i + 1) % 3][0]
                        p2 = approx[(i + 2) % 3][0]
                        v1 = p0 - p1
                        v2 = p2 - p1
                        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(cosine_angle) * 180 / np.pi
                        angles.append(angle)
                    if all(angle >= self.min_angle for angle in angles):
                        max_perimeter_now = perimeter
                        self.vertices = approx.reshape(3, 2)
                        self.center = tuple(np.mean(self.vertices, axis=0).astype(int))
        return self.vertices

    def detect(self, img):
        self.img = img.copy()
        self.preprocess_image()
        self.find_max_triangle_vertices()
        return self.vertices, self.center

    def draw(self, img=None):
        if img is None:
            img = self.img.copy()
        if self.vertices is not None:
            cv2.drawContours(img, [self.vertices], -1, (0, 255, 0), 2)
            for (x, y) in self.vertices:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img, f"({x},{y})", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

class KalmanPredictor:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                                  [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePost = np.zeros((4, 1), np.float32)
        self.predicted = None

    def correct(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)

    def predict(self):
        prediction = self.kalman.predict()
        self.predicted = (int(prediction[0]), int(prediction[1]))
        return self.predicted

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = TriangleDetector(min_perimeter=100, min_angle=20)
    tracker = KalmanPredictor()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        vertices, center = detector.detect(frame)

        # 实时预测与修正
        predicted = tracker.predict()

        if center is not None and vertices is not None and len(vertices) == 3:
            tracker.correct(center[0], center[1])

            # 打印当前三角形三个顶点坐标
            print("当前三角形顶点:")
            for i, pt in enumerate(vertices):
                print(f"  顶点{i + 1}: {pt}")

            offset = vertices.astype(np.float32) - np.array(center, dtype=np.float32)
            predicted_vertices = offset + np.array(predicted, dtype=np.float32)

            # 画实际三角形（绿色）
            cv2.polylines(frame, [vertices.astype(np.int32)], True, (0, 255, 0), 2)
            # 画预测三角形（蓝色）
            cv2.polylines(frame, [predicted_vertices.astype(np.int32)], True, (255, 0, 0), 2)
        else:
            # 如果没检测到三角形，用预测位置画一个小圈圈作为跟踪提示
            if predicted is not None:
                cv2.circle(frame, predicted, 5, (255, 0, 0), -1)

        cv2.imshow("Triangle Tracking - Real Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
