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
        def adjust_gamma(image, gamma=1.5):
            invGamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** invGamma * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        def apply_clahe(gray_img):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray_img)

        enhanced = adjust_gamma(self.img, gamma=1.5)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = apply_clahe(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        self.pre_img = edges

    def find_max_triangle_vertices(self):
        contours, _ = cv2.findContours(self.pre_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_perimeter_now = 0
        self.vertices = None
        self.center = None

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                mean_val = cv2.mean(self.img, mask=mask)
                if mean_val[0] < 100 and mean_val[1] < 100 and mean_val[2] < 100:
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


class KalmanPredictor:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
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

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx = frame_width // 2
    cy = frame_height // 2

    HFOV = 60
    VFOV = 45
    angle_per_pixel_x = HFOV / frame_width
    angle_per_pixel_y = VFOV / frame_height

    detector = TriangleDetector(min_perimeter=100, min_angle=20)
    tracker = KalmanPredictor()

    INIT_FRAMES = 20
    AREA_TOLERANCE = 0.5
    ANGLE_TOLERANCE = 15

    init_triangles = []
    template_area = None
    template_angles = None
    frame_count = 0

    def compute_angles(pts):
        angles = []
        for i in range(3):
            p0 = pts[i]
            p1 = pts[(i + 1) % 3]
            p2 = pts[(i + 2) % 3]
            v1 = p0 - p1
            v2 = p2 - p1
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cosine) * 180 / np.pi
            angles.append(angle)
        return sorted(angles)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        vertices, center = detector.detect(frame)
        predicted = tracker.predict()

        if frame_count < INIT_FRAMES:
            if vertices is not None:
                area = cv2.contourArea(vertices)
                angles = compute_angles(vertices)
                init_triangles.append((area, angles))
            frame_count += 1
            cv2.putText(frame, f"Init Frame {frame_count}/{INIT_FRAMES}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        elif frame_count == INIT_FRAMES:
            bins = {}
            for area, angles in init_triangles:
                key = tuple(np.round(angles, 1))
                bins.setdefault(key, []).append(area)
            best_key = max(bins.items(), key=lambda x: len(x[1]))[0]
            best_areas = bins[best_key]
            template_area = np.median(best_areas)
            template_angles = best_key
            frame_count += 1
            print(f"[初始化完成] 模板面积={template_area:.1f}, 角度={template_angles}")

        else:
            if vertices is not None and center is not None:
                area = cv2.contourArea(vertices)
                angles = compute_angles(vertices)
                angle_diff = [abs(a - b) for a, b in zip(angles, template_angles)]
                if abs(area - template_area) / template_area < AREA_TOLERANCE and all(d < ANGLE_TOLERANCE for d in angle_diff):
                    tracker.correct(center[0], center[1])
                    dx = center[0] - cx
                    dy = center[1] - cy
                    yaw = dx * angle_per_pixel_x
                    pitch = -dy * angle_per_pixel_y
                    print(f"[三角形中心] x={center[0]}, y={center[1]} → Yaw={yaw:.2f}°, Pitch={pitch:.2f}°")

                    offset = vertices.astype(np.float32) - np.array(center, dtype=np.float32)
                    predicted_vertices = offset + np.array(predicted, dtype=np.float32)
                    cv2.polylines(frame, [vertices.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [predicted_vertices.astype(np.int32)], True, (255, 0, 0), 2)

                    for i in range(3):
                        pt1 = vertices[i].astype(np.float32)
                        pt2 = vertices[(i + 1) % 3].astype(np.float32)
                        q1 = (pt1 * 0.75 + pt2 * 0.25).astype(int)
                        q2 = (pt1 * 0.25 + pt2 * 0.75).astype(int)
                        cv2.circle(frame, tuple(q1), 4, (0, 255, 255), -1)
                        cv2.circle(frame, tuple(q2), 4, (0, 255, 255), -1)
            else:
                if predicted is not None:
                    cv2.circle(frame, predicted, 5, (255, 0, 0), -1)

        cv2.imshow("Triangle Tracking - Real Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
