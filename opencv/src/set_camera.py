import cv2

def set_camera_parameters(device_index=0):
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_BRIGHTNESS, -37)
    cap.set(cv2.CAP_PROP_CONTRAST, 0)
    cap.set(cv2.CAP_PROP_SATURATION, 61)
    cap.set(cv2.CAP_PROP_HUE, -273)
    cap.set(cv2.CAP_PROP_GAIN, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    cap.release()

if __name__ == "__main__":
    set_camera_parameters()
