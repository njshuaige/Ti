import math
import time
import serial

class ServoController:
    def __init__(self, serial_port="COM9", baud_rate=9600, img_w=1920, img_h=1080,
                 sensor_w_mm=5.37, sensor_h_mm=3.02, focal_len_mm=2.8):
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        self.img_w = img_w
        self.img_h = img_h
        self.sensor_w_mm = sensor_w_mm
        self.sensor_h_mm = sensor_h_mm
        self.focal_len_mm = focal_len_mm

    def pixel_to_yaw_pitch(self, u, v):
        cx = self.img_w / 2
        cy = self.img_h / 2
        pixel_size_x = self.sensor_w_mm / self.img_w
        pixel_size_y = self.sensor_h_mm / self.img_h

        dx = (u - cx) * pixel_size_x
        dy = (v - cy) * pixel_size_y

        yaw = math.degrees(math.atan(dx / self.focal_len_mm))
        pitch = math.degrees(math.atan(dy / self.focal_len_mm))

        return yaw, pitch

    def send_servo_command(self, yaw_deg, pitch_deg):
        if not self.ser.is_open:
            print("串口未打开")
            return
        yaw_int = int(round(yaw_deg))
        pitch_int = int(round(pitch_deg))
        cmd = f"Y{yaw_int:03d}P{pitch_int:03d}\n"
        self.ser.write(cmd.encode('utf-8'))
        print(f"发送指令: {cmd.strip()}")

    def move_along_points(self, points, delay_sec=0.1):
        for i, (u, v) in enumerate(points):
            yaw, pitch = self.pixel_to_yaw_pitch(u, v)
            print(f"点{i+1}: 像素({u},{v}) => Yaw={yaw:.2f}, Pitch={pitch:.2f}")
            self.send_servo_command(yaw, pitch)
            time.sleep(delay_sec)

    def close(self):
        if self.ser.is_open:
            self.ser.close()
