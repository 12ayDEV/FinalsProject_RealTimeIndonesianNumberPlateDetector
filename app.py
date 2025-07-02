# ============================================================================================== #
# app.py | Interface for Real-Time Indonesian Number Plate Detection | 48220105 Raynard Prathama #
# ============================================================================================== #

import sys
import os
import cv2
import numpy as np
import onnxruntime
import easyocr
import datetime
import re
import time

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QHBoxLayout, QGroupBox,
                             QComboBox, QTextEdit, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

# ==================== CONFIG ====================
ONNX_MODEL_PATH = r'D:\Special Folder - Python\FinalsProject_RealTimeIndonesianNumberPlateDetector\indo_plate_roboflow_final2\weights\best.onnx'
DEBUG_MODE = True
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
MODEL_IMG_WIDTH = 640
MODEL_IMG_HEIGHT = 640

if not os.path.exists(ONNX_MODEL_PATH):
    print(f"Error: model not found at {ONNX_MODEL_PATH}")
    sys.exit(1)

reader = easyocr.Reader(['en'])

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad)
    im = cv2.copyMakeBorder(im, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad):
    gain = ratio_pad[0][0]; pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]; boxes[..., [1, 3]] -= pad[1]; boxes /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
    return boxes

def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0]); ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2]); ymax = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xmax-xmin)*np.maximum(0, ymax-ymin)
    box_area = (box[2]-box[0])*(box[3]-box[1]); boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    return inter/(box_area+boxes_area-inter+1e-6)

def non_max_suppression(boxes, scores, iou_thres):
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = compute_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thres]
    return keep

class Worker(QThread):
    result_ready = pyqtSignal(np.ndarray, str)
    debug_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.frame_to_process = None
        self.lock = QMutex()
        try:
            self.session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        except Exception as e:
            self.debug_message.emit(f"Error loading models: {e}")
            raise

    def set_frame(self, frame: np.ndarray):
        self.lock.lock()
        try:
            self.frame_to_process = frame.copy()
        finally:
            self.lock.unlock()

    def parse_and_format_plate(self, plate_str: str) -> str:

        # Cleans and formats a license plate string into three distinct parts.
        # Example: 'DR1717SN' -> 'DR 1717 SN'

        clean_str = plate_str.replace(' ', '')

        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', clean_str)
        
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"

        return plate_str

    def detect_characters(self, full_img, plate_coords):
        x1, y1, x2, y2 = plate_coords
        plate_crop = full_img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return ""

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        results = reader.readtext(gray, detail=1, paragraph=False)

        if results:
            results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

            lines = {}
            for res in results:
                ymin = res[0][0][1]
                word = res[1].upper()
                allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                word = ''.join([c for c in word if c in allowed_chars])

                if not word:
                    continue

                line_key = int(ymin // 20)
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append((res[0][0][0], word))

            final_lines = []
            for key in sorted(lines.keys()):
                words = sorted(lines[key], key=lambda x: x[0])
                line_text = ' '.join([w[1] for w in words])
                final_lines.append(line_text)

            full_text = ""
            if len(final_lines) >= 2:
                plate_main = final_lines[0]
                expiry_raw = final_lines[1]

                expiry_parts = re.findall(r'\d+', expiry_raw)
                expiry_formatted = ""
                if len(expiry_parts) >= 2:
                    expiry_formatted = f"{expiry_parts[0]} - {expiry_parts[1]}"
                elif len(expiry_parts) == 1 and len(expiry_parts[0]) == 4:
                    part = expiry_parts[0]
                    expiry_formatted = f"{part[:2]} - {part[2:]}"
                else:
                    expiry_formatted = ' '.join(expiry_parts)

                plate_parts = plate_main.split()
                if len(plate_parts) > 0:
                    area_code = plate_parts[0]
                    if len(area_code) > 1 and area_code.startswith('I'):
                        plate_parts[0] = area_code[1:]
                    if len(plate_parts) > 1:
                        suffix = plate_parts[-1]
                        if len(suffix) > 1 and suffix.startswith('L') and suffix[1:].isalpha():
                            plate_parts[-1] = '4' + suffix[1:]
                    plate_main = ' '.join(plate_parts)

                plate_main = self.parse_and_format_plate(plate_main)

                full_text = f"{plate_main} | {expiry_formatted}"
            
            elif final_lines:
                full_text = self.parse_and_format_plate(final_lines[0])

            return full_text

        return ""

    def run(self):
        self.running = True
        frame_count = 0

        while self.running:
            if self.frame_to_process is None:
                self.msleep(10)
                continue

            self.lock.lock()
            try:
                frame = self.frame_to_process
                self.frame_to_process = None
            finally:
                self.lock.unlock()

            if frame is None or frame.size == 0:
                continue

            frame_count += 1
            self.debug_message.emit(f"Processing frame #{frame_count}")

            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized, ratio, pad = letterbox(img_rgb, (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))
                img = img_resized.transpose(2, 0, 1).astype(np.float32)/255.0
                img = np.expand_dims(img, axis=0)

                outputs = self.session.run(
                    [self.session.get_outputs()[0].name],
                    {self.session.get_inputs()[0].name: img}
                )[0]

                outputs = np.squeeze(outputs, axis=0)
                scores = outputs[:, 4]
                predictions = outputs[scores > CONF_THRESHOLD]

                draw_frame = frame.copy()
                plate_text = ""

                if len(predictions) > 0:
                    boxes_xywh = predictions[:, :4]
                    boxes_xyxy = xywh2xyxy(boxes_xywh)
                    boxes_scaled = scale_boxes((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH), boxes_xyxy, frame.shape, (ratio, pad))
                    keep = non_max_suppression(boxes_scaled, predictions[:, 4], IOU_THRESHOLD)

                    self.debug_message.emit(f"{len(keep)} plates after NMS")

                    for idx in keep:
                        x1, y1, x2, y2 = boxes_scaled[int(idx)].astype(int)

                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0,255,0), 3)

                        plate_text = self.detect_characters(img_rgb, (x1, y1, x2, y2))
                        self.debug_message.emit(f"Character detection result: {plate_text}")

                        if plate_text:
                            cv2.putText(
                                draw_frame, plate_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
                            )

                self.result_ready.emit(draw_frame, plate_text if plate_text else "N/A")

            except Exception as e:
                self.debug_message.emit(f"Inference error: {e}")

            self.msleep(10)

# ==================== MAIN WINDOW ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setMaximumSize(1920, 1080)
        screen = QApplication.primaryScreen().availableGeometry()
        width = min(1200, screen.width() - 100)
        height = min(900, screen.height() - 100)
        self.setGeometry(
            (screen.width()-width)//2,
            (screen.height()-height)//2,
            width, height
        )
        self.setWindowTitle("48220105 RAYNARD PRATHAMA - Indonesian License Plate Recognition")

        self.worker = None
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_analyzing = False

        self.detection_history = {}
        self.locked_plate = None
        self.lock_miss_counter = 0
        self.CONFIDENCE_THRESHOLD = 5
        self.LOCK_MISS_THRESHOLD = 15
        
        self.init_ui()
        self.populate_cameras()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout()

        left_panel = QWidget()
        left_layout = QVBoxLayout()

        self.image_label = QLabel("Camera View")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: black; color: white; border: 2px solid #333;"
        )
        left_layout.addWidget(self.image_label, stretch=3)

        info_box = QGroupBox("Detection Results")
        info_layout = QVBoxLayout()
        self.plate_label = QLabel("Detected Plate: N/A")
        self.plate_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        info_layout.addWidget(self.plate_label)
        self.tax_label = QLabel("Tax Status: N/A")
        self.tax_label.setStyleSheet(
            "font-weight: bold; font-size: 16px; color: #d35400;"
        )
        info_layout.addWidget(self.tax_label)
        info_box.setLayout(info_layout)
        left_layout.addWidget(info_box)

        control_layout = QHBoxLayout()
        self.camera_selector = QComboBox()
        self.camera_selector.setFixedWidth(200)
        self.start_button = QPushButton("Start Analysis")
        self.start_button.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold;"
        )
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(
            "background-color: #c0392b; color: white; font-weight: bold;"
        )
        self.stop_button.setEnabled(False)
        control_layout.addWidget(QLabel("Camera:"))
        control_layout.addWidget(self.camera_selector)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        left_layout.addLayout(control_layout)

        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, stretch=2)

        right_panel = QGroupBox("Debug Console")
        right_layout = QVBoxLayout()
        self.debug_console = QTextEdit()
        self.debug_console.setReadOnly(True)
        self.debug_console.setStyleSheet(
            "font-family: Consolas; font-size: 10pt;"
        )
        debug_controls = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Enable Debug Mode")
        self.debug_checkbox.setChecked(True)
        self.clear_debug_button = QPushButton("Clear Console")
        debug_controls.addWidget(self.debug_checkbox)
        debug_controls.addWidget(self.clear_debug_button)
        right_layout.addLayout(debug_controls)
        right_layout.addWidget(self.debug_console)
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, stretch=1)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.camera_selector.currentIndexChanged.connect(self.preview_camera)
        self.clear_debug_button.clicked.connect(self.debug_console.clear)
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)

    def toggle_debug_mode(self, state):
        global DEBUG_MODE
        DEBUG_MODE = state == Qt.CheckState.Checked.value
        self.debug_console.append(
            f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}"
        )

    def log_debug(self, message: str):
        if not DEBUG_MODE:
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.debug_console.append(f"[{timestamp}] {message}")
        self.debug_console.verticalScrollBar().setValue(
            self.debug_console.verticalScrollBar().maximum()
        )

    def populate_cameras(self):
        self.camera_selector.clear()
        self.available_cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    self.camera_selector.addItem(f"Camera {i}")
                cap.release()
        if not self.available_cameras:
            self.camera_selector.addItem("No cameras found")
            self.camera_selector.setEnabled(False)
            self.start_button.setEnabled(False)
        else:
            self.preview_camera()

    def preview_camera(self):
        if self.is_analyzing or not self.available_cameras:
            return
        cam_idx = self.available_cameras[self.camera_selector.currentIndex()]
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                self.display_frame(frame, "Previewing Camera. Click 'Start Analysis'")

    def start_analysis(self):
        cam_idx = self.available_cameras[self.camera_selector.currentIndex()]
        self.capture = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print(f"Error: Could not open camera {cam_idx}")
            return

        self.is_analyzing = True
        self.worker = Worker()
        self.worker.debug_message.connect(self.log_debug)
        self.worker.result_ready.connect(self.update_ui_from_worker)
        self.worker.start()

        self.timer.start(33)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_selector.setEnabled(False)
        self.plate_label.setText("Detected Plate: ...")
        self.tax_label.setText("Tax Status: ...")
        
    def stop_analysis(self):
        self.is_analyzing = False
        self.timer.stop()
        if self.worker:
            self.worker.running = False
            self.worker.quit()
            self.worker.wait()
            self.worker = None
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_selector.setEnabled(True)
        
        final_plate_info = self.locked_plate
        
        if not final_plate_info and self.detection_history:
            valid_candidates = {
                plate: count for plate, count in self.detection_history.items()
                if "|" in plate and " - " in plate.split("|", 1)[1]
            }
            if valid_candidates:
                final_plate_info = max(valid_candidates, key=valid_candidates.get)
        
        if final_plate_info:
            try:
                main_plate_part, expiry_part = final_plate_info.split('|')
                self.plate_label.setText(f"Detected Plate: {main_plate_part.strip()}")

                expiry_str = expiry_part.strip()
                month_str, year_str = expiry_str.split(' - ')
                expiry_month = int(month_str)
                expiry_year = int(year_str) + 2000

                now = datetime.datetime.now()
                current_month = now.month
                current_year = now.year
                
                status = "ACTIVE"
                color = "#27ae60"  # Green
                
                if expiry_year < current_year or \
                   (expiry_year == current_year and expiry_month < current_month):
                    status = "EXPIRED"
                    color = "#c0392b"  # Red

                self.tax_label.setText(f"Tax Status: {status}")
                self.tax_label.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {color};")

            except (ValueError, IndexError) as e:
                self.log_debug(f"Could not parse final plate info for status check: {e}")
                self.tax_label.setText("Tax Status: Unknown")
                self.tax_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #d35400;")
        else:
            self.plate_label.setText("Detected Plate: N/A")
            self.tax_label.setText("Tax Status: N/A")
            self.tax_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #d35400;")

        self.detection_history.clear()
        self.locked_plate = None
        self.lock_miss_counter = 0

        self.preview_camera()

    def update_frame(self):
        if self.capture and self.is_analyzing and self.worker:
            ret, frame = self.capture.read()
            if ret and frame is not None:
                self.worker.set_frame(frame)

    def update_ui_from_worker(self, processed_frame, plate_text):
        self.display_frame(processed_frame)

        is_valid_reading = plate_text != "N/A" and any(char.isdigit() for char in plate_text)

        if self.locked_plate:
            if plate_text == self.locked_plate:
                self.lock_miss_counter = 0
            else:
                self.lock_miss_counter += 1

            if self.lock_miss_counter > self.LOCK_MISS_THRESHOLD:
                self.locked_plate = None
                self.detection_history.clear()
                self.plate_label.setText("Detected Plate: ...")
                self.tax_label.setText("Tax Status: Pending analysis...")
            else:
                self.plate_label.setText(f"Detected Plate: {self.locked_plate}")

        elif is_valid_reading:
            self.detection_history[plate_text] = self.detection_history.get(plate_text, 0) + 1
            most_frequent_plate = max(self.detection_history, key=self.detection_history.get)
            self.plate_label.setText(f"Detected Plate: {most_frequent_plate}?")

            if self.detection_history.get(plate_text, 0) >= self.CONFIDENCE_THRESHOLD:
                self.locked_plate = plate_text
                self.lock_miss_counter = 0
                self.plate_label.setText(f"Detected Plate: {self.locked_plate}")
                if "|" in self.locked_plate:
                    self.tax_label.setText("Tax Status: Ready to check")
                else:
                    self.tax_label.setText("Tax Status: N/A")
        
        elif not self.locked_plate:
            self.plate_label.setText("Detected Plate: ...")

    def display_frame(self, frame, overlay_text=None):
        try:
            if frame is None or frame.size == 0:
                return
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(
                rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888
            )
            if overlay_text:
                painter = QPainter(qt_image)
                painter.setPen(QPen(QColor(255, 255, 0)))
                painter.setFont(QFont("Arial", 20))
                painter.drawText(20, 40, overlay_text)
                painter.end()
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        except Exception as e:
            self.log_debug(f"Display error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())