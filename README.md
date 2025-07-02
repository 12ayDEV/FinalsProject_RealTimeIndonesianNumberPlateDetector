# Real-Time Indonesian License Plate Detector

This application provides a real-time interface for detecting and recognizing Indonesian vehicle license plates using a webcam feed. It identifies the license plate, performs Optical Character Recognition (OCR) to extract the plate number and expiration date, and then determines the vehicle's tax status.

## Author

* **Raynard Prathama (12ayDEV)** - 48220105

## Features

* **Live Camera Feed**: Select from available cameras connected to your system.
* **Real-Time Plate Detection**: Uses a YOLOv8 model in ONNX format to detect license plates in each frame.
* **Character Recognition (OCR)**: Employs EasyOCR to read the characters and numbers from the detected plate.
* **Intelligent Text Parsing**:
    * Formats the recognized plate number into the standard Indonesian format (e.g., `B 1234 ABC`).
    * Extracts the month and year of expiration, often found on a separate line on the plate.
* **Tax Status Verification**: Compares the plate's expiration date with the current date to determine if the vehicle's tax is `ACTIVE` or `EXPIRED`.
* **Confidence Locking**: To improve accuracy, the system "locks on" to a plate number after several consistent detections, reducing flicker and misreadings.
* **User-Friendly Interface**: Built with PyQt6, the GUI provides a clean layout for viewing the camera feed, detection results, and debug information.
* **Debug Console**: A toggleable console provides detailed, timestamped logs of the detection and recognition process for troubleshooting.

## How It Works

The application follows a multi-stage pipeline for processing each frame from the camera:

1.  **Frame Capture**: The application captures frames from the selected webcam using OpenCV.
2.  **Preprocessing**: Each frame is resized and preprocessed using a `letterbox` function to match the input requirements of the ONNX model (640x640 pixels).
3.  **Plate Detection**: The preprocessed frame is passed to the ONNX model. The model outputs bounding box coordinates for any detected license plates.
4.  **Post-processing**: The model's output is filtered by a confidence threshold, and Non-Maximum Suppression (NMS) is applied to eliminate duplicate detections for the same plate.
5.  **OCR on Plate Crop**: The region of the original frame defined by the final bounding box is cropped out. This cropped image is converted to grayscale and processed with EasyOCR to extract text.
6.  **Text & Status Logic**:
    * The extracted text is cleaned and parsed to identify the main plate number and the expiry date (month/year).
    * The application keeps a history of recent detections. Once a specific plate number is detected several times (meeting a confidence threshold), it is "locked".
    * If an expiry date is successfully parsed from a locked plate, it is compared against the current date to determine the tax status.
7.  **Display**: The original frame is updated with a bounding box drawn around the detected plate, the recognized text, and the final results (Plate Number and Tax Status) are shown in the GUI.

## Requirements

To run this application, you need the following libraries and files:

* Python 3.x
* **ONNX Model**: A trained `best.onnx` model file for license plate detection.
* Python Libraries:
    * `PyQt6`
    * `opencv-python`
    * `onnxruntime`
    * `easyocr`
    * `numpy`

You can install the required Python packages using pip:

```bash
pip install pyqt6 opencv-python onnxruntime easyocr numpy
```

## Usage

1.  **Clone the repository or download the source code.**
2.  **Place the Model File**: Make sure you have the `best.onnx` model file.
3.  **Update Model Path**: Open the `app.py` script and update the `ONNX_MODEL_PATH` variable to the correct absolute path of your `best.onnx` file.

    ```python
    # in app.py
    ONNX_MODEL_PATH = r'C:\path\to\your\model\best.onnx' 
    ```

4.  **Run the Application**: Execute the script from your terminal.

    ```bash
    python app.py
    ```

5.  **Select a Camera**: Choose one of the available cameras from the dropdown menu. A preview will be shown.
6.  **Start Analysis**: Click the "Start Analysis" button to begin the real-time detection process.
7.  **View Results**: The application will draw boxes around detected plates and display the recognized number and tax status.
8.  **Stop Analysis**: Click the "Stop" button to end the session. The final, most confident result will be displayed.
