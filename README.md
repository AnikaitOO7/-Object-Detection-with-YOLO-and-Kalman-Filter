# Human Detection with YOLO and Kalman Filter

This project demonstrates real-time human detection using the YOLO model, combined with a Kalman filter to track and predict the future movement of humans. It is designed to be highly efficient and adaptable for various use cases, including real-time object tracking and prediction systems.

## Overview

By integrating the power of YOLO (You Only Look Once) for object detection with the Kalman filter for state estimation, this system can:
- Detect humans in real-time.
- Track their movement over time.
- Predict future positions to enable timely reactions in dynamic scenarios.

The inclusion of **Supervision** further enhances efficiency, providing robust monitoring and optimization capabilities for the detection pipeline.

## Features

- **Real-Time Detection:** Leverages YOLOv8 for accurate and fast human detection.
- **Smooth Tracking:** Utilizes Kalman filter to smooth detection outputs and track movements.
- **Future Prediction:** Predicts future positions of detected humans, enabling proactive responses.
- **Efficient Monitoring:** Supervision optimizes and improves the overall detection and tracking performance.
- **Versatility:** Can be adapted for multiple use cases, including robotics, surveillance, and interactive systems.

## Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed. Download Python from [here](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/AnikaitOO7/-Object-Detection-with-YOLO-and-Kalman-Filter
cd -Object-Detection-with-YOLO-and-Kalman-Filter
```

### Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## File Structure

```plaintext
-Object-Detection-with-YOLO-and-Kalman-Filter/
│
├── supervision_live_feed.py
├── README.md
└── requirements.txt
```

## How It Works

### YOLO Detection

YOLO (You Only Look Once) is employed for real-time object detection. The YOLOv8 model detects humans in the video feed, producing bounding boxes for detected objects.

### Kalman Filter Tracking

The Kalman filter is used to smooth the bounding box positions over time, reducing noise and providing more stable tracking. It also predicts future movements based on current trajectories.

### Integration with Supervision

Supervision enhances the detection and tracking pipeline by:
- Optimizing resource utilization.
- Providing a robust framework for monitoring and improving detection accuracy.

### Real-Time Visualization

A Tkinter-based GUI displays the video feed, showcasing both raw and filtered detections. This allows users to observe the system's performance and effectiveness in real-time.

## Applications

This project can be adapted for a variety of use cases, such as:
- **Surveillance Systems:** Predict human movement for proactive security responses.
- **Robotics:** Enable autonomous navigation and interaction.
- **Sports Analysis:** Track and analyze player movements in real-time.
- **Traffic Monitoring:** Predict pedestrian movement for improved safety.

## Future Work

- Expand to multi-object tracking.
- Incorporate advanced AI models for better detection accuracy.
- Implement reactive control systems for real-world applications.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
