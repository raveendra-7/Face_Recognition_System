Persistent Face Recognition System
A Python-based face detection and identification system that uses OpenCV for real-time computer vision and MongoDB for persistent data storage. Unlike standard local scripts, this project ensures that user data and trained facial patterns are never lost, even after the program is stopped or the local dataset is deleted.

🚀 Key Features
Real-time Face Capture: Automatically detects and crops faces from your webcam using Haar Cascades.

Database Persistence: Stores face images/data in MongoDB as binary data, removing the reliance on local file storage.

Automated Re-learning: On startup, the system pulls all known faces from the database to train a K-Nearest Neighbors (KNN) classifier.

Multi-User Identification: Supports registering multiple people and identifying them simultaneously in the video stream.

🛠️ Technical Stack
Language: Python

Computer Vision: OpenCV

Database: MongoDB (Local or Atlas)

Machine Learning: Scikit-learn (K-Neighbors Classifier)

Data Processing: NumPy

📋 Prerequisites
Before running the project, ensure you have:

Python 3.8+ installed.

A running MongoDB instance (Local or Atlas).

A functional webcam.

⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/your-repo-name.git
cd your-repo-name
Install dependencies:

Bash
pip install opencv-python pymongo scikit-learn numpy
Configure MongoDB:

Ensure your MongoDB service is started.

If using a custom URI, update the connection string in face_capture.py and face_detection.py.

🖥️ Usage
1. Registering a New User
Run the capture script to detect your face and save the data to the database.

Bash
python face_capture.py
Enter your name when prompted.

The script will capture 50 samples of your face to ensure accuracy.

2. Identifying Faces
Run the detection script to start the real-time identification service.

Bash
python face_detection.py
The system will load all users from MongoDB, train the KNN model, and begin the webcam stream.

Recognized users will have their names displayed above their bounding box.

📝 Future Improvements
Vector Embeddings: Transition from raw pixel flattening to Deep Learning-based embeddings (e.g., FaceNet) for higher accuracy.

Performance: Implement multi-threading for smoother frame processing.

UI: Build a dashboard using React to manage the user database.
