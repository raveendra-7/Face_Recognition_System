import cv2
import numpy as np
from pymongo import MongoClient

# Setup MongoDB Connection
# Replace with your Atlas URI if you're using the cloud
client = MongoClient("mongodb://localhost:27017/") 
db = client["FaceID_DB"]
collection = db["faces"]

def capture_faces():
    name = input("Enter your name: ")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"Starting capture for {name}...")
    
    while count < 50:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            
            # Convert image to binary for MongoDB
            _, buffer = cv2.imencode('.jpg', face_resized)
            face_binary = buffer.tobytes()
            
            # Save to Database
            collection.insert_one({
                "name": name,
                "image": face_binary
            })
            
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Capturing to MongoDB', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to MongoDB for {name}.")

if __name__ == "__main__":
    capture_faces()
