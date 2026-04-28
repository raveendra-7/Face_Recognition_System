import cv2
import numpy as np
from pymongo import MongoClient
from sklearn.neighbors import KNeighborsClassifier

client = MongoClient("mongodb://localhost:27017/")
db = client["FaceID_DB"]
collection = db["faces"]

def train_model_from_db():
    data, labels = [], []
    names = []
    
    # Get all unique names in the DB
    unique_names = collection.distinct("name")
    if not unique_names:
        print("Database is empty!")
        return None, None

    print(f"Loading data from MongoDB for: {unique_names}")
    
    for label_id, name in enumerate(unique_names):
        names.append(name)
        # Fetch all images for this specific person
        for record in collection.find({"name": name}):
            # Convert binary back to image
            nparr = np.frombuffer(record['image'], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            data.append(img.flatten())
            labels.append(label_id)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data, labels)
    return model, names

def recognize_faces():
    model, names = train_model_from_db()
    if model is None: return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
            
            label = model.predict(face_resized)[0]
            name = names[label]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition (DB Powered)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
