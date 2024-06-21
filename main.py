from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from sqlalchemy import Column, Integer, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import face_recognition
import io
import numpy as np
import cv2
import tensorflow as tf

# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class ImageModel(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, index=True)
    data = Column(LargeBinary)
    encoding = Column(LargeBinary)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Machile learn model for emotions recognition
model = tf.keras.models.load_model('mymodel.keras')

# FastAPI app
app = FastAPI()

def verify_image_in_db(encoding, db_session):
    db_images = db_session.query(ImageModel).all()
    for db_image in db_images:
        db_encoding = np.frombuffer(db_image.encoding, dtype=np.float64)
        match = face_recognition.compare_faces([db_encoding], encoding)
        if match[0]:
            return True
    return False

def preprocess_image(img):
    """
    Preprocess the input image to match the model's expected input format.
    :param image_path: Path to the input image.
    :return: Preprocessed image.
    """
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_emotion(img):
    """
    Predict the emotion of the person in the input image.
    :param image_path: Path to the input image.
    :return: Predicted emotion.
    
    """
    # Define the class names (adjust according to your dataset)
    class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    image = preprocess_image(img)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = class_names[predicted_class]
    return predicted_emotion
    
@app.post("/register-person/")
async def upload_image(file: UploadFile = File(...), name: str = Form(...), status: str = Form(...)):
    try:

        # Read input image file
        contents = await file.read()
        
        # Ensure the file is an image and get the face encoding
        image = face_recognition.load_image_file(io.BytesIO(contents))
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            raise HTTPException(status_code=400, detail="Image must contain exactly one face")
        encoding = encodings[0]
        
        # Save image data and encoding to database
        db = SessionLocal()
        person_exists = verify_image_in_db(encoding=encoding, db_session = db)
        if (person_exists):
            raise HTTPException(status_code=400, detail="Person already exists")
        db_image = ImageModel(name=name, status=status, data=contents, encoding=encoding.tobytes())
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        db.close()
        return {"filename": file.filename, "name": name, "id": db_image.id, "status": db_image.status}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/recognize-image/")
async def check_image(file: UploadFile = File(...)):
    try:
        # Read input image file
        contents = await file.read()
        image_stream = io.BytesIO(contents)


        # Ensure the file is an image and get the face encoding
        image = face_recognition.load_image_file(io.BytesIO(contents))
        encodings = face_recognition.face_encodings(image)

        if len(encodings) != 1:
            raise HTTPException(status_code=400, detail="Image must contain exactly one face")
        encoding = encodings[0]
        
         # Check input image emotion
        np_array = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        emotion = predict_emotion(img=np_array)
        

        # Check if the encoding matches any in the database
        db = SessionLocal()
        db_images = db.query(ImageModel).all()
        
        for db_image in db_images:
            db_encoding = np.frombuffer(db_image.encoding, dtype=np.float64)
            match = face_recognition.compare_faces([db_encoding], encoding)
            if match[0]:
                db.close()
                return {"match": True, "id": db_image.id, "name": db_image.name, "status":db_image.status, "emotion":emotion}
        
        db.close()
        return {"match": False, "status": "desconhecido", "emotion":emotion}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)