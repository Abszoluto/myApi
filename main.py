from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from sqlalchemy import Column, Integer, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import face_recognition
import io
import numpy as np

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
    data = Column(LargeBinary)
    encoding = Column(LargeBinary)

# Create the database tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

@app.post("/register-image/")
async def upload_image(file: UploadFile = File(...), name: str = Form(...)):
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
        db_image = ImageModel(name=name, data=contents, encoding=encoding.tobytes())
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        db.close()
        
        return {"filename": file.filename, "name": name, "id": db_image.id}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/recognize-image/")
async def check_image(file: UploadFile = File(...)):
    try:
        # Read input image file
        contents = await file.read()
        
        # Ensure the file is an image and get the face encoding
        image = face_recognition.load_image_file(io.BytesIO(contents))
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            raise HTTPException(status_code=400, detail="Image must contain exactly one face")
        encoding = encodings[0]
        
        # Check if the encoding matches any in the database
        db = SessionLocal()
        db_images = db.query(ImageModel).all()
        
        for db_image in db_images:
            db_encoding = np.frombuffer(db_image.encoding, dtype=np.float64)
            match = face_recognition.compare_faces([db_encoding], encoding)
            if match[0]:
                db.close()
                return {"match": True, "id": db_image.id, "name": db_image.name}
        
        db.close()
        return {"match": False}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)