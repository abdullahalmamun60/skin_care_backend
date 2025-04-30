import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import RequestValidationError
from typing import List, Optional
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
import uuid
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
import pymongo
from pymongo import MongoClient
from bson import ObjectId
from config import (
    MONGO_URI, DB_NAME, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    MODEL_PATH, UPLOAD_DIR, ADMIN_EMAIL, ADMIN_PASSWORD, CORS_ORIGINS,
    CLASS_NAMES, DISEASE_INFO
)
from models import UserLogin, PredictionResponse
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Skincare AI API")

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection with retry and pooling
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, maxPoolSize=50)
    client.admin.command('ping')
    logger.info("MongoDB connection established")
except pymongo.errors.ConnectionError as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise SystemExit("MongoDB connection failed")
db = client[DB_NAME]

# Collections
users_collection = db["users"]
data_collection = db["data_entries"]
help_collection = db["help_requests"]
blog_collection = db["blogs"]

# Create indexes
users_collection.create_index("email", unique=True)
data_collection.create_index("userId")
help_collection.create_index("userId")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Load TensorFlow model with retry
model = None
if os.path.exists(MODEL_PATH):
    for attempt in range(3):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to load model: {str(e)}")
            if attempt == 2:
                logger.error("Failed to load model after 3 attempts")
                model = None
else:
    logger.warning(f"Model file '{MODEL_PATH}' not found. Some features will be unavailable.")

# Create uploads directory
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Class definitions for skin condition classification (use CLASS_NAMES from config)
classes = {i: (class_name, class_name.replace('_', ' ').title()) for i, class_name in enumerate(CLASS_NAMES)}

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    user["_id"] = str(user["_id"])
    return user

# Preprocess image for skin condition classification
def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        img_array = np.array(image)
        img = cv2.resize(img_array, (28, 28))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = (img - np.mean(img)) / np.std(img)
        img = img.reshape(1, 28, 28, 3)
        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

# Save uploaded image asynchronously
async def save_uploaded_image(file: UploadFile) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)
    
    return f"/uploads/{filename}"

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# Authentication routes
@app.post("/auth/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(password)
    image_path = None
    if image:
        image_path = await save_uploaded_image(image)
    
    user = {
        "name": name,
        "email": email,
        "phone": phone,
        "password": hashed_password,
        "image": image_path,
        "role": "user",
        "created_at": datetime.utcnow()
    }
    
    result = users_collection.insert_one(user)
    
    return {"message": "User registered successfully", "id": str(result.inserted_id)}

@app.post("/auth/login")
async def login(form_data: UserLogin = Body(...)):
    user = users_collection.find_one({"email": form_data.email})
    
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"])}, expires_delta=access_token_expires
    )
    
    user_id = str(user["_id"])
    
    return {
        "id": user_id,
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "image": user["image"],
        "phone": user["phone"],
        "token": access_token
    }

# User profile endpoints
@app.get("/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    try:
        return {
            "id": current_user["_id"],
            "name": current_user["name"],
            "email": current_user["email"],
            "phone": current_user["phone"],
            "image": current_user["image"],
            "role": current_user["role"],
            "created_at": current_user["created_at"]
        }
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching user profile: {str(e)}")

@app.post("/user/profile/update")
async def update_user_profile(
    name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        update_data = {}
        if name:
            update_data["name"] = name
        if phone:
            update_data["phone"] = phone
        if image:
            image_path = await save_uploaded_image(image)
            update_data["image"] = image_path
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields provided for update")
        
        update_data["updated_at"] = datetime.utcnow()
        
        result = users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="User not found or no changes made")
        
        updated_user = users_collection.find_one({"_id": ObjectId(current_user["_id"])})
        updated_user["_id"] = str(updated_user["_id"])
        
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": updated_user["_id"],
                "name": updated_user["name"],
                "email": updated_user["email"],
                "phone": updated_user["phone"],
                "image": updated_user["image"],
                "role": updated_user["role"],
                "created_at": updated_user["created_at"]
            }
        }
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating user profile: {str(e)}")

# User list endpoint
@app.get("/users/list")
async def list_users(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can list users")
    
    try:
        users = list(users_collection.find())
        for user in users:
            user["_id"] = str(user["_id"])
            user.pop("password", None)
            user.pop("phone", None)
        return users
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing users: {str(e)}")

# Data predictions endpoint
@app.get("/data/predictions")
async def list_predictions(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can list predictions")
    
    try:
        predictions = data_collection.distinct("prediction")
        return [p for p in predictions if p]
    except Exception as e:
        logger.error(f"Error listing predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing predictions: {str(e)}")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    logger.info("Received POST request to /predict")
    
    if not file.content_type.startswith("image/"):
        logger.error("Invalid file type")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        contents = await file.read()
        if not contents:
            logger.error("Empty file uploaded")
            raise ValueError("Empty file uploaded")
        
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        class_code, class_description = classes[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Fetch disease info from config
        info = DISEASE_INFO.get(class_code, {
            'description': "No description available.",
            'severity': "UNKNOWN",
            'solution': "Consult a dermatologist for further evaluation."
        })
        
        # Format confidence as percentage string
        confidence_percent = f"{confidence * 100:.2f}%"
        
        logger.info(f"Prediction: Result {predicted_class_index}, Confidence: {confidence_percent}")
        
        return PredictionResponse(

           result=str(predicted_class_index),
            confidence=confidence_percent,
            # disease=info['disease'],
            description=info['description'],
            severity=info['severity'],
            solution=info['solution'],
            disclaimer="This AI prediction is not a substitute for professional medical advice. Always consult a dermatologist for an accurate diagnosis."
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Data collection endpoints
@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    userId: str = Form(...),
    diseaseLevel: str = Form(...),
    notes: Optional[str] = Form(None),
    prediction: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        image_path = await save_uploaded_image(file)
        
        data_entry = {
            "userId": userId,
            "image": image_path,
            "diseaseLevel": diseaseLevel,
            "notes": notes,
            "prediction": prediction,
            "confidence": confidence,
            "status": "pending",
            "uploadDate": datetime.utcnow()
        }
        
        result = data_collection.insert_one(data_entry)
        
        return {
            "message": "Image uploaded successfully",
            "id": str(result.inserted_id),
            "image": image_path
        }
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.get("/data/list")
async def list_data_entries(
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        query = {}
        if status:
            query["status"] = status
        
        if current_user["role"] != "officer":
            query["userId"] = current_user["_id"]
        
        entries = list(data_collection.find(query).sort("uploadDate", -1))
        
        for entry in entries:
            entry["_id"] = str(entry["_id"])
            user = users_collection.find_one({"_id": ObjectId(entry["userId"])})
            entry["userName"] = user["name"] if user else "Unknown User"
        
        return entries
    except Exception as e:
        logger.error(f"Error listing data entries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing data entries: {str(e)}")

@app.post("/data/{entry_id}/approve")
async def approve_data_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can approve data entries")
    
    try:
        result = data_collection.update_one(
            {"_id": ObjectId(entry_id)},
            {
                "$set": {
                    "status": "approved",
                    "approvedBy": current_user["name"],
                    "approvalDate": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Data entry not found")
        
        return {"message": "Data entry approved successfully"}
    except Exception as e:
        logger.error(f"Error approving data entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error approving data entry: {str(e)}")

@app.post("/data/{entry_id}/reject")
async def reject_data_entry(
    entry_id: str,
    reason: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can reject data entries")
    
    try:
        result = data_collection.update_one(
            {"_id": ObjectId(entry_id)},
            {
                "$set": {
                    "status": "rejected",
                    "rejectionReason": reason,
                    "rejectedBy": current_user["name"],
                    "rejectionDate": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Data entry not found")
        
        return {"message": "Data entry rejected successfully"}
    except Exception as e:
        logger.error(f"Error rejecting data entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error rejecting data entry: {str(e)}")

@app.get("/data/download")
async def download_data_entries(
    userId: Optional[str] = Query(None),
    diseaseLevel: Optional[str] = Query(None),
    prediction: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can download data entries")
    
    try:
        query = {}
        if userId:
            query["userId"] = userId
        if diseaseLevel:
            query["diseaseLevel"] = diseaseLevel
        if prediction:
            query["prediction"] = prediction
        if status:
            query["status"] = status
        
        entries = list(data_collection.find(query))
        
        if not entries:
            raise HTTPException(status_code=404, detail="No data entries found for the specified filters")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in entries:
                image_path = entry.get("image")
                if image_path:
                    abs_path = os.path.join(UPLOAD_DIR, image_path.lstrip("/uploads/"))
                    if os.path.exists(abs_path):
                        filename = f"{entry['_id']}_{os.path.basename(image_path)}"
                        zip_file.write(abs_path, filename)
                    else:
                        logger.warning(f"Image not found: {abs_path}")
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"data_entries_{timestamp}.zip"
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
    except Exception as e:
        logger.error(f"Error downloading data entries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading data entries: {str(e)}")

# Help request endpoints
@app.post("/gethelp/send")
async def send_help_request(
    file: UploadFile = File(...),
    userId: str = Form(...),
    message: str = Form(...),
    prediction: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        image_path = await save_uploaded_image(file)
        
        help_request = {
            "userId": userId,
            "userName": current_user["name"],
            "message": message,
            "image": image_path,
            "prediction": prediction,
            "confidence": confidence,
            "status": "pending",
            "requestDate": datetime.utcnow()
        }
        
        result = help_collection.insert_one(help_request)
        
        return {
            "message": "Help request sent successfully",
            "id": str(result.inserted_id)
        }
    except Exception as e:
        logger.error(f"Error sending help request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending help request: {str(e)}")

@app.get("/gethelp/list")
async def list_help_requests(
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        query = {}
        if status:
            query["status"] = status
        
        if current_user["role"] != "officer":
            query["userId"] = current_user["_id"]
        
        requests = list(help_collection.find(query).sort("requestDate", -1))
        
        for request in requests:
            request["_id"] = str(request["_id"])
        
        return requests
    except Exception as e:
        logger.error(f"Error listing help requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing help requests: {str(e)}")

@app.post("/gethelp/{request_id}/respond")
async def respond_to_help_request(
    request_id: str,
    response: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can respond to help requests")
    
    try:
        result = help_collection.update_one(
            {"_id": ObjectId(request_id)},
            {
                "$set": {
                    "status": "responded",
                    "response": response,
                    "responderName": current_user["name"],
                    "responseDate": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Help request not found")
        
        return {"message": "Response sent successfully"}
    except Exception as e:
        logger.error(f"Error responding to help request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error responding to help request: {str(e)}")

# Blog endpoints
@app.post("/blog/create")
async def create_blog(
    title: str = Form(...),
    content: str = Form(...),
    excerpt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "officer":
        raise HTTPException(status_code=403, detail="Only officers can create blogs")
    
    try:
        image_path = None
        if image:
            image_path = await save_uploaded_image(image)
        
        blog = {
            "title": title,
            "content": content,
            "excerpt": excerpt,
            "image": image_path,
            "author": current_user["name"],
            "authorId": current_user["_id"],
            "date": datetime.utcnow()
        }
        
        result = blog_collection.insert_one(blog)
        
        return {
            "message": "Blog created successfully",
            "id": str(result.inserted_id)
        }
    except Exception as e:
        logger.error(f"Error creating blog: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating blog: {str(e)}")

@app.get("/blog/list")
async def list_blogs():
    try:
        blogs = list(blog_collection.find().sort("date", -1))
        
        for blog in blogs:
            blog["_id"] = str(blog["_id"])
            blog["authorId"] = str(blog["authorId"])
        
        return blogs
    except Exception as e:
        logger.error(f"Error listing blogs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing blogs: {str(e)}")

@app.get("/blog/{blog_id}")
async def get_blog(blog_id: str):
    try:
        blog = blog_collection.find_one({"_id": ObjectId(blog_id)})
        
        if not blog:
            raise HTTPException(status_code=404, detail="Blog not found")
        
        blog["_id"] = str(blog["_id"])
        blog["authorId"] = str(blog["authorId"])
        
        return blog
    except Exception as e:
        logger.error(f"Error getting blog: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting blog: {str(e)}")

# User history endpoint
@app.get("/user/history")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["_id"]
        
        help_requests = list(help_collection.find({"userId": user_id}).sort("requestDate", -1))
        for request in help_requests:
            request["_id"] = str(request["_id"])
            request["type"] = "help"
        
        data_entries = list(data_collection.find({"userId": user_id}).sort("uploadDate", -1))
        for entry in data_entries:
            entry["_id"] = str(entry["_id"])
            entry["type"] = "data"
        
        activities = help_requests + data_entries
        activities.sort(key=lambda x: x.get("requestDate", x.get("uploadDate")), reverse=True)
        
        return activities
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user history: {str(e)}")

# Create admin user on startup
@app.on_event("startup")
async def create_admin_user():
    try:
        if not users_collection.find_one({"email": ADMIN_EMAIL}):
            hashed_password = get_password_hash(ADMIN_PASSWORD)
            
            admin_user = {
                "name": "Admin",
                "email": ADMIN_EMAIL,
                "phone": "1234567890",
                "password": hashed_password,
                "image": None,
                "role": "officer",
                "created_at": datetime.utcnow()
            }
            
            users_collection.insert_one(admin_user)
            logger.info(f"Admin user created with email: {ADMIN_EMAIL}")
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)