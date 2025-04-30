from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    user = "user"
    officer = "officer"

class UserBase(BaseModel):
    name: str
    email: EmailStr
    phone: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: str
    role: UserRole
    image: Optional[str] = None
    created_at: datetime

class TokenResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: UserRole
    image: Optional[str] = None
    token: str

class DiseaseLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class DataEntryStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"

class DataEntryCreate(BaseModel):
    userId: str
    image: str
    diseaseLevel: DiseaseLevel
    notes: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None

class DataEntryResponse(DataEntryCreate):
    id: str
    status: DataEntryStatus
    uploadDate: datetime
    approvedBy: Optional[str] = None
    approvalDate: Optional[datetime] = None
    rejectedBy: Optional[str] = None
    rejectionDate: Optional[datetime] = None
    rejectionReason: Optional[str] = None

class HelpRequestStatus(str, Enum):
    pending = "pending"
    responded = "responded"

class HelpRequestCreate(BaseModel):
    userId: str
    userName: str
    message: str
    image: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None

class HelpRequestResponse(HelpRequestCreate):
    id: str
    status: HelpRequestStatus
    requestDate: datetime
    response: Optional[str] = None
    responderName: Optional[str] = None
    responseDate: Optional[datetime] = None

class BlogCreate(BaseModel):
    title: str
    content: str
    excerpt: str
    image: Optional[str] = None
    author: str
    authorId: str

class BlogResponse(BlogCreate):
    id: str
    date: datetime

class PredictionResponse(BaseModel):
    result: str = Field(..., description="The numerical index of the predicted class (e.g., '0')")
    confidence: str = Field(..., description="The confidence score as a percentage string (e.g., '60.58%')")
    description: str = Field(..., description="Description of the predicted condition")
    severity: str = Field(..., description="Severity level of the condition (e.g., 'low', 'medium', 'high', 'UNKNOWN')")
    solution: str = Field(..., description="Recommended action for the condition")
    disclaimer: str = Field(..., description="Disclaimer about the AI prediction")

class RejectReason(BaseModel):
    reason: str

class HelpResponse(BaseModel):
    response: str

class ActivityType(str, Enum):
    help = "help"
    data = "data"

class UserActivity(BaseModel):
    id: str
    type: ActivityType
    userId: str
    image: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    date: datetime
    status: str
    
    # Help request specific fields
    message: Optional[str] = None
    response: Optional[str] = None
    responderName: Optional[str] = None
    responseDate: Optional[datetime] = None
    
    # Data entry specific fields
    diseaseLevel: Optional[DiseaseLevel] = None
    notes: Optional[str] = None
    approvedBy: Optional[str] = None
    approvalDate: Optional[datetime] = None
    rejectionReason: Optional[str] = None