import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "skincare_ai_db")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "my_model.keras")
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'vasc', 'akiec', 'df']

# Disease Information
DISEASE_INFO = {
    'nv': {
        'disease': "Melanocytic nevi (moles)",
        'description': "Melanocytic nevi are benign pigmented skin lesions, commonly known as moles.",
        'solution': "Monitor for changes in size, shape, or color. Consult a dermatologist if changes occur.",
        'severity': "low"
    },
    'mel': {
        'disease': "Melanoma",
        'description': "Melanoma is a serious type of skin cancer that can spread to other parts of the body.",
        'solution': "Seek immediate medical attention. Early detection and treatment are critical.",
        'severity': "high"
    },
    'bkl': {
        'disease': "Benign keratosis-like lesions",
        'description': "Benign keratosis-like lesions are non-cancerous skin growths, often warty or scaly.",
        'solution': "Usually harmless, but consult a dermatologist if irritated or changing.",
        'severity': "low"
    },
    'bcc': {
        'disease': "Basal cell carcinoma",
        'description': "Basal cell carcinoma is a common skin cancer, typically slow-growing and locally invasive.",
        'solution': "Consult a dermatologist for treatment options, such as surgical removal or topical therapies.",
        'severity': "medium"
    },
    'vasc': {
        'disease': "Vascular lesions (e.g., pyogenic granuloma)",
        'description': "Pyogenic granulomas and hemorrhage are benign vascular lesions that may bleed easily.",
        'solution': "Consult a dermatologist for evaluation and possible removal if persistent or bothersome.",
        'severity': "low"
    },
    'akiec': {
        'disease': "Actinic keratosis and intraepithelial carcinoma",
        'description': "Actinic keratoses and intraepithelial carcinoma are precancerous skin lesions caused by sun exposure.",
        'solution': "Seek dermatological treatment, such as cryotherapy or topical medications, to prevent progression.",
        'severity': "medium"
    },
    'df': {
        'disease': "Dermatofibroma",
        'description': "Dermatofibroma is a benign skin growth, often firm and slightly raised.",
        'solution': "Usually harmless, but consult a dermatologist if it changes or causes discomfort.",
        'severity': "low"
    }
}

# Upload Directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Admin User Configuration
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "dermatologist@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "derm2025")

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# Validate required environment variables
required_vars = ["MONGO_URI", "SECRET_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set")