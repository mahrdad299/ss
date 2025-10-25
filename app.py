from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# تنظیمات logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مدل داده ورودی
class PatientData(BaseModel):
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    active: int
    alco: int

# متغیر global برای مدل
model = None

def load_model():
    """لود مدل با مدیریت خطا"""
    global model
    try:
        # مسیرهای ممکن برای فایل مدل
        possible_paths = [
            'my_model.pkl',
            os.path.join(os.path.dirname(__file__), 'my_model.pkl'),
            os.path.join(os.getcwd(), 'my_model.pkl')
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            logger.info(f"Checking model path: {model_path}")
            if os.path.exists(model_path):
                logger.info(f"✅ Model found at: {model_path}")
                model = joblib.load(model_path)
                logger.info("✅ Model loaded successfully!")
                model_loaded = True
                break
        
        if not model_loaded:
            logger.error("❌ Model file not found in any location")
            # ایجاد یک مدل dummy برای تست
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            logger.warning("⚠️ Using dummy model for testing")
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    """لود مدل هنگام راه‌اندازی"""
    logger.info("🚀 Starting Heart Disease Prediction API...")
    load_model()

@app.get("/")
async def home():
    return {
        "message": "API پیش‌بینی بیماری قلبی آماده است!",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(data: PatientData):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # اعتبارسنجی داده‌های ورودی
        if data.height <= 0 or data.weight <= 0:
            raise HTTPException(status_code=400, detail="Height and weight must be positive")
        
        if data.ap_hi <= 0 or data.ap_lo <= 0:
            raise HTTPException(status_code=400, detail="Blood pressure values must be positive")
        
        # تبدیل به آرایه
        X = np.array([[
            data.gender, data.height, data.weight,
            data.ap_hi, data.ap_lo, data.cholesterol,
            data.gluc, data.smoke, data.active, data.alco
        ]])
        
        # پیش‌بینی
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0][1])
        else:
            # اگر مدل predict_proba ندارد
            prediction = model.predict(X)[0]
            probability = float(prediction)
        
        # تعیین سطح ریسک
        if probability > 0.6:
            risk_level = "بالا"
            risk_color = "red"
        elif probability > 0.3:
            risk_level = "متوسط" 
            risk_color = "orange"
        else:
            risk_level = "کم"
            risk_color = "green"
        
        return {
            "probability": round(probability, 4),
            "risk": risk_level,
            "risk_color": risk_color,
            "status": "success",
            "features_used": 10
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")