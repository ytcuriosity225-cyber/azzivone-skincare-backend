import os
import io
import logging
import numpy as np
import pandas as pd
import requests
import json
from pydantic import BaseModel
from typing import Optional, Dict
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Azzivone AI Skin Analysis Backend")

# 1. CORS Configuration
# Allow frontend domains. In production, change "*" to your actual frontend URLs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Model Loading
MODEL_PATH = "azzivone_model.tflite"

interpreter = None
input_details = None
output_details = None

try:
    if os.path.exists(MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("TFLite model loaded successfully.")
    else:
        logger.warning(f"Model file {MODEL_PATH} not found.")
except Exception as e:
    logger.error(f"Error loading TFLite model: {e}")

# Mappings
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CONCERN_MAPPING = {
    'akiec': 'Rough Patches',
    'bcc': 'Texture Issues',
    'bkl': 'Pigmentation/Dark Spots',
    'df': 'Firm Bumps',
    'mel': 'Deep Pigmentation',
    'nv': 'Moles/Texture',
    'vasc': 'Redness'
}

# 3. Excel File Loading
EXCEL_PATH = "Untitled spreadsheet (2).xlsx"
products_df = None

try:
    if os.path.exists(EXCEL_PATH):
        products_df = pd.read_excel(EXCEL_PATH)
        # Convert NaN to None for JSON compliance
        products_df = products_df.where(pd.notnull(products_df), None)
        logger.info("Excel data loaded successfully.")
    else:
        logger.warning(f"Excel file {EXCEL_PATH} not found.")
except Exception as e:
    logger.error(f"Error loading Excel file: {e}")

# Helpers
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Read image, resize to 224x224, and normalize for TFLite model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Expand dims to match (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def find_recommended_products(detected_concern: str) -> list:
    """Filter products from the dataframe matching the detected concern."""
    if products_df is None:
        return []
        
    # Attempt to find the right column flexibly
    concern_col = None
    for col in products_df.columns:
        if 'concern' in str(col).lower():
            concern_col = col
            break
            
    if not concern_col:
        return []
        
    # Filter products where the target concern appears in the column (case-insensitive)
    matched = products_df[products_df[concern_col].astype(str).str.contains(detected_concern, case=False, na=False)]
    return matched.to_dict(orient="records")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API Endpoint to predict skin condition and recommend matching products."""
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Backend error: AI model is missing or failed to load.")
        
    try:
        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]
        
        max_idx = int(np.argmax(predictions))
        confidence_score = float(predictions[max_idx])
        
        predicted_code = CLASSES[max_idx]
        detected_concern = CONCERN_MAPPING.get(predicted_code, "Unknown")
        
        # Recommendations mapping
        recommended_products = find_recommended_products(detected_concern)
        
        return {
            "detected_concern": detected_concern,
            "confidence_score": confidence_score,
            "recommended_products": recommended_products
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Azzivone API running."}


# --- Chat (Gemini) integration ---
class ChatRequest(BaseModel):
    message: str
    analysis: Optional[Dict] = None


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# default model; override with env var GEMINI_MODEL if needed
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/text-bison-001")


def call_gemini(prompt_text: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Send prompt_text to Google Generative Language API (Gemini/text-bison) and return the response."""
    url = f"https://generativelanguage.googleapis.com/v1/{GEMINI_MODEL}:generateText"
    params = {"key": GEMINI_API_KEY}
    payload = {
        "prompt": {"text": prompt_text},
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(url, params=params, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Parse typical response shape: { candidates: [{ output: "..." }, ...] }
    try:
        candidates = data.get("candidates")
        if candidates and len(candidates) > 0:
            return candidates[0].get("output", "").strip()
    except Exception:
        pass

    # Fallbacks
    if isinstance(data.get("output"), str):
        return data.get("output").strip()
    return json.dumps(data)


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """Accepts a user message and prior skin analysis, returns expert advice from Gemini."""
    # Build a clear system prompt for an expert dermatologist
    analysis_text = json.dumps(payload.analysis or {}, indent=2)
    prompt = (
        "You are an expert board-certified dermatologist. "
        "A user provided the following message and previous skin analysis. "
        "Provide a concise, practical expert advice covering: likely concerns, recommended daily routine (morning/evening), product types to look for, safety or warning notes, and one short follow-up question to refine recommendations. "
        "Keep language non-technical and actionable for a general user.\n\n"
        f"User message:\n{payload.message}\n\n"
        f"Previous skin analysis data:\n{analysis_text}\n\n"
        "Format the response as plain paragraphs (no JSON)."
    )

    try:
        answer = call_gemini(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error contacting Gemini API: {str(e)}")

    if not answer:
        raise HTTPException(status_code=502, detail="Empty response from Gemini API")

    return {"advice": answer}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
