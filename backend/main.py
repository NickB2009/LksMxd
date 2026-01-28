from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine.morphology import MorphologyEngine
from engine.psl import PSLEngine
import uvicorn
import os
import sys
import io
import base64
from PIL import Image, ImageDraw

app = FastAPI(title="Morphology Scout API")

# ... (CORS middleware same as before)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engines
morphology_engine = MorphologyEngine()
psl_engine = PSLEngine()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Morphology Scout Engine (PSL Enabled)"}

@app.post("/restart")
def restart_engine():
    """
    Restarts the backend process.
    """
    print("🔄 Restarting Morphology Engine...")
    os.execv(sys.executable, ['python3'] + sys.argv)

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        contents = await file.read()
        
        # 1. Measurement Layer (Geometry)
        morphology_result = morphology_engine.process_image(contents)
        
        if not morphology_result:
             raise HTTPException(status_code=422, detail="No face detected or image unclear.")
             
        # Extract features for PSL
        features = morphology_result["features"] # Normalized 0-1
        landmarks = morphology_result["landmarks"]
        
        # 2. Reasoning Layer (PSL)
        inference_result = psl_engine.infer_harmony(features)
        
        # 3. Construct Response
        return {
            "harmony": {
                "score": round(inference_result["harmony_score"] * 100, 1), 
                "dual": {
                    "natural": round(inference_result["dual_scores"]["natural"] * 100, 1),
                    "expressive": round(inference_result["dual_scores"]["expressive"] * 100, 1),
                    "coherence": round(inference_result["dual_scores"]["coherence"] * 100, 1)
                },
                "explanations": inference_result["explanations"]
            },
            "features": features, 
            "landmarks": landmarks
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/calibration")
def get_calibration_data():
    """Generates a synthetic image and matching landmarks for alignment verification."""
    # 1. Create a 600x600 White Canvas
    W, H = 600, 600
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    
    # 2. Draw Visual Markers (Reference Truth)
    cx, cy = W//2, H//2
    draw.ellipse([cx-10, cy-10, cx+10, cy+10], fill='green', outline='black')
    draw.rectangle([0, 0, W-1, H-1], outline='black', width=5)
    draw.line([0, 0, 50, 50], fill='red', width=3)
    draw.line([0, 50, 50, 0], fill='red', width=3)
    draw.line([W-50, H-50, W, H], fill='red', width=3)
    draw.line([W-50, H, W, H-50], fill='red', width=3)
    
    hy = int(H * 0.05)
    draw.line([0, hy, W, hy], fill='blue', width=2)
    
    # 3. Create Landmark Data
    landmarks = []
    landmarks.append({"x": 0.5, "y": 0.5, "z": 0})
    for i in range(5):
        landmarks.append({"x": 0.1 + (i * 0.2), "y": 0.05, "z": 0})
    landmarks.append({"x": 0.0, "y": 0.0, "z": 0})
    landmarks.append({"x": 1.0, "y": 1.0, "z": 0})
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_b64}"
    
    return {
        "image_url": data_url,
        "harmony": {
            "score": 100.0,
            "explanations": {
                "symmetry": 1.0, "proportions": 1.0, "angles": 1.0, "golden_ratio": 1.0, "balance": 1.0
            }
        },
        "features": {
            "symmetry": 1.0, "proportions": 1.0, "angles": 1.0, "golden_ratio": 1.0, "balance": 1.0
        },
        "landmarks": landmarks
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
