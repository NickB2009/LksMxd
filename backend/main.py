from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine.morphology import MorphologyEngine
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

# Initialize Engine (Singleton-ish)
morphology_engine = MorphologyEngine()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Morphology Scout Engine"}

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
        
        # Morphology (Feature-centric analysis)
        morphology_data = morphology_engine.process_image(contents)
        
        if not morphology_data:
             raise HTTPException(status_code=422, detail="No face detected or image unclear.")
             
        return {
            "analysis": morphology_data
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/debug/calibration")
def get_calibration_data():
    """Generates a synthetic image and matching landmarks for alignment verification."""
    # 1. Create a 600x600 White Canvas
    W, H = 600, 600
    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    
    # 2. Draw Visual Markers (Reference Truth)
    # Center Green Circle
    cx, cy = W//2, H//2
    draw.ellipse([cx-10, cy-10, cx+10, cy+10], fill='green', outline='black')
    
    # Target Box (to show object-fit limits)
    draw.rectangle([0, 0, W-1, H-1], outline='black', width=5)
    
    # Corner Red Crosses
    # Top-Left (0,0)
    draw.line([0, 0, 50, 50], fill='red', width=3)
    draw.line([0, 50, 50, 0], fill='red', width=3)
    
    # Bottom-Right (1,1)
    draw.line([W-50, H-50, W, H], fill='red', width=3)
    draw.line([W-50, H, W, H-50], fill='red', width=3)
    
    # Hairline Bar (Blue) at 5% height
    hy = int(H * 0.05)
    draw.line([0, hy, W, hy], fill='blue', width=2)
    
    # 3. Create Landmark Data that MATCHES exactly
    # Normalized coordinates [0.0, 1.0]
    landmarks = []
    
    # Center point (matches green circle)
    landmarks.append({"x": 0.5, "y": 0.5, "z": 0})
    
    # "Hairline" points (match blue line)
    # 5 points across
    for i in range(5):
        landmarks.append({"x": 0.1 + (i * 0.2), "y": 0.05, "z": 0})
        
    # Corner markers (Exact corners)
    landmarks.append({"x": 0.0, "y": 0.0, "z": 0})
    landmarks.append({"x": 1.0, "y": 1.0, "z": 0})
    
    # 4. Serialize Image to Base64
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    data_url = f"data:image/png;base64,{img_b64}"
    
    return {
        "image_url": data_url,
        "analysis": {
            "landmarks": landmarks,
            "overall": {
                "harmonyScore": 100,
                "symmetryScore": 100,
                "verdict": "CALIBRATION MODE: CHECK ALIGNMENT"
            },
            # Mock other data to prevent frontend crash
            "proportions": {"phiRatio": 1.618, "fifthsScore": 100, "thirds": {"upper":33,"mid":33,"lower":33}, "lowerThirdRatio": 2.0, "fWHR": 2.0},
            "eyes": {"canthalTilt": 10, "eyeSpacingRatio": 0.5, "eyeAreaRatio": 1.5, "harmonyIndex": 100},
            "nose": {"mouthNoseRatio": 1.618, "score": 100, "noseWidthRatio": 0.25, "symmetry": 100},
            "jawline": {"gonialAngle": 110, "jawToCheekRatio": 1.0, "symmetry": 100, "lowerThirdRatio": 2.0},
            "detailed": {}
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
