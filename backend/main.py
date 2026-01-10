from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine.morphology import MorphologyEngine
from engine.rarity import calculate_rarity
from engine.market_fit import calculate_market_fit, calculate_potential
import uvicorn

app = FastAPI(title="Morphology Scout API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine (Singleton-ish)
morphology_engine = MorphologyEngine()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Morphology Scout Engine"}

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        contents = await file.read()
        
        # 1. Morphology
        morphology_metrics = morphology_engine.process_image(contents)
        
        if not morphology_metrics:
             raise HTTPException(status_code=422, detail="No face detected or image unclear.")
             
        # 2. Statistical Analysis
        rarity_report = calculate_rarity(morphology_metrics)
        market_fit_report = calculate_market_fit(morphology_metrics)
        potential_report = calculate_potential(morphology_metrics)
        
        return {
            "morphology": morphology_metrics, # Return directly, no nesting
            "rarity": rarity_report,
            "marketFit": market_fit_report,
            "potential": potential_report
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
