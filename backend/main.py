from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine.morphology import MorphologyEngine
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
