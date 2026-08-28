import os
import shutil
import tempfile
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import warnings
import sys
sys.path.append(os.getcwd())

warnings.filterwarnings("ignore")

from modules.inference import run_inference_script

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RVC Inference API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("rvc_models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

class InferenceParams(BaseModel):
    pitch: int = 0
    filter_radius: int = 3
    index_rate: float = 0.5
    volume_envelope: float = 1.0
    protect: float = 0.5
    hop_length: int = 128
    f0_method: str = "rmvpe"
    embedder_model: str = "hubert_base"
    export_format: str = "wav"
    resample_sr: int = 0
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    split_audio: bool = False
    clean_audio: bool = False
    clean_strength: float = 0.5
    is_half: bool = False
    cpu_mode: bool = False

@app.get("/")
async def root():
    return {"message": "RVC Inference API", "status": "running"}

@app.post("/convert")
async def convert_audio(
    audio: UploadFile = File(...),
    model_path: str = Form(...),
    index_path: Optional[str] = Form(None),
    pitch: int = Form(0),
    filter_radius: int = Form(3),
    index_rate: float = Form(0.5),
    volume_envelope: float = Form(1.0),
    protect: float = Form(0.5),
    hop_length: int = Form(128),
    f0_method: str = Form("rmvpe"),
    embedder_model: str = Form("hubert_base"),
    export_format: str = Form("wav"),
    resample_sr: int = Form(0),
    f0_autotune: bool = Form(False),
    f0_autotune_strength: float = Form(1.0),
    split_audio: bool = Form(False),
    clean_audio: bool = Form(False),
    clean_strength: float = Form(0.5),
    is_half: bool = Form(False),
    cpu_mode: bool = Form(False)
):
    """
    Convert audio using RVC model
    """
    try:
        # Validate model exists
        if not os.path.exists(model_path) or not model_path.endswith(".pth"):
            raise HTTPException(status_code=400, detail="Invalid model path")

        # Save uploaded audio
        audio_filename = f"upload_{os.urandom(8).hex()}.{audio.filename.split('.')[-1]}"
        audio_path = os.path.join("uploads", audio_filename)
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Generate output path
        output_filename = f"output_{os.urandom(8).hex()}.{export_format}"
        output_path = os.path.join("outputs", output_filename)

        logger.info(f"Processing audio: {audio_path}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Index: {index_path}")
        
        # Run inference
        run_inference_script(
            is_half=is_half,
            cpu_mode=cpu_mode,
            pitch=pitch,
            filter_radius=filter_radius,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            protect=protect,
            hop_length=hop_length,
            f0_method=f0_method,
            input_path=audio_path,
            output_path=output_path,
            pth_path=model_path,
            index_path=index_path,
            export_format=export_format,
            embedder_model=embedder_model,
            resample_sr=resample_sr,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            split_audio=split_audio,
            clean_audio=clean_audio,
            clean_strength=clean_strength
        )

        # Check if output was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Conversion failed - no output file generated")

        # Return the converted file
        return FileResponse(
            path=output_path,
            media_type=f"audio/{export_format}",
            filename=f"converted_{os.path.basename(audio.filename)}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if 'audio_path' in locals() and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

@app.get("/models")
async def list_models():
    """List available RVC models"""
    models = []
    for root, _, files in os.walk("rvc_models"):
        for file in files:
            if file.endswith(".pth"):
                rel_path = os.path.relpath(os.path.join(root, file), "rvc_models")
                models.append({
                    "path": os.path.join(root, file),
                    "name": file,
                    "directory": root
                })
    
    # Also check for index files
    for model in models:
        model_dir = os.path.dirname(model["path"])
        for f in os.listdir(model_dir):
            if f.endswith(".index") and "trained" not in f:
                model["index_path"] = os.path.join(model_dir, f)
                break
    
    return {"models": models}

@app.post("/refresh_models")
async def refresh_models():
    """Refresh model list cache"""
    # Just return updated list
    return await list_models()

@app.get("/f0_methods")
async def get_f0_methods():
    return {
        "methods": [
            "pm", "dio", "mangio-crepe-tiny", "mangio-crepe-small",
            "mangio-crepe-medium", "mangio-crepe-large", "mangio-crepe-full",
            "crepe-tiny", "crepe-small", "crepe-medium", "crepe-large",
            "crepe-full", "fcpe", "fcpe-legacy", "rmvpe", "rmvpe-legacy",
            "harvest", "yin", "pyin", "swipe"
        ]
    }

@app.get("/embedders")
async def get_embedders():
    return {
        "embedders": [
            "contentvec_base", "hubert_base", "japanese_hubert_base",
            "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
