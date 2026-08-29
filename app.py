# server.py
import os
import sys
import socket
import time
import uuid
import torch
import httpx
import logging
import warnings
from pathlib import Path

from fastapi.responses import HTMLResponse
from gradio import Server
from gradio.data_classes import FileData

# Your inference code
from modules.inference import run_inference_script

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

# Configure logging
for l in ["torch", "faiss", "omegaconf", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)

app = Server()
# NOTE: model caching is handled in modules/inference.py (_CONVERTER_CACHE),
# so each .pth model is loaded into memory only once across requests.

@app.api(name="convert_audio")
def convert_audio(
    audio_file: FileData,
    model_path: str,
    index_path: str = "",
    pitch: int = 0,
    f0_method: str = "rmvpe",
    embedder_model: str = "hubert_base",
    index_rate: float = 0.5,
    volume_envelope: float = 1.0,
    protect: float = 0.33,
    hop_length: int = 128,
    filter_radius: int = 3,
    export_format: str = "wav",
    resample_sr: int = 0,
    f0_autotune: bool = False,
    f0_autotune_strength: float = 1.0,
    split_audio: bool = False,
    clean_audio: bool = False,
    clean_strength: float = 0.5,
    is_half: bool = False,
    cpu_mode: bool = False
) -> FileData:
    """
    Convert audio using RVC model.
    Returns the converted audio file.
    """
    try:
        # Prepare input and output paths (unique name avoids collisions between requests)
        input_path = audio_file["path"]
        stem = input_path.rsplit(".", 1)[0]
        output_path = f"{stem}_converted_{uuid.uuid4().hex[:8]}.{export_format}"
        
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
            input_path=input_path,
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
        
        if not os.path.exists(output_path):
            raise RuntimeError("Conversion failed: no output file was produced")

        return FileData(path=output_path)
    except Exception as e:
        raise RuntimeError(f"Conversion failed: {e}")

@app.api(name="list_models")
def list_models() -> list:
    """List available RVC models from rvc_models directory."""
    models = []
    model_dir = Path("rvc_models")
    if model_dir.exists():
        models = sorted([str(p) for p in model_dir.rglob("*.pth")])
    return models

@app.api(name="find_index")
def find_index(model_path: str) -> str:
    """Find corresponding index file for a model."""
    model_dir = Path(model_path).parent
    for f in model_dir.iterdir():
        if f.suffix == ".index" and "trained" not in f.name:
            return str(f)
    return ""

@app.get("/")
async def get_frontend():
    """Serve the custom HTML frontend."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>RVC Voice Converter</h1><p>Frontend file not found.</p>")

def wait_for_share_server(max_wait_s: int = 300, delay_s: int = 15) -> bool:
    """Gradio's share server is occasionally unreachable ("Could not create share link"),
    which makes launch() silently fall back to a local-only URL that Colab users cannot
    open. The outage is transient, so wait for it to come back before launching."""
    deadline = time.time() + max_wait_s
    while True:
        try:
            # Same endpoint gradio itself uses to get the share (frpc) tunnel server
            payload = httpx.get(
                "https://api.gradio.app/v3/tunnel-request", timeout=30
            ).json()[0]
            with socket.create_connection(
                (payload["host"], int(payload["port"])), timeout=10
            ):
                return True
        except Exception:
            if time.time() >= deadline:
                return False
            print(f"[WARNING] Gradio share server unreachable, retrying in {delay_s}s ...")
            time.sleep(delay_s)

# Run the server
if __name__ == "__main__":
    wait_s = int(os.getenv("SHARE_WAIT_S", "300"))
    if not wait_for_share_server(max_wait_s=wait_s):
        print("[WARNING] Share server still unreachable — launching anyway (local URL only).")
    app.launch(show_error=True, server_name="0.0.0.0", server_port=7860, share=True)
