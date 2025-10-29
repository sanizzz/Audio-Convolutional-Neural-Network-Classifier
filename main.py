import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import librosa
import modal
import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import AudioCNN

app = modal.App("audio-cnn-inference")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["libsndfile1"])
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self) -> None:
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025,
            ),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data: np.ndarray) -> torch.Tensor:
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


def _prepare_waveform(audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != 44100:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=44100,
        )
        sample_rate = 44100

    return audio_data, sample_rate


def _run_inference(
    audio_data: np.ndarray,
    sample_rate: int,
    *,
    model: AudioCNN,
    classes: list[str],
    device: torch.device,
    audio_processor: AudioProcessor,
) -> Dict[str, Any]:
    audio_data, sample_rate = _prepare_waveform(audio_data, sample_rate)

    spectrogram = audio_processor.process_audio_chunk(audio_data)
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        output, feature_maps = model(spectrogram, return_feature_maps=True)

    output = torch.nan_to_num(output)
    probabilities = torch.softmax(output, dim=1)
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)

    predictions = [
        {"class": classes[idx.item()], "confidence": prob.item()}
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    viz_data: Dict[str, Any] = {}
    for name, tensor in feature_maps.items():
        if tensor.dim() == 4:  # [batch, channels, height, width]
            aggregated_tensor = torch.mean(tensor, dim=1)
            squeezed_tensor = aggregated_tensor.squeeze(0)
            numpy_array = squeezed_tensor.cpu().numpy()
            clean_array = np.nan_to_num(numpy_array)
            viz_data[name] = {
                "shape": list(clean_array.shape),
                "values": clean_array.tolist(),
            }

    spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
    clean_spectrogram = np.nan_to_num(spectrogram_np)

    max_samples = 8000
    waveform_sample_rate = sample_rate
    if len(audio_data) > max_samples:
        step = len(audio_data) // max_samples
        waveform_data = audio_data[::step]
    else:
        waveform_data = audio_data

    return {
        "predictions": predictions,
        "visualization": viz_data,
        "input_spectrogram": {
            "shape": list(clean_spectrogram.shape),
            "values": clean_spectrogram.tolist(),
        },
        "waveform": {
            "values": waveform_data.tolist(),
            "sample_rate": waveform_sample_rate,
            "duration": len(audio_data) / waveform_sample_rate,
        },
    }


def build_fastapi_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("/models/best_model.pth", map_location=device)
        classes = checkpoint["classes"]

        model = AudioCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        audio_processor = AudioProcessor()

        app.state.model = model
        app.state.device = device
        app.state.classes = classes
        app.state.audio_processor = audio_processor

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fastapi_app = FastAPI(lifespan=lifespan)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.post("/api/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
    ) -> JSONResponse:
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded.",
            )

        try:
            audio_bytes = await file.read()
        except Exception as exc:  # pragma: no cover - FastAPI handles details
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to read uploaded file.",
            ) from exc

        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        try:
            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported or corrupt audio file.",
            ) from exc

        state = request.app.state
        missing_attrs = [
            attr
            for attr in ("model", "classes", "device", "audio_processor")
            if not hasattr(state, attr)
        ]
        if missing_attrs:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded.",
            )

        try:
            response_payload = _run_inference(
                audio_data,
                sample_rate,
                model=state.model,
                classes=state.classes,
                device=state.device,
                audio_processor=state.audio_processor,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model inference failed.",
            ) from exc

        return JSONResponse(content=response_payload)

    return fastapi_app


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume},
    min_containers=1,
    timeout=600,
)
@modal.asgi_app()
def serve():
    return build_fastapi_app()


@app.local_entrypoint()
def main(file_path: str = "chirpingbirds.wav") -> None:
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Sample file not found at {audio_path.resolve()}. "
            "Pass --file-path <path-to-wav> when running `modal run main.py`.",
        )

    audio_data, sample_rate = sf.read(audio_path)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    url = serve.web_url
    if url.endswith("/"):
        url = url.rstrip("/")

    files = {"file": (audio_path.name, buffer, "audio/wav")}
    response = requests.post(f"{url}/api/predict", files=files, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(f"Request failed: {detail}") from exc

    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", [])
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f'Duration: {waveform_info.get("duration", 0)}')

    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f'  -{pred["class"]} {pred["confidence"]:0.2%}')
