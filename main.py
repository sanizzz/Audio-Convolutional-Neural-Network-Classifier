import io
from pathlib import Path

import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
import soundfile as sf
import librosa
from fastapi import File, HTTPException, Response, UploadFile, status
from fastapi.responses import JSONResponse

from model import AudioCNN

app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()

        waveform = waveform.unsqueeze(0)

        spectrogram = self.transform(waveform)

        return spectrogram.unsqueeze(0)


@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading models on enter")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load('/models/best_model.pth',
                                map_location=self.device)
        self.classes = checkpoint['classes']

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    @modal.fastapi_endpoint(route="/api/predict", method="POST")
    async def inference(self, file: UploadFile = File(...)):
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded.",
            )

        try:
            audio_bytes = await file.read()
        except Exception as exc:
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

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            try:
                audio_data = librosa.resample(
                    y=audio_data, orig_sr=sample_rate, target_sr=44100
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to resample audio.",
                ) from exc
            sample_rate = 44100

        try:
            spectrogram = self.audio_processor.process_audio_chunk(audio_data)
            spectrogram = spectrogram.to(self.device)

            with torch.no_grad():
                output, feature_maps = self.model(
                    spectrogram, return_feature_maps=True
                )

            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)

            predictions = [
                {"class": self.classes[idx.item()], "confidence": prob.item()}
                for prob, idx in zip(top3_probs, top3_indices)
            ]

            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:  # [batch_size, channels, height, width]
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
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model inference failed.",
            ) from exc

        response = {
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

        return JSONResponse(
            content=response,
            headers=CORS_HEADERS,
        )

    @modal.fastapi_endpoint(route="/api/predict", method="OPTIONS")
    async def options(self):
        return Response(
            status_code=204,
            headers=CORS_HEADERS,
        )


@app.local_entrypoint()
def main(file_path: str = "chirpingbirds.wav"):
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Sample file not found at {audio_path.resolve()}. "
            "Pass --file-path <path-to-wav> when running `modal run main.py`."
        )

    audio_data, sample_rate = sf.read(audio_path)

    server = AudioClassifier()
    url = server.inference.get_web_url()

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    files = {"file": ("sample.wav", buffer, "audio/wav")}
    response = requests.post(url, files=files, timeout=60)
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
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f'Duration: {waveform_info.get("duration", 0)}')

    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f'  -{pred["class"]} {pred["confidence"]:0.2%}')
