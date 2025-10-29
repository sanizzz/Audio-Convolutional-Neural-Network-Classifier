# Audio CNN Classifier

End-to-end audio understanding system that trains a residual convolutional neural network on ESC-50, serves a FastAPI inference endpoint on Modal, and visualizes predictions plus feature maps in a Next.js dashboard.

## Repository Layout

- `train.py`: Modal training job that prepares data, trains the CNN, logs metrics, and snapshots the best checkpoint.
- `model.py`: PyTorch implementation of the residual audio CNN with feature map extraction.
- `main.py`: Modal inference app exposing a FastAPI endpoint and a local smoke test.
- `audio-cnn-visual/`: Next.js 15 frontend that uploads audio, calls the inference API, and renders spectrograms, feature maps, and confidence scores.
- `requirements.txt`: Shared Python dependencies for training and inference images.
- `chirpingbirds.wav`: Sample clip for manual checks.

## Model Architecture (`model.py`)

- Stem: `Conv2d(1, 64, kernel=7, stride=2)` + `BatchNorm` + ReLU + `MaxPool`, lifting a single-channel Mel spectrogram to 64 feature maps.
- Residual stages:
  - `layer1`: 3 blocks at 64 channels.
  - `layer2`: 4 blocks, first block downsamples to 128 channels.
  - `layer3`: 6 blocks, first block downsamples to 256 channels.
  - `layer4`: 3 blocks, first block downsamples to 512 channels.
  Each `ResidualBlock` stacks two 3x3 convolutions with identity shortcuts or 1x1 projection shortcuts when the spatial size or channel count changes.
- Classification head: adaptive average pooling -> dropout (0.5) -> linear layer sized to the number of ESC-50 labels.
- Feature taps: when `return_feature_maps=True`, the forward pass collects activations for every stage and residual block (for example `layer2.block0.relu`), enabling the UI to draw heatmaps.

## Data and Audio Processing (`train.py`)

- Dataset: ESC-50 (2,000 five second clips across 50 environmental categories). The custom Modal image downloads and caches the dataset under `/opt/esc50-data`, then persists it in a volume named `esc50-data`.
- Splits: folds 1 through 4 for training, fold 5 for validation.
- Time-frequency representation: `torchaudio` Mel spectrogram with 128 mel bins, 1,024 point FFT, hop length 512, 0 to 11,025 Hz pass band, followed by amplitude-to-decibel scaling.
- Augmentations: frequency masking, time masking, MixUp (Beta(0.2, 0.2) with 30 percent probability per batch), and label smoothing (0.1).
- Normalisation: stereo waveforms averaged to mono; tensors stay in decibel space and rely on batch norm for scaling.

## Training Loop (`train.py`)

- Execution: Modal function scheduled on an A10G GPU with the pre-built image.
- Optimizer: AdamW (lr=5e-4, weight_decay=0.01) with a `OneCycleLR` scheduler that peaks at lr=2e-3 over 100 epochs.
- Data loaders: batch size 32 with shuffling for training, deterministic iteration for validation.
- Logging: per epoch loss and accuracy to stdout and TensorBoard (`/models/tensorboard_logs/run_<timestamp>` inside the `esc-model` volume). Pull the volume locally and run `tensorboard --logdir ./tenserboard_logs`.
- Checkpointing: saves `/models/best_model.pth` whenever validation accuracy improves, bundling weights, epoch, accuracy, and the ordered class list.

## Inference Service (`main.py`)

- Lifecycle: Modal class `AudioClassifier` loads `best_model.pth` on container start. The same code path works locally on CPU for quick tests.
- Input options: multipart upload with `file=@clip.wav` or JSON payload `{"audio_data": "<base64>"}`.
- Preprocessing: convert stereo to mono, resample to 44,100 Hz, feed through the same Mel spectrogram transform used for validation, and expand to batch/channel dimensions.
- Outputs:
  - Top three predictions and confidences.
  - Original waveform samples, sample rate, and duration.
  - Input spectrogram tensor.
  - Aggregated feature maps for every stage and residual block (channel means to limit payload size).
- API details: POST `/inference` returns JSON; OPTIONS handles CORS preflight. Responses include permissive CORS headers so the Next.js app can call the endpoint from the browser.

## Frontend Visualizer (`audio-cnn-visual/`)

- Stack: Next.js 15 App Router, React 19, Tailwind CSS 4, Radix UI components, environment validation via `@t3-oss/env-nextjs`.
- Upload workflow: client-side file input posts audio to the URL in `NEXT_PUBLIC_INFERENCE_URL`.
- Visual elements:
  - Top prediction cards with ESC-50 themed emoji and progress bars.
  - Spectrogram heatmap for the model input and waveform plot generated from raw samples.
  - Feature map grid grouped by residual stage (`conv1`, `layer1`, `layer2`, `layer3`, `layer4`). Each map is normalized per tensor and rendered with a diverging color palette.
  - Shared color scale legend to interpret activation strength.
- UX niceties: loading indicator during inference, validation of missing files, and clear error cards on failed requests.

## Running the Project

### Prerequisites

- Python 3.10 or newer, `pip`, and the `modal` CLI authenticated with your Modal account.
- Node.js 18+ and npm 10+.
- Optional local tools: ffmpeg and libsndfile if you prototype audio workflows outside Modal.

### Python Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell
pip install -r requirements.txt
```

### Train the Model on Modal

```bash
modal run train.py::main
```

This builds the training image (Python deps plus wget, unzip, ffmpeg, libsndfile), hydrates ESC-50 once per volume, and launches a 100 epoch job on an A10G GPU. The `esc-model` volume collects checkpoints and TensorBoard runs. Inspect results locally with:

```bash
modal volume get esc-model /models/best_model.pth ./artifacts/best_model.pth
tensorboard --logdir ./tenserboard_logs
```

### Serve the Inference API

```bash
modal serve main.py
```

Modal prints a public URL for the `/inference` endpoint. For a quick local verification without the UI:

```bash
modal run main.py::main
```

The script uploads `chirpingbirds.wav`, hits the deployed endpoint, and prints the top predictions.

### API Contract

```bash
curl -X POST "<DEPLOYED_URL>/inference" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@chirpingbirds.wav"
```

Response (trimmed):

```json
{
  "predictions": [
    {"class": "chirping_birds", "confidence": 0.94},
    {"class": "rain", "confidence": 0.03},
    {"class": "insects", "confidence": 0.02}
  ],
  "waveform": {"values": [...], "sample_rate": 44100, "duration": 5.0},
  "visualization": {
    "conv1": {"shape": [64, 109], "values": [...]},
    "layer1.block0.relu": {"shape": [64, 109], "values": [...]}
  }
}
```

### Run the Next.js Dashboard

1. Copy `audio-cnn-visual/.env.example` to `.env.local` and set `NEXT_PUBLIC_INFERENCE_URL` to the URL from `modal serve`.
2. Install dependencies and start the dev server:

   ```bash
   cd audio-cnn-visual
   npm install
   npm run dev
   ```

3. Visit http://localhost:3000, upload a WAV file (long clips are downsampled to about 8,000 samples for plotting), and explore predictions plus intermediate activations.

### Useful Assets

- `chirpingbirds.wav`: quick end-to-end smoke test.
- `tenserboard_logs/`: TensorBoard runs synced from Modal with `modal volume get`.
- Modal volumes: `esc50-data` (dataset cache) and `esc-model` (checkpoints and logs).

## Talking Points for Demos

- **Residual CNNs for audio:** deep skip-connected stacks capture both short transients (chirps, knocks) and broader textures (rain, engines) on spectrograms.
- **Regularization stack:** MixUp, SpecAugment style masks, label smoothing, dropout, and AdamW keep the model from memorizing the 2,000 training clips.
- **Observability:** TensorBoard traces and exported feature maps make it easy to explain what each layer is attending to.
- **Scalability:** Modal packages dependencies, caches ESC-50, and provisions GPUs only while training or serving.
- **Interactive storytelling:** The Next.js dashboard translates logits and feature maps into visuals stakeholders can interpret immediately.

## Next Steps (Optional Enhancements)

1. Add automated regression tests that hit `/inference` with fixture audio clips.
2. Track hyperparameters and figures of merit in a lightweight experiment tracker or Modal metadata.
3. Integrate Grad-CAM or similar attribution overlays on the spectrogram to explain individual predictions.
