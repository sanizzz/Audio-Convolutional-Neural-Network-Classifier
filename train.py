# using modal for training and inference 
import modal

app = modal.App("audio-CNN")

image = (# docker image 
    modal.Image.debian_slim()  # lightweight Debian base image
    .pip_install_from_requirements("requirements.txt")  # install Python packages
    .apt_install("wget", "unzip", "ffmpeg", "libsndfile1")  # system packages for audio processing
    .run_commands([  # shell commands to download & extract dataset
        "cd /tmp && wget https://github.com/karoldvl/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")  # include your local 'model' folder in the image
)

# Define Modal Volumes for persistent data
data_volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

@app.function(image=image,gpu="A10G",volumes={"/data":data_volume,"/models":model_volume},timeout= 60 * 60 * 3)
def train():
    print("Training")
    


@app.local_entrypoint()
def main():
    train.remote()
