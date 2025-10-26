# using modal for training and inference 
import modal
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
import sys
from model import AudioCNN
import torch.optim  as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

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

class ESC50Dataset(Dataset):
    def __init__(self,data_dir,metadata_file,split="train",transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file) # error fixed used self.metadata_file instead of meta data
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]  #to split the data filter out the 5 cause we need only 4/5 training
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5] # for validation set

        self.classes = sorted(self.metadata['category'].unique()) #list of unique classes
        self.class_to_idx = {cls:idx for idx,cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)
    
    def __len__(self):#dunder method 
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform,sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform,dim=0,keepdim = True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row['label']


def mixup_data(x,y):
    lam = np.random.beta(0.2,0.2)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    #to create synthetic sample 30 percent of first audio clip plus 70 percent of second audio clip
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a,y_b = y,y[index]
    return mixed_x,y_a,y_b,lam

def mixup_criterion(criterion,pred,y_a,y_b,lam): #calc loss
    return lam * criterion(pred,y_a) + (1 - lam) * criterion(pred,y_b)



@app.function(image=image,gpu="A10G",volumes={"/data":data_volume,"/models":model_volume},timeout= 60 * 60 * 3)
def train():
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )
    
    train_dataset = ESC50Dataset(data_dir=esc50_dir,metadata_file=esc50_dir / "meta" / "esc50.csv",split="train", transform=train_transform)
    val_dataset = ESC50Dataset(data_dir=esc50_dir,metadata_file=esc50_dir / "meta" / "esc50.csv",split="val", transform=train_transform)

    # checking the training and validation data set
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # load data in batches to train nn
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # label smoothing helps model to be more humble in its predictions 
    optimizer = optim.AdamW(model.parameters(),lr=0.005,weight_decay=0.001)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr = 0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1
    )

    best_accuracy = 0.0
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader,desc=f"Epoch {epoch +1}/{num_epochs}")
        for data,target in progress_bar:
            data,target = data.to(device),target.to(device)

            if np.random.random() > 0.7:
                data,target_a, target_b, lam = mixup_data(data,target)
                output = model(data)
                loss = mixup_criterion(criterion,output,target_a,target_b,lam)
            else:
                output = model(data)
                loss = criterion(output,target)
            
            optimizer.zero_grad()
            loss.backward() #self adjust the weights for model improvement 
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        #validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data,target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(data)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data,1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)
        
        print(f'Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}, Val loss: {avg_val_loss:.4f}, Acurracy: {accuracy:.2f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f"New best model saved: {accuracy:.2f}")
    print(f'Training completed and the Best accuracy:{best_accuracy:.2f}%')


@app.local_entrypoint()
def main():
    train.remote()
