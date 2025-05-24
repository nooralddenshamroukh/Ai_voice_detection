import torch
import librosa
import os
import numpy as np
from torch.utils.data import Dataset
import random
class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=16000, duration=3, augment=False):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.filepaths = []
        self.labels = []
        for label, folder in enumerate(["fake", "real"]):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.filepaths.append(os.path.join(folder_path, file))
                    self.labels.append(label)
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        y, sr = librosa.load(filepath, sr=self.sr)
        if self.augment and random.random() > 0.5:
            y = self.add_noise(y)
        if self.augment and random.random() > 0.5:
            y = self.pitch_shift(y, sr)
        y = librosa.util.fix_length(y, size=self.sr * self.duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min()) * 2 - 1
        log_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return log_mel, torch.tensor(label, dtype=torch.long)
    def add_noise(self, y):
        noise = np.random.normal(0, 0.005, len(y))
        return y + noise
    def pitch_shift(self, y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-1, 1))