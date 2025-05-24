import torch
import torch.nn.functional as F
import librosa
import numpy as np
from model import CNNVoiceDetector
SAMPLE_RATE = 16000
DURATION = 3
def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=SAMPLE_RATE * DURATION)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 2 - 1
    mel_db = mel_db[np.newaxis, np.newaxis, :, :]
    return torch.tensor(mel_db, dtype=torch.float32)
if __name__ == "__main__":
    audio_path = input("🎤 Enter the path of the audio file (.wav): ").strip()
    model = CNNVoiceDetector()
    model.load_state_dict(torch.load('best_voice_detector_model.pth', map_location=torch.device('cpu')))
    model.eval()
    input_tensor = extract_mel(audio_path)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)
        label = "Real Human" if prediction.item() == 1 else "Fake AI Voice"
        print(f"\n🧠 Prediction: {label} (Confidence: {confidence.item()*100:.1f}%)")
        fake_prob = probs[0][0].item() * 100
        real_prob = probs[0][1].item() * 100
        print(f"🔍 Detailed probabilities:")
        print(f"- Fake AI Voice: {fake_prob:.1f}%")
        print(f"- Real Human: {real_prob:.1f}%")