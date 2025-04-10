
import torch
import librosa
import transformers
from transformers import pipeline
import numpy as np
import streamlit as st
import pandas as pd
import soundfile as sf
from datetime import datetime
import io
import pickle
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import torch.nn as nn
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import webrtcvad
# Load model and supporting components
class Wav2VecFeatureExtractor:
    def __init__(self, model_name='facebook/wav2vec2-base', duration=3):
        self.processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.duration = duration

    def extract_features(self, waveform, sr):
        features = self.processor(waveform, sampling_rate=sr, return_tensors='pt').input_values
        return features.squeeze(0)

class CustomWav2Vec2Classifier(torch.nn.Module):
    def __init__(self, hidden_dim=768, intermediate_dim=512, output_dim=2):
        super().__init__()
        self.wav2vec = transformers.Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim*2, 3, padding=1),
            torch.nn.BatchNorm1d(hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(hidden_dim*2, hidden_dim*4, 3, padding=1),
            torch.nn.BatchNorm1d(hidden_dim*4),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=hidden_dim*4, num_heads=8, dropout=0.2, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*4, intermediate_dim),
            torch.nn.BatchNorm1d(intermediate_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(intermediate_dim, intermediate_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(intermediate_dim//2, output_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            encoder_output = self.wav2vec(x).last_hidden_state
        cnn_input = encoder_output.transpose(1, 2)
        cnn_features = self.cnn_layers(cnn_input)
        attention_input = cnn_features.transpose(1, 2)
        attended_features, _ = self.self_attention(attention_input, attention_input, attention_input)
        pooled_features = attended_features.mean(dim=1)
        return self.classifier(pooled_features)
        
#@st.cache_data(persist="disk")
#def modelpath():
    #return hf_hub_download(repo_id="Mohor/Wav2Vec2Full", filename="custom_wav2vec2_model_full.pt")
model_path=pipeline("Mohor/Wav2Vec2Full")
#model_path = hf_hub_download(repo_id="Mohor/Wav2Vec2Full", filename="custom_wav2vec2_model_full.pt")
label_path = "label_encoder_full2.pkl"
max_length = 32007

device = torch.device("cpu")
model = CustomWav2Vec2Classifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

feature_extractor = Wav2VecFeatureExtractor()

vad = webrtcvad.Vad(3)  # Most aggressive mode
CLIP_DURATION = 3  # in seconds

def is_speech_clip(y_clip, sr):
    # Convert to 16-bit PCM format as required by webrtcvad
    y_int16 = (y_clip * 32768).astype(np.int16)
    frame_duration_ms = 30
    frame_size = int(sr * frame_duration_ms / 1000)

    for i in range(0, len(y_int16) - frame_size, frame_size):
        frame = y_int16[i:i+frame_size].tobytes()
        if vad.is_speech(frame, sample_rate=sr):
            return True
    return False

def predict_from_audio(audio_path, save_csv_path="clip_predictions.csv"):
    dysarthric_confidences = []
    with sf.SoundFile(audio_path) as f:
        
        original_sr = f.samplerate
        frame_size = CLIP_DURATION * original_sr
        total_frames = len(f)
        clip_preds = []
        
        clip_idx = 0

        while f.tell() < total_frames:
            
            remaining_frames = total_frames - f.tell()
            if remaining_frames < frame_size:
                 print("qqq")
                 break  # Skip incomplete chunk (optional)
            read_frames = min(frame_size, remaining_frames)

            audio_chunk = f.read(frames=read_frames, dtype='float32')
            if len(audio_chunk) < frame_size:
                break  # Skip incomplete chunk (optional)
            if audio_chunk.ndim > 1:
    # Convert stereo to mono by averaging across channels
                audio_chunk = np.mean(audio_chunk, axis=1)
            # print("okk")
            clip_start_sec = round(clip_idx * CLIP_DURATION, 2)
            clip_end_sec = round((clip_idx + 1) * CLIP_DURATION, 2)
            clip_idx += 1

            # Resample if needed
            if original_sr != 16000:
                audio_chunk = librosa.resample(audio_chunk, orig_sr=original_sr, target_sr=16000)
                sr = 16000
            else:
                sr = original_sr

            if not is_speech_clip(audio_chunk, sr):
                clip_preds.append({
                    "start_time": clip_start_sec,
                    "end_time": clip_end_sec,
                    "prediction": "non-speech",
                    "confidence": None
                })
                continue

            features = feature_extractor.extract_features(audio_chunk, sr)
            if features.size(0) < max_length:
                features = torch.cat([features, torch.zeros(max_length - features.size(0))], dim=0)
            else:
                features = features[:max_length]

            input_tensor = features.unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_label_idx = torch.argmax(probs).item()
                pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
                confidence = probs[0, pred_label_idx].item()

            if pred_label == "Dysarthric":
                dysarthric_confidences.append(confidence)

            clip_preds.append({
                "start_time": clip_start_sec,
                "end_time": clip_end_sec,
                "prediction": pred_label,
                "confidence": confidence
            })


    df = pd.DataFrame(clip_preds)
    filename = os.path.basename(audio_path).replace(".wav", "") + "_clip_preds.csv"
    df.to_csv(filename, index=False)
    
    final_prediction = "dysarthric" if dysarthric_confidences and max(dysarthric_confidences) > 0.5 else "normal"

    return {
        "final_prediction": final_prediction,
        "max_dysarthric_confidence": max(dysarthric_confidences) if dysarthric_confidences else 0.0,
        "clip_csv": filename
    }



# # Example usage
# response = predict_from_audio("/home/var/Desktop/Mohor/InferAnomaly/1.wav")
# print(f"Final prediction: {response['final_prediction']}")
# print(f"Max Dysarthric Confidence: {round(response['max_dysarthric_confidence'] * 100, 2)}%")
# print(f"Clip-level predictions saved at: {response['clip_csv']}")

# Replace with your actual top-level directories

# directories = [
#     "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Test",
#     "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Test(Hindi)",
#     "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Train",
#     "/home/var/Desktop/Mohor/InferAnomaly/SlurredSpeech-Dataset-Train(Hindi)"
# ]

# GROUND_TRUTH = "dysarthric"

# def evaluate_predictions(y_true, y_pred):
#     precision = precision_score(y_true, y_pred, pos_label=GROUND_TRUTH, zero_division=0)
#     recall = recall_score(y_true, y_pred, pos_label=GROUND_TRUTH, zero_division=0)
#     f1 = f1_score(y_true, y_pred, pos_label=GROUND_TRUTH, zero_division=0)
#     return precision, recall, f1

# overall_y_true = []
# overall_y_pred = []

# for top_dir in directories:
#     y_true = []
#     y_pred = []

#     print(f"\nProcessing top-level directory: {top_dir}")

#     for root, _, files in os.walk(top_dir):
#         for file_name in files:
#             if file_name.endswith(".flac"):
#                 file_path = os.path.join(root, file_name)
#                 try:
#                     with open(file_path, "rb") as f:
#                         response = predict_from_audio(f)
#                         prediction = response["final_prediction"]
#                 except Exception as e:
#                     print(f"Error processing {file_path}: {e}")
#                     continue

#                 y_pred.append(prediction)
#                 y_true.append(GROUND_TRUTH)

#     precision, recall, f1 = evaluate_predictions(y_true, y_pred)
#     print(f"Top Directory: {os.path.basename(top_dir)}")
#     print(f" Precision: {precision:.4f}")
#     print(f" Recall:    {recall:.4f}")
#     print(f" F1 Score:  {f1:.4f}")

#     overall_y_true.extend(y_true)
#     overall_y_pred.extend(y_pred)

# # Compute overall metrics
# overall_precision, overall_recall, overall_f1 = evaluate_predictions(overall_y_true, overall_y_pred)

# print("\n==== Overall Metrics ====")
# print(f" Overall Precision: {overall_precision:.4f}")
# print(f" Overall Recall:    {overall_recall:.4f}")
# print(f" Overall F1 Score:  {overall_f1:.4f}")
