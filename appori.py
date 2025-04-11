import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import librosa
import soundfile as sf
import io
from infer import predict_from_audio
from plotting import plot_colored_waveform

# Inject custom CSS for maximum readability and elegance
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 26px !important;
        font-family: Georgia, serif !important;
    }
    .stButton > button {
        font-size: 28px !important;
        padding: 1.2em 2.5em !important;
        border-radius: 14px !important;
    }
    .stTextInput > div > div > input,
    .stFileUploader > div > div {
        font-size: 26px !important;
    }
    .stAlert-success {
        font-size: 26px !important;
        font-weight: bold;
    }
    .stAlert-error,
    .stAlert-info {
        font-size: 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.title("🎙 Abnormal Speech Detection")

st.markdown("Choose a method to provide your voice:")

# Option 1: Record audio in-browser
st.subheader("Option 1: Record your voice")
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000)

# Option 2: Upload existing audio file
st.subheader("Option 2: Upload an audio file")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# Store the audio data and file path
audio_data = None
audio_file = None

# Reset button
if st.button("🔄 Reset"):
    audio_bytes = None
    uploaded_file = None
    audio_data = None
    audio_file = None

# If audio is recorded or uploaded, save it
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    audio_file = "recorded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)
    audio_data = audio_file

elif uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    audio_file = "uploaded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_data = audio_file

# Button to trigger prediction
if audio_data and st.button("🧠 Infer"):
    with st.spinner("Analyzing..."):
        try:
            prediction_result = predict_from_audio(audio_data)
            st.success(f"Prediction Result: {prediction_result['final_prediction']}")
            datatemp = prediction_result["clips_preds"]
            st.success(f"Per-clip predictions: {datatemp}")
            fig = plot_colored_waveform(audio_data, datatemp)
            fig.update_layout(height=700, width=1100)  # Larger plot in both dimensions
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
else:
    if audio_data:
        st.info("Click 'Infer' to process the audio.")
    else:
        st.info("Please record your voice or upload an audio file.")
