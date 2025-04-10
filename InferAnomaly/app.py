import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import librosa
import soundfile as sf
import io
from InferAnomaly.infer import predict_from_audio
  # Assuming your function is imported from 'infer.py'

st.set_page_config(page_title="Foreign Object Detection", layout="centered")
st.title("🎙 Foreign Object Detection")

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
if st.button("Reset"):
    audio_bytes = None
    uploaded_file = None
    audio_data = None
    audio_file = None
    

# If audio is recorded or uploaded, save it
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    # Save the recorded audio as a WAV file
    audio_file = "recorded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)
    audio_data = audio_file  # Set the audio_data to the saved file

elif uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    # Save the uploaded audio file as a WAV file
    audio_file = "uploaded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_data = audio_file  # Set the audio_data to the saved file

# Button to trigger prediction
if audio_data and st.button("Infer"):
    
    with st.spinner("Analyzing..."):
        try:
            # Call the predict_from_audio function
            prediction_result = predict_from_audio(audio_data)

            # Display the result from the prediction
            st.success(f"Prediction Result: {prediction_result["final_prediction"]}")
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
else:
    if audio_data:
        st.info("Click 'Infer' to process the audio.")
    else:
        st.info("Please record your voice or upload an audio file.")
