import streamlit as st
from audio_recorder_streamlit import audio_recorder
from infer import predict_from_audio
from plotting import plot_colored_waveform

# Set wide layout
st.set_page_config(
    page_title="Abnormal Speech Detection",
    layout="wide"
)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Georgia', serif;
        font-size: 1.2rem;
        background-color: #f8f9fa;
        color: #1a1a1a;
        overflow-x: hidden;
    }

    @media (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 1rem;
        }
        h1 {
            font-size: 2rem !important;
        }
        h2 {
            font-size: 1.4rem !important;
        }
        .block-container {
            padding: 1.2rem 1rem !important;
            max-width: 95% !important;
        }
        .result-box {
            font-size: 1rem !important;
        }
        .stButton > button {
            font-size: 1rem !important;
            padding: 0.5rem 1rem !important;
        }
        .audio-recorder-container svg {
            width: 60px !important;
            height: 60px !important;
        }
    }

    .block-container {
        padding: 2rem 3rem !important;
        max-width: 60%;
    }

    h1 {
        font-size: 3rem;
        color: #101820;
        font-weight: 700;
        margin-bottom: 0.5em;
    }

    h2 {
        font-size: 2rem;
        color: #1a1a1a;
        font-weight: 600;
        margin-top: 1.5em;
    }

    .glass-box {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .glass-sep {
        height: 2px;
        background: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255,255,255,0.6), rgba(255, 255, 255, 0));
        border-radius: 1px;
        margin: 3rem 0;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(2px);
    }

    .result-box {
        font-size: 1.25rem;
        background: #ffffff;
        color: #111;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        border-radius: 14px;
        font-weight: 600;
        border-left: 6px solid #4b6cb7;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }

    .stButton > button {
        font-size: 1.25rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        border: none;
        transition: transform 0.2s ease-in-out, background 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #2575fc, #6a11cb);
        color: pink;
        transform: scale(1.05);
    }

    .vertical-glass-separator {
        width: 2px;
        height: 100%;
        background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0.6), rgba(255,255,255,0));
        border-radius: 1px;
        box-shadow: 0 0 10px rgba(255,255,255,0.2);
        backdrop-filter: blur(2px);
        margin: 0 1.5rem;
    }

    footer {
        text-align: center;
        font-size: 1rem;
        margin-top: 3rem;
        color: #666;
    }

    .glass-base {
        height: 4px;
        width: 60%;
        background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.7), rgba(255,255,255,0));
        border-radius: 2px;
        margin: 3rem 0;
        box-shadow: 0 0 15px rgba(255,255,255,0.4), 0 2px 10px rgba(0,0,0,0.1);
        backdrop-filter: blur(3px);
        transition: all 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)
# Custom CSS
# st.markdown("""
#     <style>
#     html, body, [class*="css"] {
#         font-family: 'Georgia', serif;
#         font-size: 22px;
#         background-color: #f8f9fa;
#         color: #1a1a1a;
#         overflow-x: hidden;
#     }

#     .block-container {
#         padding: 2rem 3rem !important;
#         max-width: 60%;
#     }

#     h1 {
#         font-size: 48px;
#         color: #101820;
#         font-weight: 700;
#         margin-bottom: 0.5em;
#     }

#     h2 {
#         font-size: 32px;
#         color: #1a1a1a;
#         font-weight: 600;
#         margin-top: 1.5em;
#     }

#     .glass-box {
#         background: rgba(255,255,255,0.95);
#         border-radius: 20px;
#         padding: 2rem 3rem;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
#     }
#     .glass-sep {
#     height: 2px;
#     background: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255,255,255,0.6), rgba(255, 255, 255, 0));
#     border-radius: 1px;
#     margin: 3rem 0;
#     box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
#     backdrop-filter: blur(2px);
# }


#     .result-box {
#         font-size: 24px;
#         background: #ffffff;
#         color: #111;
#         padding: 1.25rem 1.5rem;
#         margin-top: 1.5rem;
#         border-radius: 14px;
#         font-weight: 600;
#         border-left: 6px solid #4b6cb7;
#         word-wrap: break-word;
#         overflow-wrap: break-word;
#         white-space: pre-wrap;
#     }

#     .stButton > button {
#         font-size: 22px;
#         padding: 0.75rem 2rem;
#         border-radius: 12px;
#         background: linear-gradient(to right, #6a11cb, #2575fc);
#         color: white;
#         border: none;
#         transition: transform 0.2s ease-in-out, background 0.3s;
#     }

#     .stButton > button:hover {
#         background: linear-gradient(to right, #2575fc, #6a11cb);
#         color: pink; /* Font color on hover */
#         transform: scale(1.05);
#     }

#     .vertical-separator {
#         border-left: 2px solid #cccccc;
#         height: 100%;
#         margin: auto 1rem;
#     }

#     footer {
#         text-align: center;
#         font-size: 18px;
#         margin-top: 3rem;
#         color: #666;
#     }

#     .audio-recorder-container svg {
#         width: 80px !important;
#         height: 80px !important;
#     }
#             .glass-base {
#     height: 4px;
#     width: 60%;
#     background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.7), rgba(255,255,255,0));
#     border-radius: 2px;
#     margin: 3rem 0;
#     box-shadow: 0 0 15px rgba(255,255,255,0.4), 0 2px 10px rgba(0,0,0,0.1);
#     backdrop-filter: blur(3px);
#     transition: all 0.3s ease-in-out;
# }

#             .vertical-glass-separator {
#     width: 2px;
#     height: 100%;
#     background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0.6), rgba(255,255,255,0));
#     border-radius: 1px;
#     box-shadow: 0 0 10px rgba(255,255,255,0.2);
#     backdrop-filter: blur(2px);
#     margin: 0 1.5rem;
# }

#     </style>
# """, unsafe_allow_html=True)

# Header
st.markdown("<h1>🎙 Foreign Object Detection</h1>", unsafe_allow_html=True)
st.markdown("Detect abnormalities in your voice using an AI-based model.")

# Step 1: Record or Upload
with st.container():
    st.markdown('<div class="glass-base">', unsafe_allow_html=True)
    st.markdown("<h2>Step 1️⃣: Provide Your Voice</h2>", unsafe_allow_html=True)

    col1, col_space, col2 = st.columns([3, 1, 3])
    
    with col1:
        st.markdown("**🎤 Record Your Voice**", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="audio-recorder-container">', unsafe_allow_html=True)
            audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=16000)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_space:
        st.markdown('<div class="vertical-glass-separator">', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📁 Upload a WAV File**")
        uploaded_file = st.file_uploader("Drag and drop or browse", type=["wav"])

    st.markdown("</div>", unsafe_allow_html=True)

# Step 2: Preview
audio_data = None
audio_file = None
with st.container():
    st.markdown('<div class="glass-sep">', unsafe_allow_html=True)
    st.markdown("<h2>Step 2️⃣: Preview</h2>", unsafe_allow_html=True)

    if st.button("🔄 Reset All"):
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
    else:
        st.info("Please record or upload a voice sample.")

    st.markdown("</div>", unsafe_allow_html=True)

# Step 3: Inference
with st.container():
    st.markdown('<div class="glass-sep">', unsafe_allow_html=True)
    st.markdown("<h2>Step 3️⃣: Run Analysis</h2>", unsafe_allow_html=True)

    if audio_data:
        if st.button("🧠 Run Inference"):
            with st.spinner("Analyzing audio..."):
                try:
                    result = predict_from_audio(audio_data)

                    # Display final prediction
                    st.markdown(f"<div class='result-box'>🟢 Final Prediction:\n<strong>{result['final_prediction']}</strong></div>", unsafe_allow_html=True)

                    # Display per-clip predictions
                    st.markdown(f"<div class='result-box'>📋 Confidence in Prediction:\n{round(result['max_confidence']*100,3)} %</div>", unsafe_allow_html=True)

                    # Plot waveform
                    fig = plot_colored_waveform(audio_data, result["clips_preds"])
                    fig.update_layout(height=600, width=800)
                    st.plotly_chart(fig, use_container_width=True)

                    # Replay audio after plot
                    st.audio(audio_data, format="audio/wav")

                except Exception as e:
                    st.markdown(f"<div class='result-box' style='border-left-color:#d32f2f;'>❌ Error: {e}</div>", unsafe_allow_html=True)
        else:
            st.info("Click the button to start analysis.")
    else:
        st.info("No valid audio file to process.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<footer>✨</footer>", unsafe_allow_html=True)
#change graph labels done
