from audio_recorder_streamlit import audio_recorder
import tempfile
import soundfile as sf
import numpy as np
import io
import librosa
import os
import whisper
import joblib
import streamlit as st
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import preprocess_audio
from src.extract_audio_features import extract_audio_features
from src.extract_text_features import extract_text_features


st.set_page_config(page_title="Human-Likeness Speech Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Human-Likeness Speech Detector")
st.caption("Upload or record your voice to estimate how human-like it sounds.")

# ------------------ Load Models ------------------
def get_latest_model(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

@st.cache_resource
def load_models():
    text_model = joblib.load(get_latest_model("models/text_model_*.pkl"))
    text_scaler = joblib.load(get_latest_model("models/text_scaler_*.pkl"))
    acoustic_model = joblib.load(get_latest_model("models/acoustic_model_*.pkl"))
    acoustic_scaler = joblib.load(get_latest_model("models/acoustic_scaler_*.pkl"))
    return text_model, text_scaler, acoustic_model, acoustic_scaler

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

text_model, text_scaler, acoustic_model, acoustic_scaler = load_models()
whisper_model = load_whisper()

# ------------------ Input Section ------------------
st.markdown("### 🎤 Record or Upload Your Voice")
st.write("You can either **record** your voice right now or **upload** an audio file to analyze.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 🎙️ Record Voice")
    st.caption("Click the button below to start recording. Speak for a few seconds, then click again to stop.")
    recorded_audio = audio_recorder(
        text="",             # hide the built-in text
        recording_color="#ff4b4b",  # red indicator while recording
        neutral_color="#4CAF50",    # green idle button
        icon_name="microphone",     # mic icon only
        icon_size="3x"
    )
    if recorded_audio:
        st.audio(recorded_audio, format="audio/wav")
        st.success("✅ Recording captured successfully!")

with col2:
    st.markdown("#### 📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (wav, mp3, m4a, flac):",
        type=["wav", "mp3", "m4a", "flac"],
        label_visibility="collapsed"
    )

# Decide which source to use
audio_data = None
source_label = ""
if recorded_audio:
    audio_data = io.BytesIO(recorded_audio)
    source_label = "recorded"
elif uploaded_file:
    audio_data = uploaded_file
    source_label = "uploaded"

# ------------------ Analyze Button ------------------
st.markdown("---")
if audio_data:
    analyze_btn = st.button("🚀 Analyze Human-Likeness", use_container_width=True)
    if analyze_btn:
        st.info(f"Processing your **{source_label}** audio… please wait ⏳")
        # (rest of analysis logic continues below)
else:
    st.warning("Please record your voice or upload a file before analyzing.")

# ------------------ Process Button ------------------
if audio_data:
    if st.button("🚀 Analyze Human-Likeness"):
        st.info("⏳ Processing audio...")

        temp_dir = os.path.join(os.getcwd(), "web_app", "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        temp_input = os.path.join(temp_dir, "temp_input.wav")
        processed_path = os.path.join(temp_dir, "temp_processed.wav")

        # Save temp input
        data, sr = librosa.load(audio_data, sr=16000, mono=True)
        sf.write(temp_input, data, sr)

        # Preprocess
        preprocess_audio(temp_input, processed_path)

        # Transcribe
        transcript = whisper_model.transcribe(processed_path)["text"]

        # Extract features
        mel, audio_vec = extract_audio_features(processed_path)
        text_vec = extract_text_features(transcript)

        # Scale + predict
        text_scaled = text_scaler.transform([text_vec])
        audio_scaled = acoustic_scaler.transform([audio_vec])

        text_prob = text_model.predict_proba(text_scaled)[0][1]
        audio_prob = acoustic_model.predict_proba(audio_scaled)[0][1]
        combined_prob = 0.3 * text_prob + 0.7 * audio_prob
        human_percent = combined_prob * 100

        # Display results
        st.subheader("🧠 Human-Likeness Estimate")
        st.metric("Human-likeness", f"{human_percent:.2f}%")
        st.progress(int(human_percent))

        with st.expander("📊 Details"):
            st.write("Transcript:", transcript)
            st.write("Text prob:", round(text_prob, 3))
            st.write("Audio prob:", round(audio_prob, 3))
            st.write("Text features:", np.round(text_vec, 3).tolist())
            st.write("Audio features:", np.round(audio_vec, 3).tolist())
