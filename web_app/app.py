
from audio_recorder_streamlit import audio_recorder   # 🔴 NEW IMPORT

import os
import sys
import numpy as np
import streamlit as st
import whisper
import joblib
import glob

# ---- Allow imports from src/ ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_audio
from src.extract_audio_features import extract_audio_features
from src.extract_text_features import extract_text_features


# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Human-Likeness Detector", page_icon="🎧", layout="centered")
st.title("🎧 Human-Likeness Speech Detector")
st.caption("Upload an audio file to estimate **how human-like** it sounds. (0% = totally AI-like, 100% = fully human-like)")


# -------------------- Utility Functions --------------------
def get_latest_model(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


@st.cache_resource
def load_models():
    text_model_path = get_latest_model("models/text_model_*.pkl")
    text_scaler_path = get_latest_model("models/text_scaler_*.pkl")
    acoustic_model_path = get_latest_model("models/acoustic_model_*.pkl")
    acoustic_scaler_path = get_latest_model("models/acoustic_scaler_*.pkl")

    if not all([text_model_path, text_scaler_path, acoustic_model_path, acoustic_scaler_path]):
        st.error("❌ Models not found. Please train them first.")
        st.stop()

    text_model = joblib.load(text_model_path)
    text_scaler = joblib.load(text_scaler_path)
    acoustic_model = joblib.load(acoustic_model_path)
    acoustic_scaler = joblib.load(acoustic_scaler_path)

    return text_model, text_scaler, acoustic_model, acoustic_scaler


@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


text_model, text_scaler, acoustic_model, acoustic_scaler = load_models()
whisper_model = load_whisper()


# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])


# -------------------- Record Audio Section --------------------
st.markdown("### 🎙️ Record Your Voice (Optional)")
st.caption("Click below to start and stop recording")

if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

recorded_audio = audio_recorder(
    pause_threshold=25.0,  # keeps recording until 2 seconds of silence
    sample_rate=16000,
    energy_threshold=(-1.0, 1.0)
)

# If new audio is recorded, save it to session state
if recorded_audio:
    st.session_state.recorded_audio = recorded_audio

# Use the saved audio
if st.session_state.recorded_audio:
    st.audio(st.session_state.recorded_audio, format="audio/wav")

    record_path = os.path.join("web_app", "temp_files", "recorded_audio.wav")
    os.makedirs(os.path.dirname(record_path), exist_ok=True)

    # Save the recorded audio file
    with open(record_path, "wb") as f:
        f.write(st.session_state.recorded_audio)

    if st.button("🚀 Analyze Recorded Audio"):
        st.info("⏳ Processing recorded audio... Please wait...")
        temp_input = record_path
        processed_path = os.path.join("web_app", "temp_files", "temp_processed.wav")

        # Step 1 — Preprocess
        success = preprocess_audio(temp_input, processed_path)
        if not success or not os.path.exists(processed_path):
            st.error("❌ Failed to preprocess audio.")
            st.stop()

        # Step 2 — Transcribe
        try:
            transcript = whisper_model.transcribe(processed_path)["text"]
        except Exception as e:
            st.error(f"⚠️ Transcription failed: {e}")
            st.stop()

        # Step 3 — Extract Features
        mel, audio_vec = extract_audio_features(processed_path)
        text_vec = extract_text_features(transcript)

        if audio_vec is None or text_vec is None:
            st.error("⚠️ Feature extraction failed.")
            st.stop()

        # Step 4 — Scale Features
        text_scaled = text_scaler.transform([text_vec])
        audio_scaled = acoustic_scaler.transform([audio_vec])

        # Step 5 — Predict
        text_prob = text_model.predict_proba(text_scaled)[0][1]
        audio_prob = acoustic_model.predict_proba(audio_scaled)[0][1]
        combined_prob = 0.3 * text_prob + 0.7 * audio_prob
        human_percent = combined_prob * 100

        # Step 6 — Display Result
        st.subheader("🧠 Human-Likeness Estimate (Recorded Audio)")
        st.metric("Human-likeness", f"{human_percent:.2f}%")
        st.progress(int(human_percent))

        # Step 7 — Cleanup
        for f in [temp_input, processed_path]:
            try:
                os.remove(f)
            except Exception:
                pass
#-------------------------------------------------------------
if uploaded_file:
    st.audio(uploaded_file)
    process_btn = st.button("🚀 Analyze Human-Likeness")

    if process_btn:
        st.info("⏳ Processing audio... Please wait...")

        # Temporary paths
        temp_dir = os.path.join(os.getcwd(), "web_app", "temp_files")
        os.makedirs(temp_dir, exist_ok=True)

        temp_input = os.path.join(temp_dir, "temp_input.wav")
        processed_path = os.path.join(temp_dir, "temp_processed.wav")

        # Save uploaded file
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.read())

        # Step 1 — Preprocess
        success = preprocess_audio(temp_input, processed_path)
        if not success or not os.path.exists(processed_path):
            st.error("❌ Failed to preprocess audio.")
            st.stop()

        # Step 2 — Transcribe
        try:
            transcript = whisper_model.transcribe(processed_path)["text"]
        except Exception as e:
            st.error(f"⚠️ Transcription failed: {e}")
            st.stop()

        # Step 3 — Extract Features
        mel, audio_vec = extract_audio_features(processed_path)
        text_vec = extract_text_features(transcript)

        if audio_vec is None or text_vec is None:
            st.error("⚠️ Feature extraction failed.")
            st.stop()

        # Step 4 — Scale Features
        text_scaled = text_scaler.transform([text_vec])
        audio_scaled = acoustic_scaler.transform([audio_vec])

        # Step 5 — Predict
        text_prob = text_model.predict_proba(text_scaled)[0][1]
        audio_prob = acoustic_model.predict_proba(audio_scaled)[0][1]

        # Combine: weighted average (tuneable)
        combined_prob = 0.3 * text_prob + 0.7 * audio_prob
        human_percent = combined_prob * 100

        # Step 6 — Display Result
        st.subheader("🧠 Human-Likeness Estimate")
        st.metric("Human-likeness", f"{human_percent:.2f}%")
        st.progress(int(human_percent))
        
        with st.expander("📊 View Details"):
            st.write("### 🔤 Text Features (linguistic)")
            text_labels = [
                "Filler count (0)",
                "Repetition count (1)",
                "Phrase count (2)",
                "Average sentence length (3)"
            ]
            for label, value in zip(text_labels, np.round(text_vec, 3).tolist()):
                st.write(f"{label}: {value}")

            st.write("### 🎧 Audio Features (acoustic)")
            audio_labels = [
                "Pitch mean (0)",
                "Pitch std (1)",
                "Silence ratio (2)",
                "Energy variance (3)",
                "Speaking rate (4)"
            ]
            for label, value in zip(audio_labels, np.round(audio_vec, 3).tolist()):
                st.write(f"{label}: {value}")

            st.divider()
            st.write(f"**Text model human-prob:** {text_prob:.3f}")
            st.write(f"**Audio model human-prob:** {audio_prob:.3f}")


        # Step 7 — Cleanup
        for f in [temp_input, processed_path]:
            try:
                os.remove(f)
            except Exception:
                pass
