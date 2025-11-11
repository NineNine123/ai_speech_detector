import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.preprocess import preprocess_audio
from src.transcribe import transcribe_audio
from src.extract_audio_features import extract_audio_features
from src.extract_text_features import extract_text_features

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
TRANSCRIPT_PATH = "data/transcripts"
FEATURES_PATH = "features"
METADATA_FILE = "data/metadata.csv"


def process_all():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(TRANSCRIPT_PATH, exist_ok=True)
    os.makedirs(FEATURES_PATH, exist_ok=True)

    metadata = []
    audio_feature_list = []
    text_feature_list = []

    # Process both human and AI folders
    for label in ["human", "ai"]:
        input_dir = os.path.join(RAW_PATH, label)
        files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
        ]

        print(f"\n🎧 Processing {len(files)} {label.upper()} audio files...\n")

        for filename in tqdm(files, desc=f"{label.capitalize()}"):
            try:
                file_id = os.path.splitext(filename)[0]
                raw_path = os.path.join(input_dir, filename)
                processed_path = os.path.join(PROCESSED_PATH, label, f"{file_id}_processed.wav")
                transcript_path = os.path.join(TRANSCRIPT_PATH, label, f"{file_id}.txt")

                # Step 1 — Preprocess audio
                ok = preprocess_audio(raw_path, processed_path)
                if not ok:
                    continue

                # Step 2 — Transcribe to text
                transcript = transcribe_audio(processed_path, transcript_path)

                # Step 3 — Extract acoustic features
                mel_spec, audio_vec = extract_audio_features(processed_path)
                if audio_vec is None:
                    continue

                # Step 4 — Extract text features
                text_vec = extract_text_features(transcript)

                # Collect metadata + features
                metadata.append({
                    "file": filename,
                    "label": label,
                    "processed_path": processed_path,
                    "transcript_path": transcript_path
                })
                audio_feature_list.append(audio_vec)
                text_feature_list.append(text_vec)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

    # Save metadata
    df_meta = pd.DataFrame(metadata)
    os.makedirs("data", exist_ok=True)
    df_meta.to_csv(METADATA_FILE, index=False)

    # Save features
    np.save(os.path.join(FEATURES_PATH, "audio_features.npy"), np.array(audio_feature_list))
    df_text = pd.DataFrame(
        text_feature_list,
        columns=["filler", "repetition", "phrases", "avg_sentence_len"]
    )
    df_text.to_csv(os.path.join(FEATURES_PATH, "text_features.csv"), index=False)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n✅ All processing complete at {timestamp}!")
    print(f"   Metadata → {METADATA_FILE}")
    print(f"   Audio features → {FEATURES_PATH}/audio_features.npy")
    print(f"   Text features → {FEATURES_PATH}/text_features.csv\n")


if __name__ == "__main__":
    process_all()
