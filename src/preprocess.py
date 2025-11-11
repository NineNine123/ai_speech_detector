import os
import librosa
import soundfile as sf
import numpy as np

TARGET_DURATION = 20.0
SAMPLE_RATE = 16000

def preprocess_audio(input_path, output_path):
    """
    Loads, trims, normalizes, and saves a fixed-length 20s audio file.
    Always creates the output file, even if the directory doesn’t exist.
    """
    try:
        # Ensure output directory exists (handles cases like "./temp_processed.wav")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Pad or trim to 20 seconds
        target_len = int(TARGET_DURATION * SAMPLE_RATE)
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # Save file
        sf.write(output_path, audio, SAMPLE_RATE)
        return True

    except Exception as e:
        print(f"⚠️ Error processing {input_path}: {e}")
        return False


if __name__ == "__main__":
    preprocess_audio("test.wav", "temp_processed.wav")
