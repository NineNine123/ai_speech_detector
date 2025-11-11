import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 20.0  # seconds

def extract_audio_features(audio_path):
    """
    Extracts main acoustic features:
      - Mel spectrogram (for deep models)
      - Pitch mean and std
      - Silence ratio
      - Energy variance
      - Speaking rate estimate (frames/sec)
    Returns:
      mel_spec_db (2D array), feature_vector (1D array)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Ensure uniform length (pad/trim to 20s)
        target_len = int(TARGET_DURATION * SAMPLE_RATE)
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))))

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Silence ratio
        rms = librosa.feature.rms(y=audio)[0]
        silence_thresh = 0.02 * np.max(rms)
        silence_ratio = np.sum(rms < silence_thresh) / len(rms)

        # Energy variance (humans vary more)
        energy_var = np.var(rms)

        # Speaking rate (roughly energy bursts per second)
        energy_peaks = np.sum(rms > silence_thresh)
        speaking_rate = energy_peaks / TARGET_DURATION

        # Final compact feature vector
        feature_vector = np.array([
            pitch_mean,
            pitch_std,
            silence_ratio,
            energy_var,
            speaking_rate
        ])

        return mel_spec_db, feature_vector

    except Exception as e:
        print(f"⚠️ Error extracting audio features from {audio_path}: {e}")
        return None, None


if __name__ == "__main__":
    test_file = "data/processed/human/test_processed.wav"
    mel, vec = extract_audio_features(test_file)
    print("Mel shape:", mel.shape if mel is not None else "None")
    print("Feature vector:", vec)
