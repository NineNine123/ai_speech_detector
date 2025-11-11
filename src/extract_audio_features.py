import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION = 20.0  # seconds

def extract_audio_features(audio_path):
    """
    Extract advanced acoustic features:
      - Pitch mean/std (frequency stability)
      - Jitter (pitch instability)
      - Shimmer (amplitude instability)
      - Spectral flux (smoothness)
      - Silence ratio
      - Energy variance
      - Speaking rate
      - Zero-crossing rate
      - MFCC mean/std (tone & timbre)
    Returns:
      mel_spec_db (for deep use), feature_vector (for classifier)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Trim or pad to 20s
        target_len = int(TARGET_DURATION * SAMPLE_RATE)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # --- Silence ratio ---
        silence_thresh = 0.02 * np.max(rms)
        silence_ratio = np.sum(rms < silence_thresh) / len(rms)

        # --- Pitch features ---
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmax=4000)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) else 0

        # --- Jitter (pitch instability) ---
        if len(pitch_values) > 2:
            pitch_diff = np.abs(np.diff(pitch_values))
            jitter = np.mean(pitch_diff / (pitch_values[:-1] + 1e-6))
        else:
            jitter = 0

        # --- Shimmer (amplitude instability) ---
        amp = np.abs(y)
        if len(amp) > 2:
            amp_diff = np.abs(np.diff(amp))
            shimmer = np.mean(amp_diff / (amp[:-1] + 1e-6))
        else:
            shimmer = 0

        # --- Spectral flux ---
        spec = np.abs(librosa.stft(y))
        flux = np.mean(np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))) / np.mean(spec)

        # --- Energy variance ---
        energy_var = np.var(rms)

        # --- Speaking rate ---
        energy_peaks = np.sum(rms > silence_thresh)
        speaking_rate = energy_peaks / TARGET_DURATION

        # --- Zero-crossing rate ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # --- MFCCs (tone texture) ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Combine into feature vector
        feature_vector = np.concatenate([
            [pitch_mean, pitch_std, jitter, shimmer, flux, silence_ratio,
             energy_var, speaking_rate, zcr],
            mfcc_mean, mfcc_std
        ])

        # Mel spectrogram (for potential CNN models)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db, feature_vector

    except Exception as e:
        print(f"⚠️ Error extracting features from {audio_path}: {e}")
        return None, None


if __name__ == "__main__":
    test_file = "data/processed/human/test_processed.wav"
    mel, vec = extract_audio_features(test_file)
    print("✅ Audio features extracted.")
    print("Feature vector length:", len(vec))
    print("Feature preview:", np.round(vec[:10], 3))
