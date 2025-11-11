import os
import whisper

# Load Whisper model once globally
model = whisper.load_model("base")

def transcribe_audio(audio_path, output_text_path):
    """
    Transcribes a given audio file into text using Whisper.
    Saves transcript to the specified path.
    """
    try:
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()

        os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
        with open(output_text_path, "w") as f:
            f.write(text)

        return text

    except Exception as e:
        print(f"⚠️ Error transcribing {audio_path}: {e}")
        return ""


if __name__ == "__main__":
    test_audio = "data/processed/human/test_processed.wav"
    test_out = "data/transcripts/human/test.txt"
    t = transcribe_audio(test_audio, test_out)
    print("Transcript:", t)
