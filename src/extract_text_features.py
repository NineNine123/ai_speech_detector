import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure tokenizers are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Common human filler words
FILLER_WORDS = ["uh", "um", "uhm", "erm", "eh", "ah", "hmm", "mm"]

# Human-like conversational phrases
HUMAN_PHRASES = [
    "you know",
    "i think",
    "i mean",
    "let me think",
    "kind of",
    "sort of",
    "like",
    "i guess",
    "you see",
    "i don't know",
    "to be honest"
]

def extract_text_features(transcript):
    """
    Extracts linguistic features:
      - filler_count: # of hesitation sounds
      - repetition_count: # of consecutive repeated words
      - phrase_count: # of human-like phrases
      - avg_sentence_len: average tokens per sentence
    Returns: numpy array [filler, repetition, phrases, avg_sentence_len]
    """
    import numpy as np

    if not transcript or not isinstance(transcript, str):
        return np.zeros(4)

    text = transcript.lower().strip()

    # Tokenize
    words = word_tokenize(text)
    if len(words) == 0:
        return np.zeros(4)

    # Filler count
    filler_count = sum(words.count(f) for f in FILLER_WORDS)

    # Repetition count (e.g., "I I", "think think")
    repetition_count = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])

    # Human phrase count
    phrase_count = sum(text.count(p) for p in HUMAN_PHRASES)

    # Average sentence length
    sentences = sent_tokenize(text)
    avg_sentence_len = np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0

    return np.array([filler_count, repetition_count, phrase_count, avg_sentence_len])


if __name__ == "__main__":
    example = "Um I think... I think this is kind of cool, you know?"
    features = extract_text_features(example)
    print("Text features:", features)
