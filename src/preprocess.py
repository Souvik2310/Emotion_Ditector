# src/preprocess.py
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ensure punkt and stopwords are present
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove non-ascii characters (emojis)
    text = "".join([ch for ch in text if ch.isascii()])

    # collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords (optional: you can toggle this later)
    words = word_tokenize(text)
    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)
