import csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Initialize stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def load_comments_and_labels(file_path):
    texts, labels = [], []
    with open(file_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header if there is one
        for row in reader:
            if row:
                texts.append(row[0]) 
                labels.append(row[1])  
    return texts, labels


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def tokenize_text(text):
    return word_tokenize(text)


def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]


def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        cleaned = clean_text(comment)
        tokenized = tokenize_text(cleaned)
        no_stop_words = remove_stop_words(tokenized)
        stemmed = stem_words(no_stop_words)
        preprocessed_comments.append(" ".join(stemmed))
    return preprocessed_comments
