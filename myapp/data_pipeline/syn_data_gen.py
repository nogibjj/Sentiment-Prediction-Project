import numpy as np
import random

# Set a seed for reproducibility
random.seed(0)
np.random.seed(0)

# Define word probabilities for two classes: 'positive' and 'negative'
# These probabilities are arbitrary and for illustrative purposes
# In a real-world scenario, they would be derived from the distribution of words from actual data
word_probs = {
    "positive": {
        "excellent": 0.2,
        "fantastic": 0.2,
        "good": 0.2,
        "great": 0.2,
        "love": 0.2,
    },
    "negative": {
        "bad": 0.2,
        "poor": 0.2,
        "terrible": 0.2,
        "awful": 0.2,
        "horrible": 0.2,
    },
}


# Function to generate synthetic sentences
def generate_synthetic_sentence(class_label, word_probs, sentence_length=10):
    words = list(word_probs[class_label].keys())
    probabilities = list(word_probs[class_label].values())

    sentence = np.random.choice(words, size=sentence_length, p=probabilities)
    return " ".join(sentence)


# Function to create a synthetic dataset
def create_synthetic_dataset(word_probs, num_samples_per_class=1000):
    synthetic_sentences = []
    synthetic_labels = []

    for class_label in word_probs:
        for _ in range(num_samples_per_class):
            sentence = generate_synthetic_sentence(class_label, word_probs)
            synthetic_sentences.append(sentence)
            synthetic_labels.append(1 if class_label == "positive" else 0)

    return synthetic_sentences, synthetic_labels


# Generating a balanced synthetic dataset
def create_balanced_synthetic_dataset(sen_limit=1000):
    synthetic_sentences, synthetic_labels = create_synthetic_dataset(
        word_probs, sen_limit
    )
    return synthetic_sentences, synthetic_labels


# Output a few examples from the synthetic dataset
# for i in range(5):
#    print(f"Sentence: {synthetic_sentences[i]} - Label: {synthetic_labels[i]}")

# Output the number of examples in each class

