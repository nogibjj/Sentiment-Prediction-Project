import csv
from nltk.sentiment import SentimentIntensityAnalyzer

THRESHOLD = 0.05


def label_sentiment_nltk(comment):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(comment)["compound"]
    if polarity > THRESHOLD:
        return "positive"
    elif polarity < -THRESHOLD:
        return "negative"
    else:
        return "neutral"


def create_sentiment_labels(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["comment", "sentiment"])  # Write header

        next(reader, None)  # Skip header if present
        for row in reader:
            comment = row[0]  # Assuming comment is in the first column
            sentiment = label_sentiment_nltk(comment)
            writer.writerow([comment, sentiment])
