# ============================================================
# Intelligent Book Recommendation System
# File: preprocessing.py
# Purpose: Data Cleaning and Text Preprocessing
# ============================================================

import pandas as pd
import numpy as np
import re
import ast
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ------------------------------------------------------------
# Download NLTK resources (first run only)
# ------------------------------------------------------------

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

DATASET = "/Users/brajeshmishra/Downloads/book_dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# ------------------------------------------------------------
# Dataset Information
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------------------------------------
# Remove Duplicates
# ------------------------------------------------------------

before = len(df)

df.drop_duplicates(subset=["title"], inplace=True)

after = len(df)

print(f"\nDuplicates Removed : {before-after}")

# ------------------------------------------------------------
# Handle Missing Values
# ------------------------------------------------------------

important_columns = [
    "title",
    "description",
    "genres",
    "author"
]

df.dropna(subset=important_columns, inplace=True)

df.reset_index(drop=True, inplace=True)

print("\nRemaining Records :", len(df))

# ------------------------------------------------------------
# Convert List Columns to Text
# ------------------------------------------------------------

def convert_list(text):
    """
    Convert string representation of list
    into plain text.

    Example:
    "['Fantasy','Adventure']"

    becomes

    Fantasy Adventure
    """

    if pd.isna(text):
        return ""

    try:
        value = ast.literal_eval(text)

        if isinstance(value, list):
            return " ".join(str(i) for i in value)

        return str(value)

    except:
        return str(text)


df["genres"] = df["genres"].apply(convert_list)
df["author"] = df["author"].apply(convert_list)

# ------------------------------------------------------------
# Text Cleaning
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Text preprocessing pipeline.
    """

    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


print("\nCleaning Description...")

df["clean_description"] = df["description"].apply(clean_text)

print("Cleaning Genres...")

df["clean_genres"] = df["genres"].apply(clean_text)

print("Cleaning Authors...")

df["clean_author"] = df["author"].apply(clean_text)

# ------------------------------------------------------------
# Combine Text Features
# ------------------------------------------------------------

df["combined_text"] = (
    df["clean_genres"]
    + " "
    + df["clean_author"]
    + " "
    + df["clean_description"]
)

# ------------------------------------------------------------
# Remove Empty Records
# ------------------------------------------------------------

df = df[df["combined_text"].str.strip() != ""]

df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------
# Final Dataset Summary
# ------------------------------------------------------------

print("\nFinal Dataset Shape")

print(df.shape)

print("\nSample Combined Text\n")

print(df["combined_text"].head())

# ------------------------------------------------------------
# Save Cleaned Dataset
# ------------------------------------------------------------

OUTPUT_FILE = "cleaned_books.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned Dataset Saved Successfully")

print(OUTPUT_FILE)

print("\nPreprocessing Completed.")