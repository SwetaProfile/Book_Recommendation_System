# ============================================================
# Intelligent Book Recommendation System
# File: recommendation.py
# Purpose: Content-Based Book Recommendation Engine
# ============================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Load Cleaned Dataset
# ------------------------------------------------------------

DATASET = "cleaned_books.csv"

df = pd.read_csv(DATASET)

print("=" * 60)
print("Book Recommendation Engine")
print("=" * 60)

# ------------------------------------------------------------
# Check Required Column
# ------------------------------------------------------------

if "combined_text" not in df.columns:
    raise ValueError(
        "combined_text column not found.\n"
        "Run preprocessing.py first."
    )

# ------------------------------------------------------------
# Bag of Words
# ------------------------------------------------------------

print("\nCreating Bag of Words Matrix...")

bow = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

bow_matrix = bow.fit_transform(df["combined_text"])

print("BoW Shape :", bow_matrix.shape)

# ------------------------------------------------------------
# TF-IDF
# ------------------------------------------------------------

print("\nCreating TF-IDF Matrix...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(df["combined_text"])

print("TF-IDF Shape :", tfidf_matrix.shape)

# ------------------------------------------------------------
# Cosine Similarity
# ------------------------------------------------------------

print("\nCalculating Cosine Similarity...")

similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity Matrix Shape :", similarity_matrix.shape)

# ------------------------------------------------------------
# Create Index Mapping
# ------------------------------------------------------------

indices = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()

# ------------------------------------------------------------
# Search Function
# ------------------------------------------------------------

def search_book(book_name):
    """
    Search for books using partial title matching.
    """

    matches = df[
        df["title"].str.contains(
            book_name,
            case=False,
            na=False
        )
    ]

    return matches[["title", "author"]]

# ------------------------------------------------------------
# Recommendation Function
# ------------------------------------------------------------

def recommend(book_title, top_n=10):
    """
    Recommend similar books.
    """

    title = book_title.lower()

    if title not in indices:

        print("\nBook not found.")

        suggestions = search_book(book_title)

        if len(suggestions) > 0:

            print("\nDid you mean:\n")

            print(suggestions.head(10).to_string(index=False))

        return None

    idx = indices[title]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n + 1]

    recommendations = []

    for i, score in similarity_scores:

        recommendations.append({

            "Title":
            df.iloc[i]["title"],

            "Author":
            df.iloc[i]["author"],

            "Genres":
            df.iloc[i]["genres"],

            "Publisher":
            df.iloc[i]["publisher"]
            if "publisher" in df.columns
            else "N/A",

            "Similarity Score":
            round(score * 100, 2)

        })

    return pd.DataFrame(recommendations)

# ------------------------------------------------------------
# Book Details
# ------------------------------------------------------------

def book_details(book_title):
    """
    Display complete details of a selected book.
    """

    result = df[
        df["title"].str.lower() == book_title.lower()
    ]

    if result.empty:

        print("Book not found.")

        return

    row = result.iloc[0]

    print("\n")
    print("=" * 60)

    print("BOOK DETAILS")

    print("=" * 60)

    print("Title       :", row["title"])
    print("Author      :", row["author"])
    print("Genres      :", row["genres"])

    if "publisher" in df.columns:
        print("Publisher   :", row["publisher"])

    if "ratingsCount" in df.columns:
        print("Ratings     :", row["ratingsCount"])

    if "reviewsCount" in df.columns:
        print("Reviews     :", row["reviewsCount"])

    print("\nDescription\n")

    print(row["description"])

# ------------------------------------------------------------
# Interactive Mode
# ------------------------------------------------------------

if __name__ == "__main__":

    while True:

        print("\n")
        print("=" * 60)

        book = input(
            "Enter Book Title (or 'exit'): "
        )

        if book.lower() == "exit":
            break

        details = book_details(book)

        recommendations = recommend(book, top_n=10)

        if recommendations is not None:

            print("\n")
            print("=" * 60)

            print("Top Recommended Books")

            print("=" * 60)

            print(recommendations.to_string(index=False))

    print("\nThank you for using the Book Recommendation System!")