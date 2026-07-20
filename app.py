# ============================================================
# Intelligent Book Recommendation System
# Flask Application
# ============================================================

from flask import Flask, render_template, request
from recommendation import df, recommend

app = Flask(__name__)

import os

print("=" * 50)
print("Current Working Directory:", os.getcwd())
print("App Root Path:", app.root_path)
print("Template Folder:", app.template_folder)
print("Template Exists:",
      os.path.exists(os.path.join(app.root_path, "templates", "index.html")))
print("=" * 50)


@app.route("/", methods=["GET", "POST"])
def home():

    books = sorted(df["title"].unique())

    details = None
    recommendations = None
    selected_book = None

    if request.method == "POST":

        selected_book = request.form["book"]

        top_n = int(request.form["top_n"])

        details = df[df["title"] == selected_book].iloc[0]

        recommendations = recommend(selected_book, top_n)

    return render_template(
        "index.html",
        books=books,
        details=details,
        recommendations=recommendations,
        selected_book=selected_book
    )



if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5001
    )