from flask import Flask, render_template, request
import re
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model and vectorizer
loaded_model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vector.pkl", "rb"))

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words("english"))


def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)

    corpus = []

    for word in review:
        if word not in stpwrds:
            corpus.append(lemmatizer.lemmatize(word))

    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)

    return prediction


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction_text = ""

    if request.method == "POST":

        message = request.form.get("news")

        print("User Input:", message)

        pred = fake_news_det(message)

        print("Prediction:", pred)

        if pred[0] == 1:
            prediction_text = "Prediction of the News : Looking Fake News 📰"
        else:
            prediction_text = "Prediction of the News : Looking Real News 📰"

    return render_template(
        "prediction.html",
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)