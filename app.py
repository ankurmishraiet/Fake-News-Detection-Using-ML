from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


# Load Model

model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))



@app.route("/")
def home():

    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    if request.method == "POST":


        news = request.form["news"]


        # Convert text into vector

        data = vectorizer.transform([news])


        # Prediction

        prediction = model.predict(data)[0]


        # Confidence

        try:

            probability = model.predict_proba(data)

            confidence = round(max(probability[0]) * 100, 2)


        except:


            confidence = "N/A"



        if prediction == 1:


            result = "FAKE NEWS"

            message = "This news article appears to be misleading or fake."


            color = "danger"



        else:


            result = "REAL NEWS"

            message = "This news article appears to be genuine."


            color = "success"



        return render_template(
            "index.html",
            prediction=result,
            message=message,
            confidence=confidence,
            color=color
        )



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
