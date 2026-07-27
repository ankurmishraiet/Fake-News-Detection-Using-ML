from flask import Flask, render_template, request, send_file
import pickle
from datetime import datetime
from reportlab.pdfgen import canvas


app = Flask(__name__)


# Load Model

model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vector.pkl", "rb"))



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

            probability = model.decision_function(data)

            confidence = round(
                abs(probability[0]) * 100,
                2
            )


            if confidence > 99:
                confidence = 99


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

            color=color,

            news=news

        )





@app.route("/download_report")
def download_report():


    news = request.args.get("news")

    prediction = request.args.get("prediction")

    confidence = request.args.get("confidence")



    file_name = "Fake_News_Detection_Report.pdf"



    pdf = canvas.Canvas(file_name)



    pdf.setTitle(
        "Fake News Detection Report"
    )



    pdf.drawString(
        100,
        750,
        "Fake News Detection Report"
    )



    pdf.drawString(
        100,
        700,
        "Prediction: " + str(prediction)
    )



    pdf.drawString(
        100,
        670,
        "Confidence: " + str(confidence) + "%"
    )



    pdf.drawString(
        100,
        630,
        "Date: " + str(datetime.now())
    )



    pdf.drawString(
        100,
        580,
        "News:"
    )



    # PDF line length limit

    text = str(news)

    y = 550


    for line in text.split():

        if y < 100:
            break


        pdf.drawString(
            100,
            y,
            line
        )

        y -= 20



    pdf.save()



    return send_file(
        file_name,
        as_attachment=True
    )





if __name__ == "__main__":


    app.run(
        host="0.0.0.0",
        port=5000
    )
