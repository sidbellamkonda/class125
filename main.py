from flask import Flask, jsonify, request
from classifier import get_prediction
app = Flask(__name__)
@app.route("/predict-digit", methods = ["POST"])
def predict_data():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction,
    }), 200
app.run(debug = True)