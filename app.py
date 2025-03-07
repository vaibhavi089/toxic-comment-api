from flask import Flask, request, jsonify
import pickle
import re
import string

app = Flask(__name__)

# Load the trained model and vectorizer
with open("models/toxic_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("models/vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.strip()  
    return text

@app.route("/", methods=["GET"])
def home():
    return "Toxic Comment Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict_toxicity():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = preprocess_text(data["text"])
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)

    return jsonify({"text": data["text"], "prediction": "Toxic" if prediction[0] == 1 else "Safe"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
