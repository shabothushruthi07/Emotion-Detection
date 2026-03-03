from flask import Flask, render_template, request, url_for
import pandas as pd
import re
import os
from sklearn.metrics import accuracy_score
from transformers import pipeline

app = Flask(__name__)

# --- Clean text helper ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

# Dummy model accuracy list (add your values)
model_accuracies = [
    {"name": "Logistic Regression", "accuracy": 0.91},
    {"name": "Naive Bayes", "accuracy": 0.89},
    {"name": "SVM", "accuracy": 0.93},
]

# --- Load DL model (pretrained emotion classifier) ---
dl_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Add DL model info (accuracy N/A)
model_accuracies.append({
    "name": "DistilRoBERTa (DL, emotion model)",
    "accuracy": None
})

# Identify top ML model
top_model = max([m for m in model_accuracies if m['accuracy'] is not None], key=lambda x: x['accuracy'])

# --- Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    cleaned = clean_text(user_input)

    # Use DL emotion model
    preds = dl_model(cleaned)[0]
    top_emotion = max(preds, key=lambda x: x['score'])['label']
    depression_emotions = ['sadness', 'fear', 'anger']
    result = "Depression" if top_emotion.lower() in depression_emotions else "Not Depression"

    suggestion = ""
    if result == "Depression":
        suggestion = (
            "💡 Suggestions:<br>"
            "- Talk to someone you trust.<br>"
            "- Practice mindfulness or short walks.<br>"
            "- Consider professional help if persistent.<br>"
            "- Engage in a hobby or social activity."
        )
    else:
        suggestion = "🙂 Great! Keep up positive habits and self-care."

    return render_template(
        'result.html',
        text=user_input,
        emotion=top_emotion,
        result=result,
        suggestion=suggestion,
        models=model_accuracies,
        top_model=top_model
    )

@app.route('/models')
def models_view():
    img_path = "static"
    images = [img for img in os.listdir(img_path) if img.endswith(".PNG")]
    return render_template("models.html", models=model_accuracies, images=images, top_model=top_model)

if __name__ == '__main__':
    app.run(debug=True)
