import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("jigsaw-toxic-comment-train.csv")

# Selecting relevant columns
df = df[['comment_text', 'toxic']]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove whitespace
    return text

# Apply preprocessing
df['comment_text'] = df['comment_text'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], test_size=0.2, random_state=42)

# Convert text to numerical representation
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
with open("models/toxic_classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("models/vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Predict function
def predict_toxicity(text):
    with open("models/toxic_classifier.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("models/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Toxic" if prediction[0] == 1 else "Safe"

# Example test
test_text = input("Enter a comment: ")
print("Prediction:", predict_toxicity(test_text))
