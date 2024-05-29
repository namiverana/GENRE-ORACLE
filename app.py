from flask import Flask, render_template, request, send_from_directory
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

app = Flask(__name__)

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [text.lower() for text in X]

with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

df = pd.read_csv('C:\\Users\\Verana\\Documents\\SharpDevelop Projects\\AsynchronousActivity6\\AsynchronousActivity6\\bin\\Debug\\NAOMI\\CSEL 302\\GENRE ORACLE\\LYRICS DATASET.csv')  # Adjust the path to your dataset

X_train = df['lyrics_cleaned'] 
y_train = df['genre_name']  

pipeline = make_pipeline(LowercaseTransformer(), vectorizer, MultinomialNB())

pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    if request.method == 'POST':
        lyrics = request.form.get('lyrics', '')
        prediction = pipeline.predict([lyrics])[0]
        probabilities = pipeline.predict_proba([lyrics]).flatten()
        genre_probabilities = {genre: round(prob * 100, 2) for genre, prob in zip(pipeline.named_steps['multinomialnb'].classes_, probabilities)}
        return render_template('predict.html', prediction=prediction, probabilities=genre_probabilities)

@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory('media', filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
