import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

data_path = 'C:\\Users\\Verana\\Documents\\SharpDevelop Projects\\AsynchronousActivity6\\AsynchronousActivity6\\bin\\Debug\\NAOMI\\CSEL 302\\GENRE ORACLE\\LYRICS DATASET.csv'
df = pd.read_csv(data_path, usecols=['genre_name', 'lyrics_cleaned'])
df_clean = df.dropna()

vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, MultinomialNB())

X = df_clean['lyrics_cleaned']
y = df_clean['genre_name']

model.fit(X, y)

with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
