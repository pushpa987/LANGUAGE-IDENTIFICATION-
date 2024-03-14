from flask import Flask, render_template, request
import joblib
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model
model = joblib.load('language_identification.sav')

# Load the CountVectorizer
with open('CountVectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# Initialize the Porter Stemmer
ps = nltk.PorterStemmer()

# Function to preprocess input text
def preprocess_text(text):
    rev = re.sub('^[a-zA-Z]',' ', text)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if word not in set(stopwords.words('english'))]
    return ' '.join(rev)

# Function to predict language
def predict_language(sentence):
    languages = {
    'Arabic': 0,
    'Chinese': 1,
    'Dutch' : 2,
    'English': 3,
    'Estonian': 4,
    'French': 5,
    'Hindi': 6,
    'Indonesian' : 7,
    'Japanese' : 8,
    'Korean': 9,
    'Latin': 10,
    'Persian': 11,
    'Portugese' : 12,
    'Pushto': 13,
    'Romanian': 14,
    'Russian':15,
    'Spanish': 16,
    'Swedish': 17,
    'Tamil': 18,
    'Thai': 19,
    'Turkish' : 20,
    'Urdu' : 21
    }

    preprocessed_sentence = preprocess_text(sentence)
    rev = cv.transform([preprocessed_sentence]).toarray()
    output = model.predict(rev)[0]

    keys = list(languages)
    values = list(languages.values())
    position = values.index(output)

    return keys[position]

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        language = predict_language(sentence)
        return render_template('result.html', sentence=sentence, language=language)

if __name__ == '__main__':
    app.run(debug=True)
