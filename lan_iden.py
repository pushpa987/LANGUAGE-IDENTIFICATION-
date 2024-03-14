import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import pickle

# Load dataset
data = pd.read_csv('/content/dataset.csv')

# Initialize Porter Stemmer and download stopwords
nltk.download('stopwords')
ps = PorterStemmer()

# Preprocess data
corpus = []
for i in range(len(data['Text'])):
    rev = re.sub("^[a-zA-Z]", ' ', data['Text'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if set(stopwords.words())]
    rev = ' '.join(rev)
    corpus.append(rev)

# Create CountVectorizer
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()

# Label encoding for target variable
label = LabelEncoder()
y = label.fit_transform(data['language'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Naive Bayes classifier
classifier = MultinomialNB().fit(X_train, y_train)

# Predict on test set
pred = classifier.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Save model
joblib.dump(classifier, 'language_identification.sav')
with open('CountVectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Load model
model = joblib.load('language_identification.sav')

# Function to test model on new sentences
def test_model(test_sentence):
    languages = {
        0: 'Arabic', 1: 'Chinese', 2: 'Dutch', 3: 'English', 4: 'Estonian',
        5: 'French', 6: 'Hindi', 7: 'Indonesian', 8: 'Japanese', 9: 'Korean',
        10: 'Latin', 11: 'Persian', 12: 'Portuguese', 13: 'Pushto', 14: 'Romanian',
        15: 'Russian', 16: 'Spanish', 17: 'Swedish', 18: 'Tamil', 19: 'Thai',
        20: 'Turkish', 21: 'Urdu'
    }

    rev = re.sub('^[a-zA-Z]', ' ', test_sentence)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if word not in set(stopwords.words())]
    rev = ' '.join(rev)

    rev = cv.transform([rev]).toarray()
    output = model.predict(rev)[0]

    if output in languages.values():
        print("Predicted Language:", output)
    else:
        print("Unknown")

# Example sentences not in the dataset
test_model('This is a test sentence.')
test_model('यह एक परीक्षण वाक्य है।')
test_model('これはテスト文です。')
