from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# Load the vectorizer along with the model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

from nltk.tokenize import RegexpTokenizer
#for tokexizing the data into
# "my name is"  => ["my","name","is"]
from nltk.stem.porter import PorterStemmer
#cleaning the data like "liking " -> "like"
from nltk.corpus import  stopwords
# to remove the unwanted data like the is
import nltk
nltk.download('stopwords')
# Downloading the stopwords
#tokenizer with spaceblank
tokenizer = RegexpTokenizer(r"\w+")

en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
def getCleanedText(text):
  text = text.lower()
  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text



@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = getCleanedText(review)
    rev = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(rev)
    if prediction[0] == 1:
        prediction_text = f'POSITIVE RESPONSE   '
    else:
        prediction_text = f'NEGETIVE RESPONSE   '
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()

