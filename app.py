from flask import Flask, render_template, request
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Download stopwords
nltk.download('stopwords')

# Tokenizer with space blank
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanedText(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = getCleanedText(review)
    
    # Use the same vectorizer as used during training
    rev = vectorizer.transform([cleaned_review])
    
    # Ensure that the number of features matches the trained model
    if rev.shape[1] != model.coef_.shape[1]:
        return "Error: Mismatch in the number of features between the model and input data"
    
    prediction = model.predict(rev)
    if prediction[0] == 1:
        prediction_text = "POSITIVE RESPONSE"
    elif prediction[0] == 0:
        prediction_text = "NEGATIVE RESPONSE"
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
