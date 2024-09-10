import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

# Download NLTK data when the module is imported
download_nltk_data()

# Now we can use NLTK functions
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, pd.Series):
        return text.apply(clean_single_text)
    elif isinstance(text, list):
        return [clean_single_text(t) for t in text]
    else:
        return clean_single_text(text)

def clean_single_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize (split into words)
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    # Join words back into string
    return ' '.join(words)

