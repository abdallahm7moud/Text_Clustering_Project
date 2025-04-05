import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_data(text, remove_numbers=True, stemming=False, remove_urls=True, remove_punctuation=True, lemmatization=False):
    """
    Preprocess the text data: tokenization, stop word removal, stemming/lemmatization, and cleaning of special characters.
    
    """
    text = text.lower()

    # Remove URLs if enabled
    if remove_urls:
        text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers if enabled
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Remove HTML tags and normalize whitespace
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove punctuation if enabled
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # Tokenization and Stopword Removal
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Apply stemming or lemmatization if enabled
    if lemmatization:
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    elif stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    return ' '.join(words)
