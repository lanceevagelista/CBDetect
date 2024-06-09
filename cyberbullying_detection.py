import re
import torch
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from better_profanity import profanity
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load mBERT model
model_folder_path = 'mbert_model'
mbert_model = BertForSequenceClassification.from_pretrained(model_folder_path, ignore_mismatched_sizes=True)
mbert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Initialize the stemmer
stemmer = PorterStemmer()

# Initialize better-profanity
profanity.load_censor_words()

def load_hate_words(file_path):
    # Load hate words from a file
    with open(file_path, 'r', encoding='utf-8') as file:
        hate_words = [line.strip() for line in file]
    return hate_words

def load_all_hate_words(hate_words_file_paths):
    # Load hate words from multiple files
    all_hate_words = []
    for file_path in hate_words_file_paths:
        hate_words = load_hate_words(file_path)
        all_hate_words.extend(hate_words)
    return all_hate_words

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def classify_output(output):
    # Apply softmax activation function
    probabilities = F.softmax(output, dim=1)

    # Thresholding: Choose the class with the highest probability
    _, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item()

def detect_cyberbullying_with_mbert(text):
    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Tokenize input text for mBERT
    inputs = mbert_tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass, get logits
    logits = mbert_model(**inputs).logits

    # Get predicted class (0: No Cyberbullying, 1: Cyberbullying Detected)
    predicted_class = classify_output(logits)

    if predicted_class == 1:
        return 'Cyberbullying Detected'
    else:
        return 'No Cyberbullying Detected'

def detect_cyberbullying(text, hate_words):
    # Remove HTML tags and convert text to lowercase for case-insensitive matching
    lowercase_text = BeautifulSoup(text.lower(), "html.parser").get_text()

    # List to store detected hate words
    detected_hate_words = []

    # Check if any hate word is present in the text
    for hate_word in hate_words:
        if re.search(rf'\b{re.escape(hate_word)}\b', lowercase_text):
            detected_hate_words.append(hate_word)

    # Apply stemming
    stemmed_text = ' '.join([stemmer.stem(word) for word in lowercase_text.split()])

    # Check for profanity using better-profanity
    if profanity.contains_profanity(stemmed_text):
        detected_hate_words.append("Profanity detected")

    # If hate words or profanity are detected, check with mBERT model
    if detected_hate_words:
        # Use mBERT for final classification
        result = detect_cyberbullying_with_mbert(text)
        return result, detected_hate_words

    return 'No Cyberbullying Detected', []
