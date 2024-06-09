import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load mBERT model
model_folder_path = 'mbert_model'
mbert_model = BertForSequenceClassification.from_pretrained(model_folder_path, ignore_mismatched_sizes=True)
mbert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def detect_cyberbullying_with_mbert(text):
    # Tokenize input text
    inputs = mbert_tokenizer(text, return_tensors="pt")

    # Forward pass, get logits
    logits = mbert_model(**inputs).logits

    # Get predicted class (0: No Cyberbullying, 1: Cyberbullying Detected)
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 1:
        return 'Cyberbullying Detected'
    else:
        return 'No Cyberbullying Detected'
