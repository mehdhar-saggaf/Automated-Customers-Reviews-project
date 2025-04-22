import re
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, BertConfig

# Path to your saved model
MODEL_PATH = "C:/Users/ASUS/OneDrive/Desktop/Automated-Customers-Reviews-project/product_review_flask/app/models/sentiment_model"

def load_sentiment_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    config = BertConfig.from_pretrained(MODEL_PATH)
    config.num_labels = 3  # ✅ make sure it's 3 to match the checkpoint
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=config,
        ignore_mismatched_sizes=True  # ✅ prevents loading errors
    )
    model.eval()
    return model, tokenizer

def clean_text(text: str) -> str:
    """
    Clean review text by removing HTML tags, special characters, and lowercasing.
    """
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def predict_sentiment(texts, model, tokenizer, batch_size=32):
    """
    Predict sentiment for a list of texts using a BERT model.
    Returns a list of labels: negative, neutral, positive
    """
    cleaned = [clean_text(t) for t in texts]
    predictions = []
    labels = ["negative", "neutral", "positive"]

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            predictions.extend([labels[p] for p in preds])

    return predictions
