# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained sentiment analysis model for Malayalam
tokenizer = AutoTokenizer.from_pretrained("mohamedarish/BERT-malayalam-sentiment-l3cube")
model = AutoModelForSequenceClassification.from_pretrained("mohamedarish/BERT-malayalam-sentiment-l3cube")

app = FastAPI()

class Comment(BaseModel):
    text: str

@app.post("/predict/")
async def predict_sentiment(comment: Comment):
    inputs = tokenizer(comment.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    sentiment_class = torch.argmax(logits, dim=-1).item()
    
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map[sentiment_class]
    
    return {"sentiment": sentiment}
