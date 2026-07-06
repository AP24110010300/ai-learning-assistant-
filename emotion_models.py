import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from preprocessing import clean_text, get_keyword_scores

EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

class EmotionPredictor:
    """BiLSTM Model Wrapper"""
    def __init__(self, model_dir="models/bilstm"):
        self.model_path = os.path.join(model_dir, "bilstm_model.h5")
        self.tokenizer_path = os.path.join(model_dir, "tokenizer.pickle")
        self.model = None
        self.tokenizer = None
        self.max_length = 50
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)

    def predict(self, text):
        if not self.model or not self.tokenizer:
            return None

        cleaned_text = clean_text(text)
        if not cleaned_text:
            cleaned_text = "none"

        sequences = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length)
        
        preds = self.model.predict(padded, verbose=0)[0]
        
        # Keyword enhancement
        keyword_scores = get_keyword_scores(cleaned_text)
        
        # Combine base model predictions with keyword scores
        final_scores = {}
        for i, emotion in enumerate(EMOTION_CLASSES):
            # Scale prediction to 0-100 and add keyword boost
            score = (preds[i] * 100) + keyword_scores[emotion]
            final_scores[emotion] = score
            
        # Renormalize to sum to 1.0 (or 100%)
        total = sum(final_scores.values())
        if total > 0:
            for k in final_scores:
                final_scores[k] = final_scores[k] / total
        else:
            for k in final_scores:
                final_scores[k] = 1.0 / len(EMOTION_CLASSES)

        top_emotion = max(final_scores, key=final_scores.get)
        confidence = final_scores[top_emotion]

        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'scores': final_scores,
            'cleaned_text': cleaned_text
        }


class BERTEmotionClassifier:
    """BERT Model Wrapper"""
    def __init__(self, model_dir="models/bert_emotion_model_final"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_dir):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.eval()
            except Exception as e:
                print(f"Error loading BERT model: {e}")

    def predict(self, text):
        if not self.model or not self.tokenizer:
            return None

        cleaned_text = clean_text(text)
        if not cleaned_text:
            cleaned_text = "none"

        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
            
        # If output length matches our classes, zip them
        if len(probs) == len(EMOTION_CLASSES):
            base_scores = {EMOTION_CLASSES[i]: probs[i] for i in range(len(EMOTION_CLASSES))}
        else:
            # Fallback mapping if model dimensions mismatch
            base_scores = {emotion: 1.0/len(EMOTION_CLASSES) for emotion in EMOTION_CLASSES}

        # Keyword enhancement
        keyword_scores = get_keyword_scores(cleaned_text)
        
        final_scores = {}
        for emotion in EMOTION_CLASSES:
            score = (base_scores[emotion] * 100) + keyword_scores[emotion]
            final_scores[emotion] = score
            
        # Renormalize
        total = sum(final_scores.values())
        if total > 0:
            for k in final_scores:
                final_scores[k] = final_scores[k] / total
        else:
            for k in final_scores:
                final_scores[k] = 1.0 / len(EMOTION_CLASSES)

        top_emotion = max(final_scores, key=final_scores.get)
        confidence = final_scores[top_emotion]

        return {
            'emotion': top_emotion,
            'confidence': confidence,
            'scores': final_scores,
            'cleaned_text': cleaned_text
        }


def get_mixed_emotions(scores, threshold=0.15):
    """Detects multiple emotions above a certain confidence threshold."""
    mixed = []
    # Sort scores descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Always include top emotion
    mixed.append(sorted_scores[0])
    
    # Check if second/third emotions are above threshold (e.g., 15%)
    for i in range(1, len(sorted_scores)):
        if sorted_scores[i][1] >= threshold:
            mixed.append(sorted_scores[i])
            
    return mixed
