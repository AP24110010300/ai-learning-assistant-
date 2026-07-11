import os
from dotenv import load_dotenv

load_dotenv(override=True)

print("Testing BERT...")
from emotion_models import BERTEmotionClassifier
bert_model = BERTEmotionClassifier(model_dir="models/bert_emotion_model_final")
res = bert_model.predict("1+1")
print("BERT result:", res)

print("Testing BiLSTM...")
from emotion_models import EmotionPredictor
bilstm_model = EmotionPredictor(model_dir="models/bilstm")
res2 = bilstm_model.predict("1+1")
print("BiLSTM result:", res2)

print("Testing Groq...")
from groq_integration import get_groq_response
res3 = get_groq_response("Computer Science", "1+1", "Confused", 0.9)
print("Groq result:", res3)
print("Done")
