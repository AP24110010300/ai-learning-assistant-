import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure directories exist
os.makedirs("models/bilstm", exist_ok=True)
os.makedirs("models/bert_emotion_model_final", exist_ok=True)

print("Generating dummy BiLSTM Model...")
# 1. Generate BiLSTM Model
vocab_size = 5000
embedding_dim = 64
max_length = 50

# Dummy tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(["hello world", "this is a test", "I am confused", "I am frustrated", "I am bored"])
with open('models/bilstm/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Model architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')  # 5 emotion classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create a tiny dummy dataset to initialize weights
import numpy as np
X_dummy = np.random.randint(0, vocab_size, (10, max_length))
y_dummy = np.zeros((10, 5))
y_dummy[:, 0] = 1 # Dummy labels
model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

model.save("models/bilstm/bilstm_model.h5")
print("BiLSTM Model and tokenizer saved successfully to models/bilstm/")

print("Generating BERT Model assets...")
# 2. Setup BERT Model (5-class classification)
# We use a lightweight base model and initialize it for 5 classes
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Save to the expected local directory
tokenizer.save_pretrained("models/bert_emotion_model_final")
bert_model.save_pretrained("models/bert_emotion_model_final")

print("BERT Model saved successfully to models/bert_emotion_model_final/")
print("Model asset generation complete!")
