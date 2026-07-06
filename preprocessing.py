import re

# Emotion Keywords for enhancement
EMOTION_KEYWORDS = {
    "Bored": ["bored", "tired", "sleepy", "uninterested", "dull", "tedious", "monotonous", "yawn", "exhausted"],
    "Confident": ["confident", "easy", "got this", "simple", "understand", "clear", "ready", "prepared", "know"],
    "Confused": ["confused", "lost", "stuck", "what", "how", "why", "don't understand", "unclear", "puzzled", "makes no sense"],
    "Curious": ["curious", "interesting", "fascinating", "wonder", "want to know", "intrigued", "eager", "explore"],
    "Frustrated": ["frustrated", "angry", "annoyed", "stupid", "hate", "impossible", "giving up", "irritated", "fail"]
}

def clean_text(text):
    """Cleans input text by removing special characters but preserving emotion-carrying punctuation."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except common punctuation that carries emotion (!, ?, .)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    
    return text.strip()

def get_keyword_scores(text):
    """Calculates keyword-based emotion scores with a 10x weight for explicit emotions."""
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS.keys()}
    text_lower = text.lower()
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[emotion] += 10  # 10x weight
                
    return scores
