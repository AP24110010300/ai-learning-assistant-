import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fallback template responses in case API fails
EMOTION_RESPONSES = {
    "Bored": {
        "emoji": "🥱",
        "message": "It's completely normal to feel tired, especially when a problem looks intriguing but requires a lot of energy. When a problem feels overwhelming, try to break it into tiny pieces. Just focus on understanding one small part at a time, rather than the whole. For now, prioritize resting and recharging."
    },
    "Confident": {
        "emoji": "😎",
        "message": "That's the spirit! You're clearly well-prepared and understand the core concepts. Keep up the momentum, and try challenging yourself with advanced edge-cases to solidify your knowledge."
    },
    "Confused": {
        "emoji": "🤔",
        "message": "Feeling stuck is a huge part of learning. It means you're pushing your boundaries. Let's take a step back and look at the fundamentals. Often, reviewing the basic definitions can clear up the fog."
    },
    "Curious": {
        "emoji": "🤩",
        "message": "Your curiosity is your greatest asset! Exploring beyond the required material is how true mastery happens. Dive deeper into those interesting tangents, but make sure to anchor them back to your main goal."
    },
    "Frustrated": {
        "emoji": "😤",
        "message": "I hear you, and it's okay to feel annoyed when things don't work. Step away for five minutes, take a deep breath, and come back with fresh eyes. You've got this, it just takes persistence."
    }
}

def get_groq_response(field, problem, emotion, confidence):
    """Generates empathetic, field-aware responses using Groq."""
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "your-groq-api-key-here":
        return EMOTION_RESPONSES.get(emotion, {}).get("message", "Keep up the good work! (Fallback template)")

    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
You are an empathetic, highly effective academic AI Learning Assistant.

A student studying '{field}' has shared the following challenge:
"{problem}"

Our Emotion Detection engine has analyzed their input and identified their primary emotion as '{emotion}' with {confidence*100:.1f}% confidence.

Generate a short, personalized, and encouraging response (maximum 3-4 sentences). 
Acknowledge their feelings, provide a brief field-specific tip, and suggest a constructive next step.
Do NOT use overly robotic or formal language. Be conversational and supportive.
"""

        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # Fast, efficient model for this task
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and empathetic AI Learning Assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return EMOTION_RESPONSES.get(emotion, {}).get("message", "Keep up the good work! (Fallback template)")
