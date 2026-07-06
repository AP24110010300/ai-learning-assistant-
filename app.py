import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from emotion_models import EmotionPredictor, BERTEmotionClassifier, get_mixed_emotions
from groq_integration import get_groq_response, EMOTION_RESPONSES

# 1. Page Configuration (Epic 5.1)
st.set_page_config(page_title="AI Learning Assistant", page_icon="🤖", layout="wide")

# 2. Session State Initialization
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
    
CSV_FILE = 'history.csv'

def save_to_csv(record):
    df_new = pd.DataFrame([record])
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, index=False)

def get_csv_examples_count():
    if os.path.exists(CSV_FILE):
        return len(pd.read_csv(CSV_FILE))
    return 0

# 3. Cache Models to avoid reloading (Epic 3.6)
@st.cache_resource
def load_models():
    bilstm = EmotionPredictor(model_dir="models/bilstm")
    bert = BERTEmotionClassifier(model_dir="models/bert_emotion_model_final")
    return bilstm, bert

bilstm_model, bert_model = load_models()
models_loaded = (bilstm_model.model is not None) and (bert_model.model is not None)

# 4. Sidebar Dashboard (Epic 5.1)
with st.sidebar:
    st.header("📊 Dashboard")
    status = "✅ Models loaded" if models_loaded else "❌ Models missing"
    st.write(f"Models: {status}")
    st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")
    st.write(f"CSV Examples: {get_csv_examples_count()}")
    
    if st.button("Clear History"):
        st.session_state.emotion_history = []
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        st.rerun()
        
    if st.session_state.emotion_history:
        st.subheader("Recent Sessions")
        recent = st.session_state.emotion_history[-3:]
        for item in reversed(recent):
            st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")

# 5. Main Application Header
st.title("🤖 Emotion-Aware Learning Assistant")
st.write("Get personalized help based on your field and emotional state")

# 6. Inputs & Settings (Epic 5.2)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Tell us about your learning challenge")
    
    field_options = [
        "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
        "Engineering", "Business", "Literature", "History", "Psychology", "Other"
    ]
    
    field = st.selectbox(
        "What field are you studying?",
        field_options,
        help="Select your area of study for personalized responses"
    )
    
    problem = st.text_area(
        f"Describe your {field} problem or challenge:",
        placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
        height=120
    )
    
    st.write("**Quick Examples:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    if q_col1.button("I'm confused about recursion"):
        problem = "I'm confused about recursion"
    if q_col2.button("Debugging is frustrating"):
        problem = "Debugging is frustrating"
    if q_col3.button("I'm curious about ML"):
        problem = "I'm curious about ML"
        
    submit_btn = st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True)

with col2:
    st.subheader("⚙️ Settings")
    use_ai = st.checkbox("Use AI Response (Groq)", value=True)
    save_data = st.checkbox("Save to CSV for learning", value=True)
    show_details = st.checkbox("Show analysis details", value=True)
    
    st.markdown("---")
    st.write("**📊 Predict from Saved Data**")
    use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)
    if use_csv_prediction:
        count = get_csv_examples_count()
        if count > 0:
            st.info(f"Using {count} saved examples for prediction")
        else:
            st.warning("No CSV data available yet.")

# 7. Prediction & Display Logic
if submit_btn and problem:
    with st.spinner("Analyzing emotion and generating guidance..."):
        
        # Predictions
        bilstm_result = bilstm_model.predict(problem)
        bert_result = bert_model.predict(problem)
        
        if not bilstm_result or not bert_result:
            st.error("Models failed to load. Please check your models/ directory.")
            st.stop()
            
        # Model Comparison UI
        st.markdown("---")
        st.subheader("🔬 Model Predictions Comparison")
        
        res_col1, res_col2 = st.columns(2)
        
        # BiLSTM Column
        with res_col1:
            st.write("**BiLSTM Student Adaptive**")
            bilstm_mixed = get_mixed_emotions(bilstm_result['scores'])
            
            if len(bilstm_mixed) > 1:
                mixed_text = " + ".join([f"{EMOTION_RESPONSES.get(em[0], {}).get('emoji', '')} {em[0]}" for em in bilstm_mixed])
                st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
            else:
                em = bilstm_result['emotion']
                emoji = EMOTION_RESPONSES.get(em, {}).get("emoji", "")
                st.metric("Emotion", f"{emoji} {em}", f"{bilstm_result['confidence']:.1%}")
                
            for emotion_name, score in sorted(bilstm_result['scores'].items(), key=lambda x: x[1], reverse=True):
                st.progress(float(score), text=f"{emotion_name}: {score:.1%}")
                
        # BERT Column
        with res_col2:
            st.write("**BERT Transformer**")
            bert_mixed = get_mixed_emotions(bert_result['scores'])
            
            if len(bert_mixed) > 1:
                mixed_text = " + ".join([f"{EMOTION_RESPONSES.get(em[0], {}).get('emoji', '')} {em[0]}" for em in bert_mixed])
                st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
            else:
                em = bert_result['emotion']
                emoji = EMOTION_RESPONSES.get(em, {}).get("emoji", "")
                st.metric("Emotion", f"{emoji} {em}", f"{bert_result['confidence']:.1%}")
                
            for emotion_name, score in sorted(bert_result['scores'].items(), key=lambda x: x[1], reverse=True):
                st.progress(float(score), text=f"{emotion_name}: {score:.1%}")

        # Final AI Response
        st.markdown("---")
        st.subheader("🤖 AI Learning Assistant Response")
        
        primary_emotion = bilstm_result['emotion']
        primary_confidence = bilstm_result['confidence']
        
        st.info(f"💡 AI Response based on BiLSTM prediction: **{primary_emotion}**")
        
        if use_ai:
            ai_response = get_groq_response(field, problem, primary_emotion, primary_confidence)
            st.write(ai_response)
        else:
            fallback = EMOTION_RESPONSES.get(primary_emotion, {}).get("message", "Keep going!")
            st.write(fallback)
            
        if show_details:
            with st.expander("Analysis Details"):
                st.write(f"**Original Problem:** {problem}")
                st.write(f"**Processed Text:** {bilstm_result['cleaned_text']}")
                st.write(f"**BiLSTM Confidence:** {bilstm_result['confidence']:.3f}")
                st.write(f"**AI Model:** Llama3 (via Groq)")
                st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
        # Save Interaction
        record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "field": field,
            "problem": problem,
            "emotion": primary_emotion,
            "confidence": primary_confidence,
            "model": "BiLSTM"
        }
        
        st.session_state.emotion_history.append(record)
        if save_data:
            save_to_csv(record)
            
        # Add BERT to CSV as well for charting
        if save_data:
            bert_record = record.copy()
            bert_record["emotion"] = bert_result['emotion']
            bert_record["confidence"] = bert_result['confidence']
            bert_record["model"] = "BERT"
            save_to_csv(bert_record)

# 8. Analytics Dashboard (Epic 5.4)
if st.session_state.emotion_history or get_csv_examples_count() > 0:
    st.markdown("---")
    st.header("📈 Learning Analytics")
    
    # Load data
    if use_csv_prediction and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(st.session_state.emotion_history)
        
    if not df.empty:
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                emotion_counts = df['emotion'].value_counts()
                fig1 = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                              title="Emotion Distribution")
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                df_copy = df.copy()
                df_copy['time'] = df_copy['timestamp'].dt.strftime('%H:%M:%S')
                fig2 = px.line(df_copy, x='time', y='confidence', color='emotion',
                               title="Emotional Journey", markers=True)
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab2:
            if 'model' in df.columns:
                field_emotion = df.groupby(['field', 'emotion', 'model']).size().reset_index(name='count')
                fig3 = px.bar(field_emotion, x='field', y='count', color='emotion', facet_col='model',
                              title="Emotions by Study Field & Model")
            else:
                field_emotion = df.groupby(['field', 'emotion']).size().reset_index(name='count')
                fig3 = px.bar(field_emotion, x='field', y='count', color='emotion',
                              title="Emotions by Study Field")
            st.plotly_chart(fig3, use_container_width=True)
            
        with tab3:
            st.write("Recent Activity Data")
            st.dataframe(df.tail(10))
