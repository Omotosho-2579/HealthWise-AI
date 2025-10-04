import streamlit as st
import joblib
import json
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.nlp_utils import NLPProcessor
from utils.data_utils import KnowledgeGraphRetriever, load_wellness_tips, load_medical_terms
from utils.recommender import WellnessRecommender
from utils.report_processor import MedicalReportProcessor

# Page configuration
st.set_page_config(
    page_title="HealthWise AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all ML models and data (cached for performance)."""
    try:
        # Initialize NLP processor
        nlp_processor = NLPProcessor(
            "models/intent_model.joblib",
            "models/vectorizer.joblib"
        )
        
        # Initialize knowledge graph retriever
        kg_retriever = KnowledgeGraphRetriever(
            "models/faiss_index.joblib",
            "data/knowledge_graph.json"
        )
        
        # Load wellness tips and medical terms
        wellness_tips = load_wellness_tips()
        medical_terms = load_medical_terms()
        
        # Initialize recommender
        recommender = WellnessRecommender(wellness_tips)
        
        # Initialize report processor
        report_processor = MedicalReportProcessor(medical_terms)
        
        return {
            'nlp_processor': nlp_processor,
            'kg_retriever': kg_retriever,
            'recommender': recommender,
            'report_processor': report_processor
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure all model files are in the 'models' and 'data' directories.")
        return None


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'user_id': 'demo_user',
            'health_goals': ['general_wellness'],
            'conditions': []
        }


def process_chat_query(query: str, models: dict) -> dict:
    """
    Process user chat query through the AI pipeline.
    """
    # EMERGENCY DETECTION - Check for critical symptoms first
    emergency_keywords = [
        'chest pain', 'chest pressure', 'crushing chest', 'tight chest',
        'pain radiating', 'pain down arm', 'pain in jaw',
        'shortness of breath', 'difficulty breathing', 'can\'t breathe',
        'sudden severe headache', 'worst headache', 'severe headache',
        'confusion', 'slurred speech', 'face drooping', 'arm weakness',
        'severe bleeding', 'bleeding heavily', 'won\'t stop bleeding',
        'suicidal', 'want to die', 'kill myself', 'end my life',
        'passed out', 'loss of consciousness', 'can\'t wake up'
    ]
    
    query_lower = query.lower()
    is_emergency = any(keyword in query_lower for keyword in emergency_keywords)
    
    if is_emergency and ('chest' in query_lower or 'heart' in query_lower or 'arm' in query_lower):
        return {
            'response': """🚨 **EMERGENCY - CALL 112 IMMEDIATELY** 🚨

The symptoms you're describing (chest pain/pressure with pain radiating to arm) are **warning signs of a possible heart attack**.

**DO NOT WAIT - CALL 112 NOW**

While waiting for emergency services:
- Stop what you're doing and sit or lie down
- Chew an aspirin if available (unless allergic)
- Loosen tight clothing
- Stay calm and don't drive yourself

**Heart Attack Warning Signs:**
- Chest pain, pressure, or discomfort
- Pain radiating to arms, jaw, neck, or back
- Shortness of breath
- Cold sweats, nausea, lightheadedness

⚠️ **This is a medical emergency. I'm an AI chatbot and cannot provide emergency care. Call 112 or your local emergency number immediately.**""",
            'intent': 'emergency_detected',
            'confidence': 1.0,
            'entities': {'emergency': True},
            'sources': []
        }
    
    # Suicidal ideation detection
    if any(word in query_lower for word in ['suicidal', 'kill myself', 'want to die', 'end my life']):
        return {
            'response': """🆘 **CRISIS SUPPORT AVAILABLE** 🆘

If you're having thoughts of suicide, please reach out for help immediately:

**Immediate Help:**
- Nigeria Red Cross Society: **0803 123 0430, 0809 993 7357**
- 📱 Emergency Response Africa (ERA): **0 8000 2255 372**
- 🚨 Emergency Services: **112**

**You are not alone.** These feelings are temporary, and help is available 24/7.

**What to do right now:**
1. Call one of the numbers above - counselors are ready to listen
2. Stay with someone or go to a public place
3. Remove access to means of self-harm
4. Go to nearest emergency room if in immediate danger

I'm an AI and can't provide crisis counseling, but trained professionals are waiting to help you. Please reach out right now - your life matters.

**International Crisis Lines:** https://www.opencounseling.com/suicide-hotlines""",
            'intent': 'crisis_detected',
            'confidence': 1.0,
            'entities': {'crisis': True},
            'sources': []
        }
    
    # Normal processing for non-emergency queries
    intent, confidence = models['nlp_processor'].predict_intent(query)
    entities = models['nlp_processor'].extract_entities(query)
    kg_results = models['kg_retriever'].search(query, top_k=3)
    
    # Generate response based on intent
    if intent == 'symptom_checker':
        response = generate_symptom_response(query, kg_results, entities)
    elif intent == 'medication_explainer':
        response = generate_medication_response(kg_results)
    elif intent == 'general_wellness':
        response = generate_wellness_response(kg_results)
    elif intent == 'mental_health':
        response = generate_mental_health_response(kg_results)
    elif intent == 'health_summary':
        response = generate_health_summary()
    else:
        response = "I'm here to help with your health questions. Could you please rephrase that?"
    
    return {
        'response': response,
        'intent': intent,
        'confidence': confidence,
        'entities': entities,
        'sources': kg_results
    }

def generate_symptom_response(query: str, kg_results: list, entities: dict) -> str:
    """Generate response for symptom checker queries."""
    if kg_results:
        main_result = kg_results[0]
        response = f"**Based on what you've described:**\n\n{main_result['content']}\n\n"
        
        if entities['symptoms']:
            response += f"**Identified symptoms:** {', '.join(entities['symptoms'])}\n\n"
        
        response += "⚠️ **Important:** This is educational information only. Please consult a healthcare provider for proper diagnosis and treatment."
        
        return response
    else:
        return "I understand you're experiencing symptoms. While I can provide general information, it's important to consult with a healthcare provider for proper evaluation. Could you describe your symptoms in more detail?"


def generate_medication_response(kg_results: list) -> str:
    """Generate response for medication-related queries."""
    if kg_results:
        main_result = kg_results[0]
        response = f"**Medication Information:**\n\n{main_result['content']}\n\n"
        response += "💊 **Remember:** Always take medications as prescribed by your healthcare provider. Never adjust dosages without consulting your doctor."
        return response
    else:
        return "I don't have specific information about that medication in my knowledge base. Please consult your pharmacist or healthcare provider for accurate medication information."


def generate_wellness_response(kg_results: list) -> str:
    """Generate response for general wellness queries."""
    if kg_results:
        main_result = kg_results[0]
        response = f"**Wellness Advice:**\n\n{main_result['content']}\n\n"
        response += "✨ Remember, small consistent changes lead to lasting improvements in health!"
        return response
    else:
        return "I'd be happy to provide wellness guidance. Could you be more specific about what aspect of wellness you're interested in? (e.g., sleep, nutrition, exercise, stress management)"


def generate_mental_health_response(kg_results: list) -> str:
    """Generate response for mental health queries."""
    if kg_results:
        main_result = kg_results[0]
        response = f"**Mental Health Support:**\n\n{main_result['content']}\n\n"
        response += "🧠 **Support Resources:** If you're in crisis or need immediate help, please contact:\n"
        response += "- National Suicide Prevention Lifeline: 988\n"
        response += "- Crisis Text Line: Text HOME to 741741\n"
        response += "- Your local emergency services: 911"
        return response
    else:
        return "Mental health is incredibly important. While I can provide general guidance, please consider speaking with a mental health professional who can provide personalized support. How can I help you today?"


def generate_health_summary() -> str:
    """Generate a mock health summary."""
    return """**Your Health Summary:**

📊 **Recent Activity:**
- Last check-in: 2 days ago
- Wellness tips viewed: 5 this week

🎯 **Your Health Goals:**
- General wellness maintenance
- Improve sleep quality

💡 **Recommendation:** You're doing great! Keep up with your wellness routine. Consider exploring our sleep hygiene tips for better rest.

*Note: This is a demo summary. In production, this would show real user data from integrated health devices and logs.*"""


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 HealthWise AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Your Intelligent Health Companion</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> HealthWise AI provides educational information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check your setup.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # User profile configuration
        st.subheader("Your Health Goals")
        goals = st.multiselect(
            "Select your health goals:",
            ["improve_sleep", "reduce_stress", "weight_management", 
             "heart_health", "improve_mental_health", "general_wellness"],
            default=st.session_state.user_profile['health_goals']
        )
        st.session_state.user_profile['health_goals'] = goals
        
        st.markdown("---")
        
        # Feature selection
        st.subheader("🎯 Features")
        feature = st.radio(
            "Choose a feature:",
            ["💬 AI Chat Assistant", "🎁 Personalized Recommendations", "📄 Simplify Medical Report"],
            index=0
        )
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Main content area based on selected feature
    if feature == "💬 AI Chat Assistant":
        render_chat_interface(models)
    elif feature == "🎁 Personalized Recommendations":
        render_recommendations(models)
    elif feature == "📄 Simplify Medical Report":
        render_report_simplifier(models)


def render_chat_interface(models: dict):
    """Render the chat interface."""
    st.header("💬 Chat with HealthWise AI")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">👤 You: {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 HealthWise: {message["content"]}</div>', 
                          unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    user_input = st.text_input("💭 Ask me anything about your health:", key="chat_input", 
                               placeholder="e.g., I have a headache, what should I do?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send 📤", type="primary")
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Process query
        with st.spinner("🤔 Thinking..."):
            result = process_chat_query(user_input, models)
        
        # Add bot response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': result['response']
        })
        
       # Show intent and confidence in expander
        with st.expander("🔍 Analysis Details"):
            st.write(f"**Detected Intent:** {result['intent']}")
            st.write(f"**Confidence:** {result['confidence']:.2%}")
            if result['entities'].get('symptoms'):
                st.write(f"**Symptoms Found:** {', '.join(result['entities']['symptoms'])}")
            
            # Show emergency/crisis flags
            if result['entities'].get('emergency'):
                st.error("⚠️ EMERGENCY SITUATION DETECTED")
            
            if result['entities'].get('crisis'):
                st.error("🆘 CRISIS SITUATION DETECTED")
    
            st.experimental_rerun()
    
    # Quick action buttons
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Sleep tips 😴"):
            # Add to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'How can I improve my sleep?'
            })
            # Process immediately
            with st.spinner("🤔 Thinking..."):
                result = process_chat_query('How can I improve my sleep?', models)
            # Add response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['response']
            })
            st.experimental_rerun()
    
    with col2:
        if st.button("Stress relief 🧘"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'How can I reduce stress?'
            })
            with st.spinner("🤔 Thinking..."):
                result = process_chat_query('How can I reduce stress?', models)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['response']
            })
            st.experimental_rerun()
    
    with col3:
        if st.button("Healthy eating 🥗"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'Tips for healthy eating?'
            })
            with st.spinner("🤔 Thinking..."):
                result = process_chat_query('Tips for healthy eating?', models)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['response']
            })
            st.experimental_rerun()

def render_recommendations(models: dict):
    """Render personalized recommendations."""
    st.header("🎁 Your Personalized Wellness Recommendations")
    
    # Get recommendation
    recommendation = models['recommender'].get_recommendation(st.session_state.user_profile)
    
    # Display recommendation
    tip = recommendation['tip']
    
    st.markdown(f"### {tip['title']}")
    st.info(recommendation['reason'])
    st.markdown(tip['content'])
    st.caption(f"**Tags:** {', '.join(tip['tags'])}")
    
    # Refresh button
    if st.button("🔄 Get Another Recommendation"):
        st.experimental_rerun()
    
    st.markdown("---")
    
    # Proactive nudge checker
    st.subheader("🔔 Health Nudges")
    st.write("Check if you need any proactive health reminders:")
    
    col1, col2 = st.columns(2)
    with col1:
        avg_sleep = st.slider("Average sleep hours (last 3 days):", 4, 10, 7)
        stress_level = st.select_slider("Stress level:", ["low", "medium", "high"], "medium")
    
    with col2:
        daily_steps = st.number_input("Average daily steps:", 0, 20000, 5000, 500)
    
    user_data = {
        'avg_sleep_hours': avg_sleep,
        'stress_level': stress_level,
        'daily_steps': daily_steps
    }
    
    if st.button("Check for Nudges 🔍"):
        should_nudge, nudge_type = models['recommender'].check_proactive_nudge(user_data)
        
        if should_nudge:
            st.warning(f"⚠️ **Health Nudge:** We noticed you might benefit from {nudge_type.replace('_', ' ')} guidance. Check the chat for personalized tips!")
        else:
            st.success("✅ **Great job!** You're maintaining healthy habits. Keep it up!")


def render_report_simplifier(models: dict):
    """Render medical report simplification feature."""
    st.header("📄 Simplify Your Medical Report")
    
    st.info("🔒 **Privacy First:** Your medical reports are processed securely and never stored on our servers.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a medical report image (JPG, PNG):",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Report")
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process button
        if st.button("🔍 Extract & Simplify", type="primary"):
            with st.spinner("Processing your report..."):
                # Extract text
                extracted_text = models['report_processor'].extract_text_from_image(image)
                
                # Simplify terms
                result = models['report_processor'].simplify_medical_terms(extracted_text)
            
            with col2:
                st.subheader("📝 Simplified Report")
                
                # Show simplified text
                st.text_area("Simplified Version:", result['simplified_text'], height=300)
                
                # Show found terms
                if result['terms_found']:
                    st.markdown("### 🔤 Medical Terms Explained")
                    for term in result['terms_found']:
                        st.markdown(f"**{term['complex']}:** {term['simple']}")
                else:
                    st.info("No complex medical terms found in this report.")
                
                # Download button
                st.download_button(
                    label="💾 Download Simplified Report",
                    data=result['simplified_text'],
                    file_name="simplified_report.txt",
                    mime="text/plain"
                )
    else:
        st.markdown("""
        ### 📋 How to use:
        1. Upload an image of your medical report
        2. Click "Extract & Simplify"
        3. View your report with easy-to-understand explanations
        4. Download the simplified version
        
        **Supported formats:** JPG, PNG
        """)


# Run the app
if __name__ == "__main__":
    main()
