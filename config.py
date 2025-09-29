import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model file paths
INTENT_MODEL_PATH = MODELS_DIR / "intent_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
NER_MODEL_PATH = MODELS_DIR / "ner_model.joblib"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.joblib"

# Data file paths
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "knowledge_graph.json"
WELLNESS_TIPS_PATH = DATA_DIR / "wellness_tips.json"
MEDICAL_TERMS_PATH = DATA_DIR / "medical_terms.json"

# API Configuration
APP_NAME = "HealthWise AI"
APP_VERSION = "1.0.0"

# Security settings
# IMPORTANT: In production, use environment variables for sensitive data
# Never store real patient data in plain text
# Implement proper encryption for data at rest and in transit
ENABLE_DATA_ANONYMIZATION = True
LOG_USER_QUERIES = False  # Set to False to protect user privacy
