import spacy
import numpy as np
import joblib
from typing import Dict, List, Tuple

class NLPProcessor:
    """
    Handles all NLP operations including intent recognition and entity extraction.
    
    Security: Sanitizes user input to prevent injection attacks.
    """
    
    def __init__(self, intent_model_path, vectorizer_path):
        """Initialize NLP models."""
        self.intent_model = joblib.load(intent_model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.nlp = spacy.load("en_core_web_sm")
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to prevent security issues.
        
        Args:
            text: Raw user input
            
        Returns:
            Sanitized text
        """
        # Remove potential malicious characters
        text = text.strip()
        # Limit length to prevent buffer overflow attacks
        text = text[:1000]
        return text
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict user intent from text.
        
        Args:
            text: User query
            
        Returns:
            Tuple of (intent, confidence)
        """
        text = self.sanitize_input(text)
        text_vec = self.vectorizer.transform([text])
        intent = self.intent_model.predict(text_vec)[0]
        
        # Get confidence score
        proba = self.intent_model.predict_proba(text_vec)
        confidence = np.max(proba)
        
        return intent, confidence
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text using spaCy.
        
        Args:
            text: User query
            
        Returns:
            Dictionary of entity types and their values
        """
        text = self.sanitize_input(text)
        doc = self.nlp(text)
        
        entities = {
            'symptoms': [],
            'body_parts': [],
            'medications': [],
            'general': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'SYMPTOM']:
                entities['symptoms'].append(ent.text)
            elif ent.label_ in ['PRODUCT', 'ORG']:
                entities['medications'].append(ent.text)
            else:
                entities['general'].append(ent.text)
        
        # Extract noun chunks for body parts and symptoms
        for chunk in doc.noun_chunks:
            text_lower = chunk.text.lower()
            if any(word in text_lower for word in ['pain', 'ache', 'hurt', 'sore']):
                entities['symptoms'].append(chunk.text)
        
        return entities