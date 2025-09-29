import json
import joblib
import numpy as np
import spacy
from typing import Dict, List

class KnowledgeGraphRetriever:
    """
    Retrieves relevant information from the medical knowledge graph.
    
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, faiss_index_path, knowledge_graph_path):
        """Initialize knowledge graph and FAISS index."""
        self.index = joblib.load(faiss_index_path)
        
        with open(knowledge_graph_path, 'r') as f:
            self.knowledge_graph = json.load(f)
        
        self.nlp = spacy.load("en_core_web_sm")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search knowledge graph for relevant information.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge graph entries
        """
        # Generate query embedding
        doc = self.nlp(query)
        query_vector = doc.vector.astype('float32').reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Return matching entries
        results = []
        for idx in indices[0]:
            if idx < len(self.knowledge_graph):
                results.append(self.knowledge_graph[idx])
        
        return results


def load_wellness_tips() -> List[Dict]:
    """Load wellness tips from JSON file."""
    with open("data/wellness_tips.json", 'r') as f:
        return json.load(f)


def load_medical_terms() -> Dict[str, str]:
    """Load medical term simplification dictionary."""
    with open("data/medical_terms.json", 'r') as f:
        return json.load(f)