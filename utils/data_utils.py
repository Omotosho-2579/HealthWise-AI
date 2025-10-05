import json
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeGraphRetriever:
    """
    Retrieves relevant information from the medical knowledge graph.
    
    Uses TF-IDF for efficient keyword-based matching.
    """
    
    def __init__(self, faiss_index_path, knowledge_graph_path):
        """Initialize knowledge graph and create TF-IDF index."""
        # Load knowledge graph
        with open(knowledge_graph_path, 'r') as f:
            self.knowledge_graph = json.load(f)
        
        # Create searchable text for each entry (keywords weighted heavily)
        self.documents = []
        for entry in self.knowledge_graph:
            # Repeat keywords 5 times for higher weight
            keywords_weighted = ' '.join(entry['keywords']) * 5
            searchable_text = f"{entry['topic']} {keywords_weighted} {entry['content']}"
            self.documents.append(searchable_text)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit vectorizer on all documents
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        print(f"✓ TF-IDF index created with {len(self.knowledge_graph)} entries")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search knowledge graph for relevant information using TF-IDF.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge graph entries
        """
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return matching entries
        results = []
        for idx in top_indices:
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
