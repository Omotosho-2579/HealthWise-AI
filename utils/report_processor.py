from PIL import Image
import pytesseract
import re
from typing import Dict

class MedicalReportProcessor:
    """
    Processes medical reports: OCR extraction and term simplification.
    
    Security Note: This component handles sensitive medical documents.
    Ensure:
    - Images are processed in-memory only
    - No files are permanently stored
    - All processing is done server-side
    - Results are encrypted before transmission
    """
    
    def __init__(self, medical_terms: Dict[str, str]):
        """
        Initialize processor with medical terms dictionary.
        
        Args:
            medical_terms: Dictionary mapping complex terms to simple explanations
        """
        self.medical_terms = medical_terms
    
    def extract_text_from_image(self, image) -> str:
        """
        Extract text from medical report image using OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        # Preprocess image for better OCR
        # Convert to grayscale
        image = image.convert('L')
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        return text.strip()
    
    def simplify_medical_terms(self, text: str) -> Dict:
        """
        Replace complex medical terms with simple explanations.
        
        Args:
            text: Raw medical report text
            
        Returns:
            Dictionary with simplified text and found terms
        """
        simplified_text = text
        found_terms = []
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(self.medical_terms.items(), 
                            key=lambda x: len(x[0]), 
                            reverse=True)
        
        for complex_term, simple_term in sorted_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(complex_term), re.IGNORECASE)
            if pattern.search(simplified_text):
                found_terms.append({
                    'complex': complex_term,
                    'simple': simple_term
                })
                simplified_text = pattern.sub(
                    f"{complex_term} ({simple_term})", 
                    simplified_text
                )
        
        return {
            'original_text': text,
            'simplified_text': simplified_text,
            'terms_found': found_terms
        }