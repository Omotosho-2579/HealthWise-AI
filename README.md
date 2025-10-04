# 🏥 HealthWise AI - Intelligent Health Companion

A production-ready AI/ML-powered healthcare chatbot built with Python and Streamlit.

## 🌟 Features

- **💬 AI Chat Assistant**: Natural language conversation for health queries
- **🎯 Intent Recognition**: Classifies user queries into symptom checking, medication info, wellness advice, etc.
- **🧠 Medical NER**: Extracts symptoms, medications, and body parts from text
- **📚 Knowledge Graph**: Retrieves relevant medical information using FAISS similarity search
- **🎁 Personalized Recommendations**: Content-based wellness tips tailored to user goals
- **🔔 Proactive Nudges**: Rule-based system for health reminders
- **📄 Report Simplification**: OCR + medical term translation for easier understanding

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **ML Libraries**: scikit-learn, spaCy, FAISS
- **OCR**: pytesseract, Pillow
- **Data**: pandas, numpy

## 📁 Repository Structure

```
healthwise-ai/
├── streamlit_app.py          # Main Streamlit application
├── train_models.py            # Model training script (run on Colab)
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── models/                    # Trained models (generated)
│   ├── intent_model.joblib
│   ├── vectorizer.joblib
│   └── faiss_index.joblib
├── data/                      # Data files (generated)
│   ├── knowledge_graph.json
│   ├── wellness_tips.json
│   └── medical_terms.json
└── utils/                     # Utility modules
    ├── __init__.py
    ├── nlp_utils.py
    ├── data_utils.py
    ├── recommender.py
    └── report_processor.py


## 📄 License & Copyright

© 2024 [Your Name]. All Rights Reserved.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

⚠️ **Important**: This software is provided for educational purposes. Unauthorized copying, modification, or distribution is prohibited without explicit permission.

