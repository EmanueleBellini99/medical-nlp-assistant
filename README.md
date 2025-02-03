# Medical NLP Assistant

A medical question-answering system using NLP, RAG (Retrieval Augmented Generation), and voice interaction capabilities.

## Features
- Medical knowledge Q&A using RAG architecture
- Voice interaction with Text-to-Speech and Speech-to-Text
- Vector database for efficient knowledge retrieval
- Fine-tuned language model for medical domain
- Interactive command-line interface

## Installation

```bash
# Clone repository
git clone https://github.com/EmanueleBellini99/medical-nlp-assistant.git
cd medical-nlp-assistant

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the assistant
python main.py
```

## Project Structure
```
medical-nlp-assistant/
├── src/
│   ├── __init__.py
│   ├── model.py      # Language model implementation
│   ├── rag.py        # RAG system implementation
│   └── voice.py      # Voice interface components
├── main.py           # Main application script
├── requirements.txt  # Project dependencies
└── README.md        # Documentation
```

## Contributors
- Bellini Emanuele
- Biancini Mattia 
- Kerscher Niklas
- Palumbo Jacopo