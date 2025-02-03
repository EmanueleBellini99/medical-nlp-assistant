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

## Setup and Usage

1. Initialize the vector database:
```bash
python initialize_db.py
```

2. Optional: Train the model:
```bash
python train.py
```

3. Start the chatbot:
```bash
python main.py
```

## Project Structure
```
medical-nlp-assistant/
├── src/
│   ├── __init__.py
│   ├── model.py      # Model implementation and training functions
│   ├── rag.py        # RAG system implementation
│   ├── voice.py      # Voice interface components
│   └── preprocess.py # Dataset loading and analysis
├── main.py           # Interactive chat interface
├── train.py          # Training script
├── initialize_db.py  # Database initialization
├── requirements.txt  # Project dependencies
└── README.md        # Documentation
```

## Files Description
- `initialize_db.py`: Creates vector database from medical dataset
- `train.py`: Handles model fine-tuning and training
- `main.py`: Runs interactive chatbot interface
- `src/model.py`: Language model setup and training utilities
- `src/rag.py`: Retrieval Augmented Generation implementation
- `src/voice.py`: Text-to-Speech and Speech-to-Text components
- `src/preprocess.py`: Dataset preprocessing functions

## Contributors
- Bellini Emanuele
- Biancini Mattia 
- Kerscher Niklas
- Palumbo Jacopo