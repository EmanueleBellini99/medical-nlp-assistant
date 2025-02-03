# Medical NLP Assistant

A medical question-answering system using NLP, RAG (Retrieval Augmented Generation), and voice interaction capabilities.

## Features
- Medical knowledge Q&A
- Voice interaction (TTS & STT)
- RAG implementation with vector database
- Fine-tuned language model

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the chatbot:
```python
from src.model import load_model
from src.rag import RAG
from src.voice import setup_tts, setup_stt

# Initialize components
model, tokenizer = load_model()
rag = RAG(model, tokenizer, memory)
```

## Contributors
- Bellini Emanuele
- Biancini Mattia 
- Kerscher Niklas
- Palumbo Jacopo