# Medical NLP Assistant

A medical question-answering system using NLP, RAG (Retrieval Augmented Generation), and voice interaction capabilities.

## Features
- Medical knowledge Q&A using RAG architecture
- Voice interaction with Text-to-Speech and Speech-to-Text
- Vector database for efficient knowledge retrieval
- Fine-tuned language model for medical domain
- Interactive command-line interface

## Project Structure
```
medical-nlp-assistant/
├── NLP.ipynb           # Main implementation notebook
└── README.md          # Documentation
```

## Installation and Usage

Follow these steps to run the notebook:

1. Create a new virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies (the notebook will also handle this):
```bash
pip install jupyter datasets gensim matplotlib seaborn tqdm tensorboard bitsandbytes accelerate transformers vectordb2 peft lightning unsloth ffmpeg-python openai-whisper torchaudio ipywebrtc
```

3. Open and run the notebook:
```bash
jupyter notebook NLP.ipynb
```

## Contributors
- Bellini Emanuele
- Biancini Mattia 
- Kerscher Niklas
- Palumbo Jacopo