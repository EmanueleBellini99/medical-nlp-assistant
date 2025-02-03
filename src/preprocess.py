from datasets import load_dataset
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk
from typing import Tuple, List, Dict

def load_medical_dataset() -> pd.DataFrame:
    """Load and prepare the medical dataset."""
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    return pd.DataFrame(dataset['train'])

def get_vocab(text_series) -> Counter:
    """Generate vocabulary from text series."""
    stop_words = set(stopwords.words('english'))
    vocab = Counter()
    for text in text_series:
        tokens = re.findall(r'\w+', text.lower())
        tokens = [token for token in tokens if token not in stop_words]
        vocab.update(tokens)
    return vocab

def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Analyze dataset statistics."""
    df['input_length'] = df['input'].apply(len)
    df['output_length'] = df['output'].apply(len)
    
    input_vocab = get_vocab(df['input'])
    output_vocab = get_vocab(df['output'])
    
    return {
        'doc_count': len(df),
        'input_stats': df['input_length'].describe().to_dict(),
        'output_stats': df['output_length'].describe().to_dict(),
        'input_vocab_size': len(input_vocab),
        'output_vocab_size': len(output_vocab),
        'input_common_words': input_vocab.most_common(10),
        'output_common_words': output_vocab.most_common(10)
    }