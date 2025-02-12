# Standard libraries
import re
from collections import Counter
import sys

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Progress bar
from tqdm import tqdm

# NLP libraries
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.decomposition import PCA

def analyze_medical_dataset():
    # Load dataset
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    df = pd.DataFrame(dataset['train'])

    # Basic statistics
    print(f"Number of documents: {len(df)}")
    print(f"\nColumns in dataset: {df.columns.tolist()}")
    print(f"\nSample document:\n{df.iloc[0]}")

    # Document length statistics
    df['input_length'] = df['input'].apply(len)
    df['output_length'] = df['output'].apply(len)

    print("\nInput text length statistics:")
    print(df['input_length'].describe())
    print("\nOutput text length statistics:")
    print(df['output_length'].describe())

    # Vocabulary analysis
    stop_words = set(stopwords.words('english'))
    def get_vocab(text_series):
        vocab = Counter()
        for text in text_series:
            tokens = re.findall(r'\w+', text.lower())
            tokens = [token for token in tokens if token not in stop_words]
            vocab.update(tokens)
        return vocab

    input_vocab = get_vocab(df['input'])
    output_vocab = get_vocab(df['output'])

    print(f"\nUnique words in input text: {len(input_vocab)}")
    print(f"Unique words in output text: {len(output_vocab)}")

    # Most common words
    print("\nTop 10 most common words in questions:")
    print(pd.DataFrame(input_vocab.most_common(10), columns=['Word', 'Count']))

    print("\nTop 10 most common words in answers:")
    print(pd.DataFrame(output_vocab.most_common(10), columns=['Word', 'Count']))

    # Plotting
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.hist(df['input_length'], bins=50)
    plt.title('Distribution of Input Text Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')

    plt.subplot(1,2,2)
    plt.hist(df['output_length'], bins=50)
    plt.title('Distribution of Output Text Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Train Word2Vec model on combined text
    texts = [re.findall(r'\w+', text.lower()) for text in df['input'] + ' ' + df['output']]
    w2v_model = Word2Vec(texts, vector_size=100, window=5, min_count=2)

    # Find similar words for medical terms
    medical_terms = ['patient', 'treatment', 'disease', 'symptoms', 'drug']
    print("\nSimilar words for medical terms:")
    for term in medical_terms:
        try:
            similar = w2v_model.wv.most_similar(term)
            print(f"\n{term}:")
            for word, score in similar[:5]:
                print(f"  {word}: {score:.3f}")
        except KeyError:
            print(f"\n{term} not in vocabulary")

    # Edge cases
    short_inputs = df[df['input_length'] < df['input_length'].quantile(0.05)]
    long_inputs = df[df['input_length'] > df['input_length'].quantile(0.95)]
    print(f"\nShort Inputs:\n{short_inputs[['input', 'output']]}")
    print(f"\nLong Inputs:\n{long_inputs[['input', 'output']]}")

    # Save data for future
    w2v_model.save("medical_word2vec.model")
    similar_terms = {term: w2v_model.wv.most_similar(term) for term in medical_terms if term in w2v_model.wv}
    pd.DataFrame(similar_terms).to_csv("similar_terms.csv")

    # 3D visualization
    vocab = list(w2v_model.wv.index_to_key)  # Extract vocabulary
    word_vectors = w2v_model.wv[vocab]  # Get vectors for the vocabulary

    # Apply PCA to reduce dimensionality to 3D
    pca = PCA(n_components=3)
    reduced_vectors_3d = pca.fit_transform(word_vectors)

    # Extract PCA components for visualization
    x = reduced_vectors_3d[:, 0]
    y = reduced_vectors_3d[:, 1]
    z = reduced_vectors_3d[:, 2]

    # Create a 3D scatter plot using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, marker='o', s=50)

    # Annotate each point with the corresponding word
    for i in range(len(vocab)):
        ax.text(x[i], y[i], z[i], vocab[i], size=8)

    # Set plot title and labels
    ax.set_title("3D PCA of Word Embeddings", fontsize=16)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)

    plt.show()

    # Correlation between input and output lengths
    correlation = df[['input_length', 'output_length']].corr()
    print(f"\nCorrelation between input and output lengths:\n{correlation}")

    return df, w2v_model