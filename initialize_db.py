from datasets import load_dataset
from vectordb import Memory
import time

def initialize_vector_db():
    """Initialize the vector database with the medical dataset."""
    print("Loading dataset...")
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
    
    print("Initializing vector database...")
    memory = Memory(
        memory_file="./memory.vecs",
        chunking_strategy={"mode":"paragraph"},
        embeddings="TaylorAI/bge-micro-v2"
    )

    questions = []
    metadata = []

    # Extract questions and answers
    for item in dataset["train"]:
        questions.append(item["input"])
        metadata.append({"answer": item["output"]})

    print(f"Adding {len(questions)} entries to database...")
    start_time = time.time()
    
    # Save to vector database
    try:
        memory.save(questions, metadata)
        end_time = time.time()
        print(f"Database initialized in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
        
    return True

if __name__ == "__main__":
    initialize_vector_db()