from src import load_model, RAG, VoiceInterface
from vectordb import Memory
import time

def main():
    # Initialize components
    print("Loading model...")
    model, tokenizer = load_model()
    
    print("Initializing vector database...")
    memory = Memory(
        memory_file="./memory.vecs",
        chunking_strategy={"mode":"paragraph"},
        embeddings="TaylorAI/bge-micro-v2"
    )
    
    print("Setting up RAG system...")
    rag = RAG(model, tokenizer, memory)
    
    print("Initializing voice interface...")
    voice = VoiceInterface()
    
    # Start conversation
    print("\nMedical Assistant ready! Press Ctrl+C to exit.\n")
    dialogue_history = ["Hi, I'm your Medical Assistant. How can I help you?"]
    print(f"Assistant: {dialogue_history[0]}")
    
    try:
        while True:
            user_message = input("User: ")
            dialogue_history.append(user_message)
            
            chatbot_response = rag.chat(user_message)
            dialogue_history.append(chatbot_response)
            
            print(f"Assistant: {chatbot_response}")
            voice.text_to_speech(chatbot_response)
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nThank you for using the Medical Assistant!")

if __name__ == "__main__":
    main()