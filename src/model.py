from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import torch

def load_model(model_name="unsloth/phi-4"):
    """Load and configure the language model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, max_new_tok=50, temperature=0.3):
    """Generate text response from the model."""
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generate_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tok,
            temperature=temperature,
            do_sample=True,
        )
        prompt_len = inputs.input_ids.shape[1]
        generated_tokens = generate_ids[0][prompt_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up GPU memory
    del inputs, generate_ids, generated_tokens
    torch.cuda.empty_cache()

    # Post-processing
    banned_words = ["Assistant:", "User:", "Question:", "Answer:", "# Context", 
                   "<START_CONTEXT>", "## Medical knowledge", "# Past conversations history"]
    for banned in banned_words:
        idx = generated_text.find(banned)
        if idx != -1:
            generated_text = generated_text[:idx]

    return generated_text