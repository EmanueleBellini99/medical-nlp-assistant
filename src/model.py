# Machine Learning and NLP libraries
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TextStreamer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import lightning as L
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer

def load_model(model_name="unsloth/phi-4"):
    """Load and initialize the model with proper configuration."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
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

def prepare_for_training(model_name="unsloth/phi-4"):
    """Prepare model and tokenizer for training."""
    model, tokenizer = load_model(model_name)
    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
    
    # Prepare training data
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset_mapped = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    dataset_mapped = dataset_mapped.map(formatting_prompts_func, batched=True)
    
    return model, tokenizer, dataset_mapped

def setup_trainer(model, tokenizer, dataset_mapped, output_dir="outputs"):
    """Configure and return the trainer."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_mapped,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    
    return trainer

def print_gpu_stats():
    """Print GPU memory usage statistics."""
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{used_memory} GB of memory reserved.")
    return used_memory, max_memory