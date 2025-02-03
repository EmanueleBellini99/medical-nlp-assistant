import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from datasets import load_dataset

def load_model(model_name="unsloth/phi-4"):
    """Load and configure the language model."""
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

def setup_training(model_name="unsloth/phi-4"):
    """Setup model and tokenizer for training."""
    model, tokenizer = load_model(model_name)
    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")
    return model, tokenizer

def prepare_training_data(dataset_name="medalpaca/medical_meadow_medical_flashcards"):
    """Prepare dataset for training."""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset_mapped = load_dataset(dataset_name, split="train")
    dataset_mapped = dataset_mapped.map(formatting_prompts_func, batched=True)
    return dataset_mapped

def train_model(model, tokenizer, dataset_mapped, output_dir="outputs"):
    """Train the model."""
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
    trainer_stats = trainer.train()
    return trainer_stats