import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -----------------------------
# 1. Load TinyLLaMA (4-bit QLoRA)
# -----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# -----------------------------
# 2. LoRA Config
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 3. Load QA Dataset (Excel)
# -----------------------------
qa_df = pd.read_csv("qa_dataset.csv")

# Expected columns: "question", "answer"
qa_df = qa_df.rename(columns={
    "question": "instruction",
    "answer": "output"
})

qa_dataset = Dataset.from_pandas(qa_df)

# -----------------------------
# 4. Load Intents Dataset (JSON)
# -----------------------------
import json

with open("intents.json") as f:
    intents_data = json.load(f)

intent_samples = []

# Assuming format like:
# { "intents": [ { "tag": "", "patterns": [], "responses": [] } ] }

for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        for response in intent["responses"]:
            intent_samples.append({
                "instruction": pattern,
                "output": response
            })

intent_dataset = Dataset.from_list(intent_samples)

# -----------------------------
# 5. Merge Datasets
# -----------------------------
dataset = concatenate_datasets([qa_dataset, intent_dataset])

# -----------------------------
# 6. Format for Training
# -----------------------------
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)

# -----------------------------
# 7. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./tinyllama-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none"
)

# -----------------------------
# 8. Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
    packing=True
)

# -----------------------------
# 9. Train
# -----------------------------
trainer.train()

# -----------------------------
# 10. Save Model
# -----------------------------
trainer.model.save_pretrained("./tinyllama_model(1)")
tokenizer.save_pretrained("./tinyllama-model(1)")