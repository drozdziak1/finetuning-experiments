from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_name = "llama-2-7b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

