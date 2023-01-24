# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


if __name__ == "__main__":
    download_model()