import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    global step
    
    step = 0
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    global step

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    new_user_input_ids = tokenizer(prompt, return_tensors='pt').to("cuda")

    reply_ids = model.generate(**new_user_input_ids)

    result = tokenizer.batch_decode(reply_ids)

    # Return the results as a dictionary
    return result
