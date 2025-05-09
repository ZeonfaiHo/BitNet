import datasets
import transformers
import torch
from sys import argv
import json

model_path = "./"

device = "cuda"

# Load the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
model.to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

BATCH_SIZE = 32  # Adjust based on GPU memory
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_responses(message_batch):
    message_batch = [
        # f'<|begin_of_text|>User: {messages[0]["content"]}<|eot_id|>Assistant: '
        tokenizer.apply_chat_template(
           messages,
           add_generation_prompt=True,
           tokenize=False
        )
        for messages in message_batch
    ]
    inputs = tokenizer(message_batch, return_tensors="pt", padding=True, padding_side='left').to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    responses = []
    for i, output in enumerate(outputs):
        response = output[inputs['input_ids'].shape[-1]:]
        responses.append(tokenizer.decode(response, skip_special_tokens=True))
    
    return responses


message_batch = [
    [{"role": "user", "content": "What is the capital of France?"}],
]

responses = generate_responses(message_batch)

print(responses[0])