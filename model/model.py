import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from transformers import GPT2Tokenizer


if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Загрузка обученной модели на основе предобученной от Сбербанка.
try:
    model = torch.load("/Users/yuliagavrisheva/Desktop/Project_games_rules/Board-Games-Rules-Generator-master/model/model_2.pt", map_location=torch.device('cpu'))
    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
except Exception as e:
    print(f'Loading model error: {e.args}')


def generate(prompt: str, len_gen=100, temperature=0.8):
    """ Функция генерации текста. """
    try:        
        generated = tokenizer.encode(prompt)
        context = torch.tensor([generated]).to(device)
        past = None
        for i in range(len_gen):
            output, past = model(context, past_key_values=past).values()
            output = output / temperature
            token = torch.distributions.Categorical(logits=output[..., -1, :]).sample()
            generated += token.tolist()
            context = token.unsqueeze(0)
        sequence = tokenizer.decode(generated)
    except Exception as e:
        print(f"Error: {e.args}")
    print(sequence)
    return sequence.replace(prompt, '')
