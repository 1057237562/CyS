import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

def generate(tokenizer, prompt, model, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[]):
    text = ""

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 20)]),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta
        
def flush_generator(generator):
    response = ""
    for chunk in generator:
        response += chunk
        response = response.replace('�', '')
    return response

def skip_reason(response: str):
    if "</think>" in response:
        response = response.split("</think>")[1]
    return response.strip()