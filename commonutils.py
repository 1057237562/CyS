import mlx.core as mx
from customutils import generate_step, incremental_generate_step
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

def generate(tokenizer, prompt, model, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[], prompt_cache=None):
    text = ""

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)], prompt_cache=prompt_cache),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta

token_offset = 0

def cache_generate(tokenizer, prompt, model, prompt_cache, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[]):
    global token_offset
    text = ""
    tokens = tokenizer.encode(prompt)
    for (token, prob), n in zip(incremental_generate_step(mx.array(tokens[token_offset:], dtype=mx.int32), model, prompt_cache, mx.array(tokens[:token_offset], dtype=mx.int32), max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)]),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta
    token_offset = len(tokens)
    
def flush_generator(generator, max_step=-1):
    response = ""
    step = 0
    for chunk in generator:
        response += chunk
        response = response.replace('�', '')
        step += 1
        if max_step != -1 and step >= max_step:
            break
    return response

def skip_reason(response: str):
    if "</think>" in response:
        response = response.split("</think>")[1]
    return response.strip()