import mlx.core as mx
from customutils import generate_step, incremental_generate_step
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
import numpy as np

def generate(tokenizer, prompt, model, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[], prompt_cache=None):
    text = ""

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)], prompt_cache=prompt_cache),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta

def cache_generate(tokenizer, prompt, model, prompt_cache, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[]):
    token_offset = prompt_cache[0].offset
    text = ""
    tokens = tokenizer.encode(prompt)
    for (token, prob), n in zip(incremental_generate_step(mx.array(tokens[token_offset:], dtype=mx.int32), model, prompt_cache, mx.array(tokens[:token_offset], dtype=mx.int32), max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)]),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta

def pop_kvcache(cache, ranges: list[tuple]): # Cache shape (1,8,x,128)
    length = sum(end - start + 1 for start, end in ranges)
    for c in cache:
        k, v = c.state
        def pop(s):
            ptr = 0
            seg = []
            for start, end in ranges:
                seg.append(s[:,:,ptr:start,:])
                ptr = end + 1
            seg.append(s[:,:,ptr:,:])
            return mx.concat(seg, axis=2)
        c.state = (pop(k), pop(v))
        c.keys, c.values = c.state
        c.offset -= length
        
def pop_kvcache(cache, index: int): # Cache shape (1,8,x,128)
    state = []
    for c in cache:
        k, v = c.state
        def pop(s):
            seg = []
            seg.append(s[:,:,:index,:])
            seg.append(s[:,:,index + 1:,:])
            return mx.concat(seg, axis=2)
        state.append((k[:,:,index:index+1,:], v[:,:,index:index+1,:]))
        c.state = (pop(k), pop(v))
        c.keys, c.values = c.state
        c.offset -= 1
    return state
        
def insert_kvcache(cache, index: int, state: list[tuple]):
    for c, (k, v) in zip(cache, state):
        key, value = c.state
        c.state = (mx.concat([key[:,:,:index,:], k, key[:,:,index:,:]], axis=2), mx.concat([value[:,:,:index,:], v, value[:,:,:index,:]], axis=2))
        c.keys, c.values = c.state
        c.offset += 1

# /nothink = 26865
def fill_cache(tokens, model, prompt_cache, temp=0.6, top_p=0.95, top_k=20):
    token_offset = prompt_cache[0].offset
    next(incremental_generate_step(mx.array(tokens[token_offset:], dtype=mx.int32), model, prompt_cache, mx.array(tokens[:token_offset], dtype=mx.int32), max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)]))
    return tokens
    
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