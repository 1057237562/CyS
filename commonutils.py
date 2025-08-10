import mlx.core as mx
from customutils import generate_step, incremental_generate_step
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from collections import deque

class TokenKVCache:
    def __init__(self, cache, tokens):
        self.kvcache = cache
        self.cache_tokens = tokens
        
    def cache(self, tokens):
        ranges = pop_diff_range(self.cache_tokens, tokens)
        if len(ranges) > 0:
            erase_kvcache(self.kvcache, ranges)
        self.cache_tokens = tokens
        return self.kvcache
    
    def pop_cache(self, index: int):
        token = self.cache_tokens.pop(index)
        kv = pop_kvcache(self.kvcache, index)
        return (token, kv)
    
    def insert_cache(self, index : int, tkv : tuple):
        t, kv = tkv
        self.cache_tokens.insert(index, t)
        insert_kvcache(self.kvcache, index, kv)

def generate(tokenizer, prompt, model, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[], prompt_cache=None):
    text = ""

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(prompt)), model, max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)], prompt_cache=prompt_cache),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta

def cache_generate(tokenizer, prompt, model, prompt_cache : TokenKVCache, temp=0.6, top_p=0.95, top_k=20, context_length=16384, stop_words=[]):
    tokens = tokenizer.encode(prompt)
    cache = prompt_cache.cache(tokens)
    token_offset = cache[0].offset
    text = ""
    for (token, prob), n in zip(incremental_generate_step(mx.array(tokens[token_offset:], dtype=mx.int32), model, cache, mx.array(tokens[:token_offset], dtype=mx.int32), max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)]),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        delta = tokenizer.decode(token)
        text += delta
        yield delta
        
def diff_range(list1, list2):
    ranges = []
    ptr = -2
    for index, (a, b) in enumerate(zip(list1, list2)):
        if a != b:
            if index - ptr == 1:
                ranges[-1][1] = index
            else:
                ranges.append([index, index])
            ptr = index
    return ranges

def pop_diff_range(list1, list2):
    queue = deque(list1)
    ranges = []
    ptr = 0
    for v in list2:
        start = ptr
        while len(queue) > 0 and queue[0] != v:
            ptr += 1
            queue.popleft()
        if start != ptr:
            ranges.append((start, ptr - 1))
        if len(queue) == 0:
            break
        queue.popleft()
        ptr += 1
    if len(queue) != 0:
        ranges.append((ptr, ptr + len(queue) - 1))
    return ranges

def erase_kvcache(cache, ranges: list[tuple]): # Cache shape (1,8,x,128)
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
        c.offset = c.keys.shape[2]
        
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
        c.offset = c.keys.shape[2]
    return state
        
def insert_kvcache(cache, index: int, state: list[tuple]):
    for c, (k, v) in zip(cache, state):
        key, value = c.state
        c.state = (mx.concat([key[:,:,:index,:], k, key[:,:,index:,:]], axis=2), mx.concat([value[:,:,:index,:], v, value[:,:,index:,:]], axis=2))
        c.keys, c.values = c.state
        c.offset = c.keys.shape[2]

# /nothink = 26865
def fill_cache(tokens, model, prompt_cache : TokenKVCache, temp=0.6, top_p=0.95, top_k=20):
    cache = prompt_cache.cache(tokens)
    token_offset = cache[0].offset
    next(incremental_generate_step(mx.array(tokens[token_offset:], dtype=mx.int32), model, cache, mx.array(tokens[:token_offset], dtype=mx.int32), max_tokens=-1, sampler=make_sampler(temp, top_p, top_k=top_k), logits_processors=[make_repetition_penalty(1.1, 60)]))
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