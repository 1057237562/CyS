import copy
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
)

import functools
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_model_path, load_model
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.tuner.utils import dequantize as dequantize_model
from mlx_lm.tuner.utils import load_adapters, nparams
from mlx_lm.models import cache

def _get_classes(config: dict):
    import model as arch
    return arch.Model, arch.ModelArgs

def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If ``False`` eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, lazy, model_config=model_config, get_model_classes=_get_classes)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )

    return model, tokenizer

generation_stream = mx.new_stream(mx.default_device())

def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[int, int]] = None
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_prorgress_callback (Callable[int, int]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """

    y = prompt
    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y

                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)
            # for c in prompt_cache:
            #     print(c.keys.shape, c.values.shape)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = y.size
        prompt_processed_tokens = 0
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt_processed_tokens += prefill_step_size
            y = y[prefill_step_size:]
            mx.clear_cache()

        y, logprobs = _step(y)

    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1
        
def incremental_generate_step(
    prompt: mx.array,
    model: nn.Module,
    prompt_cache: Any,
    history_prompt: mx.array,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_prorgress_callback (Callable[int, int]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """

    y = prompt
    tokens = history_prompt

    # Create the KV cache for generation
    if prompt_cache is None:
        raise ValueError("Incremental generation requires explicit prompt cache.")
    elif len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y, cache):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)
            logits = logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y

                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            y = sampler(logprobs)
            return y, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        y, logprobs = _step(y, prompt_cache)

    mx.async_eval(y, logprobs)
    tempor_cache = copy.deepcopy(prompt_cache)
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y, tempor_cache)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1