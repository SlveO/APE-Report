"""One model per worker. PCW uses ordinary causal Llama, never APE patches.

PCW merge/positions follow AI21Labs/Parallel-Context-Windows/pcw_wrapper.py.
The stock Transformers 4.44.2 Llama implements GQA and Llama-3.1 RoPE.
"""
import random
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ape import enable_attention_prefill_prefix, enable_attention_prefill_context, enable_attention_prefill_query
from .rules import DEFAULT, choose

METHODS = ('sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic')


class SampleError(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def chunks(tokens, count=None, maximum=7500):
    if not tokens:
        raise SampleError('empty document')
    if count is None:
        return [tokens[i:i + maximum] for i in range(0, len(tokens), maximum)]
    if not 1 <= count <= len(tokens):
        raise SampleError('invalid actual chunk count')
    q, r = divmod(len(tokens), count)
    lengths = [q + (i < r) for i in range(count)]
    result, start = [], 0
    for length in lengths:
        result.append(tokens[start:start + length])
        start += length
    assert sum(result, []) == tokens
    return result


def merge_pcw(windows, lengths):
    """Longest first (stable tie), one BOS; KV remains in compact GQA layout."""
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    order = [longest] + [i for i in range(len(lengths)) if i != longest]
    cache = tuple(tuple(torch.cat([windows[i][layer][kv][:, :, (0 if i == longest else 1):, :] for i in order], dim=2)
                        for kv in (0, 1)) for layer in range(len(windows[0])))
    return cache, max(lengths), order


def merge_ape(prefix, windows):
    p = prefix[0][0].shape[2]
    return tuple((torch.cat([pre[0]] + [w[layer][0][:, :, p:] for w in windows], dim=2),
                  torch.cat([pre[1]] + [w[layer][1][:, :, p:] for w in windows], dim=2),
                  torch.cat([pre[2]] + [w[layer][2][:, p:] for w in windows], dim=1),
                  len(windows)) for layer, pre in enumerate(prefix))


@dataclass
class Encoded:
    cache: tuple
    next_position: int
    request_tokens: list
    order: list
    document_length: int
    block_lengths: list
    cache_key: tuple


class Engine:
    def __init__(self, model, tokenizer, method, *, device='cuda:0'):
        if method not in METHODS:
            raise ValueError(method)
        self.model, self.tokenizer, self.method = model.eval(), tokenizer, method
        self.device = torch.device(device)
        self.encoded = None
        self.stage_calls = {'prefix': 0, 'context': 0, 'query': 0}
        self.stage = None
        self.hook = model.model.layers[0].self_attn.register_forward_hook(self._observed)

    def _observed(self, module, args, output):
        if self.stage is not None:
            self.stage_calls[self.stage] += 1

    @classmethod
    def load(cls, path, method):
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    attn_implementation='flash_attention_2').to('cuda:0').eval()
        return cls(model, tokenizer, method)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def tensor(self, tokens):
        return torch.tensor([tokens], device=self.device, dtype=torch.long)

    def forward(self, tokens, cache=None, position=None, logits=False):
        inputs = self.tensor(tokens)
        kwargs = {'input_ids': inputs, 'past_key_values': cache, 'use_cache': True, 'return_dict': True}
        if position is not None:
            kwargs['position_ids'] = torch.arange(position, position + len(tokens), device=self.device).unsqueeze(0)
        output = self.model.model(**kwargs)
        last = self.model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1, :] if logits else None
        return output.past_key_values, last

    def prepare(self, row):
        """Pre-tokenized dev documents remain exact; no chat BOS is inferred."""
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        prefix = encode(row['prefix'])
        suffix = encode(row['suffix'])
        document = row['document_tokens'] if 'document_tokens' in row else encode(row['document'])
        if not prefix or prefix[0] != self.tokenizer.bos_token_id or prefix.count(self.tokenizer.bos_token_id) != 1:
            raise SampleError('common prefix must contain exactly one initial BOS')
        if self.tokenizer.bos_token_id in document + suffix:
            raise SampleError('unexpected BOS inside document/request')
        parts = chunks(document, count=row.get('N'))
        if row.get('L') is not None and len(document) != row['L']:
            raise SampleError('frozen exact length mismatch')
        if len(prefix) + len(document) + len(suffix) + row['max_new_tokens'] > self.model.config.max_position_embeddings:
            raise SampleError('full public input exceeds model window; no silent truncation')
        return prefix, parts, suffix

    def encode(self, prefix, parts, suffix, cache_key):
        self.stage = None
        if self.method == 'sequential':
            cache, _ = self.forward(prefix + sum(parts, []))
            return Encoded(cache, len(prefix) + sum(map(len, parts)), suffix, list(range(len(parts))), sum(map(len, parts)), list(map(len, parts)), cache_key)
        if self.method == 'pcw':
            # Each independent window starts with its own BOS. The request has none.
            windows = [self.forward([self.tokenizer.bos_token_id] + p, position=0)[0] for p in parts]
            cache, position, order = merge_pcw(windows, [len(p) + 1 for p in parts])
            return Encoded(cache, position, prefix[1:] + suffix, order, sum(map(len, parts)), list(map(len, parts)), cache_key)
        enable_attention_prefill_prefix('llama3.1-8b-instruct', self.model)
        self.stage = 'prefix'
        pre, _ = self.forward(prefix)
        enable_attention_prefill_context('llama3.1-8b-instruct', self.model)
        self.stage = 'context'
        windows = [self.forward(p, cache=pre, position=len(prefix))[0] for p in parts]
        cache = merge_ape(pre, windows)
        return Encoded(cache, len(prefix) + max(map(len, parts)), suffix, list(range(len(parts))), sum(map(len, parts)), list(map(len, parts)), cache_key)

    @torch.inference_mode()
    def restore_cache(self, row, param=DEFAULT, rule=None):
        """Rebuild only lost document KV on resume, without another answer."""
        self.sync()
        started = time.perf_counter()
        prefix, parts, suffix = self.prepare(row)
        if self.method == 'ape_dynamic':
            param = choose(rule, sum(map(len, parts)), len(parts))
        elif self.method == 'ape_default':
            param = DEFAULT
        elif self.method == 'ape_fixed' and rule is not None:
            param = tuple(rule['fixed'])
        key = (self.method, tuple(prefix if self.method != 'pcw' else [self.tokenizer.bos_token_id]),
               tuple(tuple(p) for p in parts), tuple(param))
        self.encoded = self.encode(prefix, parts, suffix, key)
        self.sync()
        return time.perf_counter() - started

    @torch.inference_mode()
    def run(self, row, *, param=DEFAULT, rule=None, cache_mode='cold'):
        if cache_mode not in {'cold', 'hot', 'search'}:
            raise ValueError(cache_mode)
        self.sync()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        before_calls = self.stage_calls.copy()
        prefix, parts, suffix = self.prepare(row)
        select_start = time.perf_counter()
        if self.method == 'ape_dynamic':
            if rule is None:
                raise ValueError('dynamic rule is not frozen')
            param = choose(rule, sum(map(len, parts)), len(parts))
        elif self.method == 'ape_default':
            param = DEFAULT
        elif self.method == 'ape_fixed' and rule is not None:
            param = tuple(rule['fixed'])
        selection_seconds = time.perf_counter() - select_start
        # KV encoding for APE is T/S-independent. Search reuse is explicitly labelled;
        # application cache identities include actual configuration and method.
        encoding_param = None if cache_mode == 'search' else tuple(param)
        key = (self.method, tuple(prefix if self.method != 'pcw' else [self.tokenizer.bos_token_id]),
               tuple(tuple(p) for p in parts), encoding_param)
        get_start = time.perf_counter()
        hit = cache_mode != 'cold' and self.encoded is not None and self.encoded.cache_key == key
        cache_get_seconds = time.perf_counter() - get_start
        if cache_mode == 'hot' and not hit:
            raise ValueError('hot request has no matching in-process GPU cache')
        build_start = time.perf_counter()
        if not hit:
            self.encoded = None
            self.encoded = self.encode(prefix, parts, suffix, key)
        self.sync()
        build_seconds = time.perf_counter() - build_start if not hit else 0.0
        encoded = self.encoded
        # History/question are intentionally never part of cached KV.
        request = prefix[1:] + suffix if self.method == 'pcw' else suffix
        if self.method.startswith('ape_'):
            enable_attention_prefill_query('llama3.1-8b-instruct', self.model, float(param[0]), float(param[1]))
            self.stage = 'query'
        else:
            self.stage = None
        cache, logits = self.forward(request, cache=encoded.cache, position=encoded.next_position, logits=True)
        eos = self.model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        tokens = []
        ttft = None
        for step in range(row['max_new_tokens']):
            token = int(logits.argmax(dim=-1).item())
            self.sync()
            if ttft is None:
                ttft = time.perf_counter() - started
            tokens.append(token)
            if token in stops or step + 1 == row['max_new_tokens']:
                break
            cache, logits = self.forward([token], cache=cache, position=encoded.next_position + len(request) + step, logits=True)
        self.sync()
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
        total = time.perf_counter() - started
        result = {'answer': answer, 'output_token_ids': tokens, 'output_tokens': len(tokens),
                'L': encoded.document_length, 'N': len(parts), 'block_lengths': encoded.block_lengths,
                'original_document_count': row.get('original_document_count'), 'window_order': encoded.order,
                'query_start_position': encoded.next_position, 'prefix_tokens': len(prefix), 'request_tokens': len(request),
                'T': param[0] if self.method.startswith('ape_') else None, 'S': param[1] if self.method.startswith('ape_') else None,
                'cache_mode': cache_mode, 'cache_hit': hit, 'timing_condition': 'search_immutable_KV' if cache_mode == 'search' else 'application_GPU_KV' if cache_mode == 'hot' else 'cold',
                'cache_build_seconds': build_seconds, 'cache_get_seconds': cache_get_seconds, 'dynamic_selection_seconds': selection_seconds,
                'ttft_seconds': ttft, 'total_seconds': total,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None,
                'stage_calls': {k: self.stage_calls[k] - before_calls[k] for k in before_calls}}
        if not answer.strip():
            raise SampleError('empty model output', result)
        return result
