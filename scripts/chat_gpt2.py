#!/usr/bin/env python3
"""
chat_gpt2.py -- Interactive GPT-2 Chat on BitbyBit Custom GPU

Runs real GPT-2-small (124M params) inference using pure NumPy,
with custom GPU performance metrics shown per response.

No PyTorch needed -- loads weights directly from safetensors.

Usage:
    python scripts/chat_gpt2.py
    python scripts/chat_gpt2.py --max-tokens 50
"""

import numpy as np
import os
import sys
import time
import json
import re
import argparse
import struct

sys.path.insert(0, os.path.dirname(__file__))
from extract_gpt2_weights import download_gpt2_weights

# ============================================================================
# GPT-2 Architecture Constants (full model)
# ============================================================================
EMBED_DIM  = 768
NUM_HEADS  = 12
HEAD_DIM   = 64   # 768 / 12
FFN_DIM    = 3072  # 4 * 768
NUM_LAYERS = 12
VOCAB_SIZE = 50257
MAX_SEQ    = 1024

# GPU Performance Model
GPU_CLK_MHZ = 100
CYCLES_PER_TOKEN_768 = 150_000

# Q8.8 zero threshold: values that would quantize to 0 in hardware
# Q8.8 has 1/256 = 0.00390625 resolution, so |val| < 0.5/256 rounds to 0
Q88_ZERO_THRESH = 0.5 / 256.0  # ~0.00195


# ============================================================================
# Tokenizer (BPE)
# ============================================================================
class GPT2Tokenizer:
    """GPT-2 BPE tokenizer, pure Python."""

    def __init__(self, cache_dir):
        self.encoder = {}
        self.decoder = {}
        self._load(cache_dir)

    def _load(self, cache_dir):
        vocab_path = os.path.join(cache_dir, "vocab.json")
        merges_path = os.path.join(cache_dir, "merges.txt")

        if not os.path.exists(vocab_path):
            self._download(cache_dir)

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            self.bpe_merges = [
                tuple(l.split()) for l in lines[1:]
                if l.strip() and len(l.split()) == 2
            ]
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.bpe_merges)}
        self.byte_enc = self._bytes_to_unicode()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self.cache = {}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        )

    def _download(self, d):
        import urllib.request
        os.makedirs(d, exist_ok=True)
        base = "https://huggingface.co/openai-community/gpt2/resolve/main"
        for f in ["vocab.json", "merges.txt"]:
            url = f"{base}/{f}"
            p = os.path.join(d, f)
            if not os.path.exists(p):
                print(f"    Downloading {f}...")
                urllib.request.urlretrieve(url, p)

    def _bytes_to_unicode(self):
        bs = (list(range(ord('!'), ord('~')+1))
              + list(range(ord('\xa1'), ord('\xac')+1))
              + list(range(ord('\xae'), ord('\xff')+1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        if len(word) < 2:
            return word

        while True:
            pairs = set()
            prev = word[0]
            for ch in word[1:]:
                pairs.add((prev, ch))
                prev = ch
            if not pairs:
                break
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        self.cache[token] = word
        return word

    def encode(self, text):
        tokens = []
        for match in re.findall(self.pat, text):
            byte_encoded = ''.join(self.byte_enc[b] for b in match.encode('utf-8'))
            for t in self._bpe(byte_encoded):
                if t in self.encoder:
                    tokens.append(self.encoder[t])
        return tokens

    def decode(self, ids):
        text = ''.join(self.decoder.get(t, '') for t in ids)
        ba = bytearray([self.byte_dec.get(c, ord(c) % 256) for c in text])
        return ba.decode('utf-8', errors='replace')


# ============================================================================
# Pure NumPy GPT-2 Inference Engine
# ============================================================================
class GPT2Engine:
    """Full GPT-2-small in pure NumPy."""

    def __init__(self, weights, use_relu=False):
        self.w = weights
        self.use_relu = use_relu
        self.kv_cache = None
        self.stats = {
            'zero_mult_skipped': 0,
            'total_mults': 0,
            'gelu_zeros': 0,
            'gelu_total': 0,
            'tokens': 0,
            'act_name': 'ReLU' if self.use_relu else 'GELU'
        }

    def reset(self):
        self.kv_cache = [{} for _ in range(NUM_LAYERS)]
        self.stats = {
            'zero_mult_skipped': 0,
            'total_mults': 0,
            'gelu_zeros': 0,
            'gelu_total': 0,
            'tokens': 0,
            'act_name': 'ReLU' if self.use_relu else 'GELU'
        }

    def _ln(self, x, g, b, eps=1e-5):
        m = np.mean(x, axis=-1, keepdims=True)
        v = np.var(x, axis=-1, keepdims=True)
        return g * (x - m) / np.sqrt(v + eps) + b

    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def _relu(self, x):
        """ReLU: exact zero for all negatives → massive zero-skip savings."""
        return np.maximum(0, x)

    def _activation(self, x):
        """Use ReLU or GELU based on engine config."""
        return self._relu(x) if self.use_relu else self._gelu(x)

    def _softmax(self, x, axis=-1):
        mx = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - mx)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _mm(self, a, b):
        """Matmul with accurate zero-skip tracking.
        
        For y = a @ b where a is (D,) and b is (D, M):
        - Total scalar multiplications = D * M
        - Each zero in a[i] skips M multiplies (entire row of b)
        - Each zero in b[i,j] skips 1 multiply
        - The Verilog zero_detect_mult skips when EITHER operand is zero
        """
        if b.ndim == 2:
            D, M = b.shape
        else:
            D = b.shape[0]
            M = 1
        
        total_mults = int(D * M)
        
        # Count zeros in input vector (each saves M multiplies)
        a_zeros = int(np.sum(np.abs(a.flatten()) < Q88_ZERO_THRESH))
        skipped_from_a = a_zeros * M
        
        # Count zeros in weight matrix (each saves 1 multiply)
        b_zeros = int(np.sum(np.abs(b.flatten()) < Q88_ZERO_THRESH))
        
        # Don't double-count: where both a and b are zero
        # Approximate: skipped = skipped_from_a + b_zeros - overlap
        # overlap ≈ (a_zeros/D) * b_zeros (probabilistic)
        overlap = int((a_zeros / max(D, 1)) * b_zeros) if a_zeros > 0 else 0
        total_skipped = min(skipped_from_a + b_zeros - overlap, total_mults)
        
        self.stats['zero_mult_skipped'] += total_skipped
        self.stats['total_mults'] += total_mults
        
        return a @ b

    def _attention(self, x, li, pos):
        p = f'h.{li}'
        qkv = self._mm(x, self.w[f'{p}.attn.c_attn.weight']) + self.w[f'{p}.attn.c_attn.bias']
        q, k, v = np.split(qkv, 3)
        q = q.reshape(NUM_HEADS, HEAD_DIM)
        k = k.reshape(NUM_HEADS, HEAD_DIM)
        v = v.reshape(NUM_HEADS, HEAD_DIM)

        c = self.kv_cache[li]
        if 'k' not in c:
            c['k'] = k[np.newaxis]
            c['v'] = v[np.newaxis]
        else:
            c['k'] = np.concatenate([c['k'], k[np.newaxis]], axis=0)
            c['v'] = np.concatenate([c['v'], v[np.newaxis]], axis=0)

        seq_len = c['k'].shape[0]
        scores = np.einsum('hd,shd->hs', q, c['k']) / np.sqrt(HEAD_DIM)
        self.stats['total_mults'] += NUM_HEADS * seq_len * HEAD_DIM
        weights = self._softmax(scores, axis=-1)
        out = np.einsum('hs,shd->hd', weights, c['v']).reshape(-1)
        return self._mm(out, self.w[f'{p}.attn.c_proj.weight']) + self.w[f'{p}.attn.c_proj.bias']

    def _ffn(self, x, li):
        p = f'h.{li}'
        h = self._mm(x, self.w[f'{p}.mlp.c_fc.weight']) + self.w[f'{p}.mlp.c_fc.bias']
        h = self._activation(h)
        # Track activation zeros — ReLU produces exact zeros, GELU near-zeros
        gz = int(np.sum(np.abs(h) < Q88_ZERO_THRESH))
        self.stats['gelu_zeros'] += gz
        self.stats['gelu_total'] += h.size
        # The GELU zeros flow into the next matmul where they'll be tracked by _mm
        return self._mm(h, self.w[f'{p}.mlp.c_proj.weight']) + self.w[f'{p}.mlp.c_proj.bias']

    def forward(self, token_id, position):
        self.stats['tokens'] += 1
        x = self.w['wte.weight'][token_id] + self.w['wpe.weight'][position]
        for li in range(NUM_LAYERS):
            p = f'h.{li}'
            n = self._ln(x, self.w[f'{p}.ln_1.weight'], self.w[f'{p}.ln_1.bias'])
            x = x + self._attention(n, li, position)
            n = self._ln(x, self.w[f'{p}.ln_2.weight'], self.w[f'{p}.ln_2.bias'])
            x = x + self._ffn(n, li)
        x = self._ln(x, self.w['ln_f.weight'], self.w['ln_f.bias'])
        return self._mm(x, self.w['wte.weight'].T)

    def generate(self, input_ids, max_new=100, temperature=0.8, top_k=40,
                 callback=None, stop_at_sentence=True, tokenizer=None):
        self.reset()
        gen = list(input_ids)
        generated_text = ""

        # Process prompt
        for i, tid in enumerate(gen):
            logits = self.forward(tid, i)

        # Generate
        for step in range(max_new):
            scaled = logits / max(temperature, 1e-8)
            if top_k > 0:
                idx = np.argsort(scaled)[-top_k:]
                mask = np.full_like(scaled, -1e10)
                mask[idx] = scaled[idx]
                scaled = mask
            probs = self._softmax(scaled)
            probs = np.clip(probs, 0, None)
            probs /= probs.sum()
            nxt = int(np.random.choice(len(probs), p=probs))
            gen.append(nxt)

            # Decode this token for sentence detection
            if tokenizer:
                tok_text = tokenizer.decode([nxt])
                generated_text += tok_text

            if callback:
                callback(nxt, step, self.stats.copy())

            # Stop at end-of-text token
            if nxt == 50256:
                break

            # Stop at sentence boundary (after min 15 tokens)
            if stop_at_sentence and step >= 15 and generated_text:
                # Check for sentence-ending punctuation followed by space/newline
                stripped = generated_text.rstrip()
                if (stripped.endswith('.') or stripped.endswith('!') or
                    stripped.endswith('?') or stripped.endswith('\n\n')):
                    break

            logits = self.forward(nxt, len(gen) - 1)

        return gen[len(input_ids):]


# ============================================================================
# Terminal UI
# ============================================================================
class ChatUI:
    """Colored terminal chat interface."""

    # ANSI colors
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[97m"
    RED     = "\033[31m"
    BG_DARK = "\033[48;5;235m"

    @staticmethod
    def banner():
        print(f"""
{ChatUI.CYAN}{ChatUI.BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    ██████╗ ██╗████████╗██████╗ ██╗   ██╗██████╗ ██╗████████╗ ║
    ║    ██╔══██╗██║╚══██╔══╝██╔══██╗╚██╗ ██╔╝██╔══██╗██║╚══██╔══╝ ║
    ║    ██████╔╝██║   ██║   ██████╔╝ ╚████╔╝ ██████╔╝██║   ██║    ║
    ║    ██╔══██╗██║   ██║   ██╔══██╗  ╚██╔╝  ██╔══██╗██║   ██║    ║
    ║    ██████╔╝██║   ██║   ██████╔╝   ██║   ██████╔╝██║   ██║    ║
    ║    ╚═════╝ ╚═╝   ╚═╝   ╚═════╝    ╚═╝   ╚═════╝ ╚═╝   ╚═╝    ║
    ║                                                              ║
    ║         {ChatUI.YELLOW}GPT-2 Chat on Custom GPU{ChatUI.CYAN}                           ║
    ║         {ChatUI.DIM}{ChatUI.WHITE}124M params · 12 layers · 768-dim{ChatUI.CYAN}{ChatUI.BOLD}                ║
    ║         {ChatUI.DIM}{ChatUI.WHITE}Q8.8 Fixed-Point · Zero-Skip · INT8 Parallel{ChatUI.CYAN}{ChatUI.BOLD}    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{ChatUI.RESET}""")

    @staticmethod
    def system_msg(msg):
        print(f"  {ChatUI.DIM}{ChatUI.CYAN}[system]{ChatUI.RESET} {ChatUI.DIM}{msg}{ChatUI.RESET}")

    @staticmethod
    def user_prompt():
        return input(f"\n  {ChatUI.GREEN}{ChatUI.BOLD}You >{ChatUI.RESET} ")

    @staticmethod
    def gpu_header():
        print(f"\n  {ChatUI.MAGENTA}{ChatUI.BOLD}BitbyBit GPU >{ChatUI.RESET} ", end="", flush=True)

    @staticmethod
    def gpu_token(text):
        print(f"{ChatUI.WHITE}{text}{ChatUI.RESET}", end="", flush=True)

    @staticmethod
    def gpu_metrics(stats, gen_time, num_tokens):
        total_mults = max(stats['total_mults'], 1)
        skipped = stats['zero_mult_skipped']
        zero_rate = (skipped / total_mults) * 100
        gelu_zero_rate = (stats['gelu_zeros'] / max(stats['gelu_total'], 1)) * 100

        est_cycles = num_tokens * CYCLES_PER_TOKEN_768
        est_latency = est_cycles / (GPU_CLK_MHZ * 1e6) * 1000  # ms
        boost = 1 / max(1 - zero_rate/100, 0.01)
        effective_cycles = int(est_cycles / boost)
        effective_lat = effective_cycles / (GPU_CLK_MHZ * 1e6) * 1000

        print(f"\n\n  {ChatUI.DIM}{'─' * 62}{ChatUI.RESET}")
        print(f"  {ChatUI.YELLOW}{ChatUI.BOLD}⚡ GPU Performance{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Tokens generated:     {ChatUI.WHITE}{num_tokens}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Est. GPU cycles:      {ChatUI.WHITE}{est_cycles:,}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Est. latency @100MHz: {ChatUI.WHITE}{est_latency:.1f} ms{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Multiplications:      {ChatUI.WHITE}{total_mults:,}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Zero-skip rate:       {ChatUI.CYAN}{zero_rate:.1f}%{ChatUI.RESET} {ChatUI.DIM}({skipped:,} mults skipped){ChatUI.RESET}")
        act_label = stats.get('act_name', 'Activation')
        print(f"  {ChatUI.DIM}├─ {act_label} sparsity:  {ChatUI.CYAN}{gelu_zero_rate:.1f}%{ChatUI.RESET} {ChatUI.DIM}(main zero source){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Effective cycles:     {ChatUI.GREEN}{effective_cycles:,}{ChatUI.RESET} {ChatUI.DIM}(after zero-skip){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Effective latency:    {ChatUI.GREEN}{effective_lat:.1f} ms{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Throughput boost:     {ChatUI.GREEN}{boost:.2f}x{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}└─ Python wall-clock:    {ChatUI.WHITE}{gen_time:.1f}s{ChatUI.RESET} {ChatUI.DIM}(NumPy on CPU){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}{'─' * 62}{ChatUI.RESET}")


# ============================================================================
# Main Chat Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="BitbyBit GPU - GPT-2 Chat")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens per response (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-stop", action="store_true",
                        help="Don't stop at sentence boundaries")
    parser.add_argument("--relu", action="store_true",
                        help="Use ReLU instead of GELU (50%%+ activation sparsity)")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(root, "weights", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Banner
    ChatUI.banner()

    # Load tokenizer
    ChatUI.system_msg("Loading GPT-2 BPE tokenizer...")
    tokenizer = GPT2Tokenizer(cache_dir)
    ChatUI.system_msg("Tokenizer ready (50,257 vocab)")

    # Load model weights
    ChatUI.system_msg("Loading GPT-2-small weights (124M params)...")
    t0 = time.time()
    raw_weights = download_gpt2_weights(cache_dir)
    load_time = time.time() - t0
    ChatUI.system_msg(f"Weights loaded in {load_time:.1f}s")

    # Initialize engine
    engine = GPT2Engine(raw_weights, use_relu=args.relu)
    act_name = "ReLU" if args.relu else "GELU"
    ChatUI.system_msg(f"NumPy inference engine initialized (activation: {act_name})")
    if args.relu:
        ChatUI.system_msg("*** ReLU MODE: Exact zeros for negatives → high zero-skip! ***")
    ChatUI.system_msg("GPU features: Zero-Skip | Variable-Precision ALU | Pipelined")
    print()
    ChatUI.system_msg("Type your message and press Enter. Type 'quit' to exit.")
    ChatUI.system_msg("Type '/stats' for detailed GPU statistics.")

    conversation_history = []
    total_tokens_generated = 0
    total_gpu_cycles = 0

    while True:
        try:
            user_input = ChatUI.user_prompt()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {ChatUI.DIM}Goodbye!{ChatUI.RESET}\n")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print(f"\n  {ChatUI.DIM}Goodbye!{ChatUI.RESET}\n")
            break

        if user_input.lower() == '/stats':
            est = total_gpu_cycles
            lat = est / (GPU_CLK_MHZ * 1e6) * 1000
            print(f"\n  {ChatUI.YELLOW}{ChatUI.BOLD}Session Statistics:{ChatUI.RESET}")
            print(f"  Total tokens generated: {total_tokens_generated}")
            print(f"  Est. total GPU cycles:  {est:,}")
            print(f"  Est. total latency:     {lat:.1f} ms")
            continue

        # Tokenize input
        input_ids = tokenizer.encode(user_input)
        if not input_ids:
            ChatUI.system_msg("Could not tokenize input, try again.")
            continue

        # Generate response
        ChatUI.gpu_header()
        gen_tokens = []
        last_stats = {}

        def on_token(token_id, step, stats):
            nonlocal last_stats
            text = tokenizer.decode([token_id])
            ChatUI.gpu_token(text)
            gen_tokens.append(token_id)
            last_stats = stats

        t_start = time.time()
        try:
            engine.generate(
                input_ids,
                max_new=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                callback=on_token,
                stop_at_sentence=not args.no_stop,
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"\n  {ChatUI.RED}Error: {e}{ChatUI.RESET}")
            continue

        gen_time = time.time() - t_start
        num_gen = len(gen_tokens)

        total_tokens_generated += num_gen
        total_gpu_cycles += num_gen * CYCLES_PER_TOKEN_768

        if last_stats:
            ChatUI.gpu_metrics(last_stats, gen_time, num_gen)


if __name__ == "__main__":
    main()
