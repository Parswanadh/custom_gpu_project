#!/usr/bin/env python3
"""
BitbyBit GPU — OPT-125M Chat with Native ReLU Zero-Skip
=========================================================
OPT-125M uses ReLU activation natively → 90%+ activation sparsity.
This demonstrates the real-world benefit of the GPU's zero_detect_mult.

Usage:
    python scripts/chat_opt.py
    python scripts/chat_opt.py --max-tokens 50 --temperature 0.7
"""

import os, sys, time, argparse, json, re
import numpy as np

# ============================================================================
# Constants
# ============================================================================
NUM_LAYERS = 12
EMBED_DIM = 768
NUM_HEADS = 12
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64
FFN_DIM = 3072
VOCAB_SIZE = 50272
MAX_POS = 2048

# GPU Performance Model
GPU_CLK_MHZ = 100
CYCLES_PER_TOKEN_768 = 150_000

# Q8.8 zero threshold
Q88_ZERO_THRESH = 0.5 / 256.0  # ~0.00195


# ============================================================================
# BPE Tokenizer (OPT uses same GPT-2 BPE tokenizer)
# ============================================================================
class OPTTokenizer:
    """GPT-2 BPE tokenizer for OPT."""

    def __init__(self, cache_dir):
        vocab_path = os.path.join(cache_dir, "vocab.json")
        merges_path = os.path.join(cache_dir, "merges.txt")

        # Download if missing
        base = "https://huggingface.co/facebook/opt-125m/resolve/main"
        import urllib.request
        for fname, url_name in [("vocab.json", "vocab.json"), ("merges.txt", "merges.txt")]:
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                print(f"  Downloading {fname}...")
                urllib.request.urlretrieve(f"{base}/{url_name}", fpath)

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [tuple(l.split()) for l in lines[1:] if l.strip() and len(l.split()) == 2]
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Special tokens
        self.eos_token_id = 2      # </s>
        self.bos_token_id = 2      # </s>  (OPT uses </s> as BOS too)
        self.pad_token_id = 1      # <pad>

    @staticmethod
    def _bytes_to_unicode():
        bs = list(range(ord('!'), ord('~') + 1)) + \
             list(range(ord('¡'), ord('¬') + 1)) + \
             list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def _get_pairs(self, word):
        return set(zip(word[:-1], word[1:]))

    def _bpe(self, token):
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return (token,)
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            a, b = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(a, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == b:
                    new_word.append(a + b)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            pairs = self._get_pairs(word)
            if not pairs:
                break
        return word

    def encode(self, text):
        """Encode to token IDs. OPT prepends </s> (BOS token)."""
        tokens = []
        for match in re.finditer(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+""", text):
            chunk = match.group()
            encoded = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            bpe_tokens = self._bpe(encoded)
            tokens.extend(self.encoder.get(t, 0) for t in bpe_tokens)
        # OPT prepends </s> as BOS
        return [self.bos_token_id] + tokens

    def decode(self, ids):
        text = "".join(self.decoder.get(i, "") for i in ids)
        raw = bytearray(self.byte_decoder.get(c, ord(c)) for c in text)
        return raw.decode("utf-8", errors="replace")


# ============================================================================
# OPT-125M Inference Engine (Pure NumPy, Native ReLU)
# ============================================================================
class OPTEngine:
    """Full OPT-125M in pure NumPy with ReLU everywhere + Q8.8 quantization."""

    def __init__(self, weights, relu_everywhere=True, use_ffn_relu=True, use_q88=True):
        self.w = weights
        self.relu_everywhere = relu_everywhere
        self.use_ffn_relu = use_ffn_relu
        self.use_q88 = use_q88
        self.kv_cache = None
        self._init_stats()

        # Pre-quantize weights to Q8.8 if enabled
        if self.use_q88:
            self.w = {k: self._q88(v) for k, v in self.w.items()}

    def _init_stats(self):
        self.stats = {
            'zero_mult_skipped': 0,
            'total_mults': 0,
            'relu_zeros': 0,
            'relu_total': 0,
            'q88_zeros': 0,
            'q88_total': 0,
            'tokens': 0,
        }

    def reset(self):
        self.kv_cache = [{} for _ in range(NUM_LAYERS)]
        self._init_stats()

    def _ln(self, x, g, b, eps=1e-5):
        m = np.mean(x, axis=-1, keepdims=True)
        v = np.var(x, axis=-1, keepdims=True)
        return g * (x - m) / np.sqrt(v + eps) + b

    def _relu(self, x):
        """ReLU: exact zero for all negatives → massive zero-skip."""
        out = np.maximum(0, x)
        # Track ReLU zeros
        rz = int(np.sum(out == 0))
        self.stats['relu_zeros'] += rz
        self.stats['relu_total'] += out.size
        return out

    def _q88(self, x):
        """Quantize to Q8.8 fixed-point (8-bit integer + 8-bit fraction).
        
        Resolution: 1/256 = 0.00390625
        Range: -128.0 to +127.99609375
        Values smaller than 1/512 round to exact zero.
        """
        clipped = np.clip(x, -128.0, 127.99609375)
        quantized = np.round(clipped * 256.0) / 256.0
        return quantized.astype(np.float32)

    def _activate(self, x):
        """Apply ReLU + optional Q8.8 quantization."""
        x = self._relu(x)
        if self.use_q88:
            before = x
            x = self._q88(x)
            qz = int(np.sum(np.abs(x) < Q88_ZERO_THRESH))
            self.stats['q88_zeros'] += qz
            self.stats['q88_total'] += x.size
        return x

    def _softmax(self, x, axis=-1):
        mx = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - mx)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _mm(self, a, b):
        """Matmul with zero-skip tracking.

        In Q8.8 mode, zeros are exact (multiples of 1/256).
        The hardware zero_detect_mult skips when EITHER operand is zero.
        """
        if b.ndim == 2:
            D, M = b.shape
        else:
            D = b.shape[0]
            M = 1

        total_mults = int(D * M)

        # Use exact zero check (Q8.8 values are exact multiples of 1/256)
        a_flat = a.flatten()
        a_zeros = int(np.sum(a_flat == 0))
        skipped_from_a = a_zeros * M

        b_flat = b.flatten()
        b_zeros = int(np.sum(b_flat == 0))

        # Avoid double-counting overlap
        if a_zeros > 0 and b_zeros > 0:
            overlap = int((a_zeros / max(D, 1)) * b_zeros)
        else:
            overlap = 0
        total_skipped = min(skipped_from_a + b_zeros - overlap, total_mults)

        self.stats['zero_mult_skipped'] += total_skipped
        self.stats['total_mults'] += total_mults

        return a @ b

    def _attention(self, x, li, pos):
        """OPT attention with ReLU after every projection."""
        p = f'model.decoder.layers.{li}.self_attn'

        # Q, K, V projections — apply ReLU after each!
        q = self._mm(x, self.w[f'{p}.q_proj.weight'].T) + self.w[f'{p}.q_proj.bias']
        k = self._mm(x, self.w[f'{p}.k_proj.weight'].T) + self.w[f'{p}.k_proj.bias']
        v = self._mm(x, self.w[f'{p}.v_proj.weight'].T) + self.w[f'{p}.v_proj.bias']

        if self.relu_everywhere:
            q = self._activate(q)
            k = self._activate(k)
            v = self._activate(v)

        q = q.reshape(NUM_HEADS, HEAD_DIM)
        k = k.reshape(NUM_HEADS, HEAD_DIM)
        v = v.reshape(NUM_HEADS, HEAD_DIM)

        # KV cache
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

        out = self._mm(out, self.w[f'{p}.out_proj.weight'].T) + self.w[f'{p}.out_proj.bias']
        if self.relu_everywhere:
            out = self._activate(out)
        return out

    def _ffn(self, x, li):
        """OPT FFN: fc1 → (ReLU) → fc2 (with optional ReLU after fc2 too)."""
        p = f'model.decoder.layers.{li}'

        # fc1
        h = self._mm(x, self.w[f'{p}.fc1.weight'].T) + self.w[f'{p}.fc1.bias']
        # Apply ReLU if enabled (native OPT uses ReLU here)
        if self.use_ffn_relu:
            h = self._activate(h)
        elif self.use_q88:
            h = self._q88(h)  # Still quantize to Q8.8

        # fc2
        out = self._mm(h, self.w[f'{p}.fc2.weight'].T) + self.w[f'{p}.fc2.bias']
        if self.relu_everywhere:
            out = self._activate(out)
        return out

    def forward(self, token_id, position):
        self.stats['tokens'] += 1

        # Token + position embeddings (quantize if Q8.8)
        x = self.w['model.decoder.embed_tokens.weight'][token_id] + \
            self.w['model.decoder.embed_positions.weight'][position + 2]
        if self.use_q88:
            x = self._q88(x)

        for li in range(NUM_LAYERS):
            p = f'model.decoder.layers.{li}'

            # Self-attention block (pre-norm)
            n = self._ln(x, self.w[f'{p}.self_attn_layer_norm.weight'],
                         self.w[f'{p}.self_attn_layer_norm.bias'])
            if self.use_q88:
                n = self._q88(n)
            x = x + self._attention(n, li, position)

            # FFN block (pre-norm)
            n = self._ln(x, self.w[f'{p}.final_layer_norm.weight'],
                         self.w[f'{p}.final_layer_norm.bias'])
            if self.use_q88:
                n = self._q88(n)
            x = x + self._ffn(n, li)

        # Final layer norm
        x = self._ln(x, self.w['model.decoder.final_layer_norm.weight'],
                     self.w['model.decoder.final_layer_norm.bias'])
        if self.use_q88:
            x = self._q88(x)

        # Project to vocab
        return self._mm(x, self.w['lm_head.weight'].T)

    def generate(self, input_ids, max_new=100, temperature=0.8, top_k=40,
                 callback=None, stop_at_sentence=True, tokenizer=None):
        self.reset()
        gen = list(input_ids)
        generated_text = ""

        # Process prompt tokens
        for i, tid in enumerate(gen):
            logits = self.forward(tid, i)

        # Generate new tokens
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

            if tokenizer:
                tok_text = tokenizer.decode([nxt])
                generated_text += tok_text

            if callback:
                callback(nxt, step, self.stats.copy())

            # Stop at EOS
            if nxt == 2:  # </s>
                break

            # Stop at sentence boundary (after min 15 tokens)
            if stop_at_sentence and step >= 15 and generated_text:
                stripped = generated_text.rstrip()
                if (stripped.endswith('.') or stripped.endswith('!') or
                    stripped.endswith('?') or stripped.endswith('\n\n')):
                    break

            logits = self.forward(nxt, len(gen) - 1)

        return gen[len(input_ids):]


# ============================================================================
# Chat UI
# ============================================================================
class ChatUI:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def banner():
        print(f"""
    {ChatUI.CYAN}╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD}  ██████╗ ██████╗ ████████╗       ██╗██████╗ ███████╗{ChatUI.CYAN}      ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD} ██╔═══██╗██╔══██╗╚══██╔══╝      ███║╚════██╗██╔════╝{ChatUI.CYAN}      ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD} ██║   ██║██████╔╝   ██║   █████╗╚██║ █████╔╝███████╗{ChatUI.CYAN}      ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD} ██║   ██║██╔═══╝    ██║   ╚════╝ ██║██╔═══╝ ╚════██║{ChatUI.CYAN}      ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD} ╚██████╔╝██║        ██║          ██║███████╗███████║{ChatUI.CYAN}      ║
    ║  {ChatUI.WHITE}{ChatUI.BOLD}  ╚═════╝ ╚═╝        ╚═╝          ╚═╝╚══════╝╚══════╝{ChatUI.CYAN}      ║
    ║                                                              ║
    ║  {ChatUI.WHITE}Meta OPT-125M on BitbyBit Custom GPU{ChatUI.CYAN}                       ║
    ║  {ChatUI.DIM}125M params · 12 layers · 768-dim{ChatUI.CYAN}                         ║
    ║  {ChatUI.GREEN}{ChatUI.BOLD}Q8.8 Fixed-Point · ReLU Zero-Skip · INT8 Parallel{ChatUI.CYAN}       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝{ChatUI.RESET}
""")

    @staticmethod
    def system_msg(msg):
        print(f"  {ChatUI.DIM}[system] {msg}{ChatUI.RESET}")

    @staticmethod
    def gpu_metrics(stats, gen_time, num_tokens):
        total_mults = max(stats['total_mults'], 1)
        skipped = stats['zero_mult_skipped']
        zero_rate = (skipped / total_mults) * 100
        relu_zero_rate = (stats['relu_zeros'] / max(stats['relu_total'], 1)) * 100
        q88_zero_rate = (stats['q88_zeros'] / max(stats['q88_total'], 1)) * 100

        est_cycles = num_tokens * CYCLES_PER_TOKEN_768
        est_latency = est_cycles / (GPU_CLK_MHZ * 1e6) * 1000
        boost = 1 / max(1 - zero_rate / 100, 0.01)
        effective_cycles = int(est_cycles / boost)
        effective_lat = effective_cycles / (GPU_CLK_MHZ * 1e6) * 1000

        print(f"\n\n  {ChatUI.DIM}{'─' * 66}{ChatUI.RESET}")
        print(f"  {ChatUI.YELLOW}{ChatUI.BOLD}⚡ GPU Performance (OPT-125M · ReLU Everywhere · Q8.8){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Tokens generated:     {ChatUI.WHITE}{num_tokens}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Est. GPU cycles:      {ChatUI.WHITE}{est_cycles:,}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Est. latency @100MHz: {ChatUI.WHITE}{est_latency:.1f} ms{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Multiplications:      {ChatUI.WHITE}{total_mults:,}{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ ReLU sparsity:        {ChatUI.GREEN}{ChatUI.BOLD}{relu_zero_rate:.1f}%{ChatUI.RESET} {ChatUI.DIM}(exact zeros from ReLU){ChatUI.RESET}")
        if stats['q88_total'] > 0:
            print(f"  {ChatUI.DIM}├─ Q8.8 quant zeros:     {ChatUI.GREEN}{q88_zero_rate:.1f}%{ChatUI.RESET} {ChatUI.DIM}(quantization-induced){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ {ChatUI.BOLD}Overall zero-skip:     {ChatUI.CYAN}{ChatUI.BOLD}{zero_rate:.1f}%{ChatUI.RESET} {ChatUI.DIM}({skipped:,} mults skipped){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Effective cycles:     {ChatUI.GREEN}{effective_cycles:,}{ChatUI.RESET} {ChatUI.DIM}(after zero-skip){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Effective latency:    {ChatUI.GREEN}{effective_lat:.1f} ms{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}├─ Throughput boost:     {ChatUI.GREEN}{ChatUI.BOLD}{boost:.2f}x{ChatUI.RESET}")
        print(f"  {ChatUI.DIM}└─ Python wall-clock:    {ChatUI.WHITE}{gen_time:.1f}s{ChatUI.RESET} {ChatUI.DIM}(NumPy on CPU){ChatUI.RESET}")
        print(f"  {ChatUI.DIM}{'─' * 66}{ChatUI.RESET}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="BitbyBit GPU — OPT-125M Chat (Native ReLU)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-stop", action="store_true", help="Don't stop at sentence boundaries")
    parser.add_argument("--relu-mode", choices=['none', 'ffn', 'all'], default='ffn',
                        help="ReLU mode: 'none'=no ReLU, 'ffn'=FFN only (native OPT), 'all'=everywhere")
    parser.add_argument("--no-q88", action="store_true", help="Disable Q8.8 quantization")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(root, "weights", "opt125m")
    os.makedirs(cache_dir, exist_ok=True)

    ChatUI.banner()

    # Load tokenizer
    ChatUI.system_msg("Loading OPT BPE tokenizer...")
    tokenizer = OPTTokenizer(cache_dir)
    ChatUI.system_msg(f"Tokenizer ready ({VOCAB_SIZE:,} vocab)")

    # Load weights
    npz_path = os.path.join(cache_dir, "opt125m_weights.npz")
    if not os.path.exists(npz_path):
        ChatUI.system_msg("ERROR: Weights not found. Run weight download first.")
        return

    ChatUI.system_msg("Loading OPT-125M weights (125M params)...")
    t0 = time.time()
    raw = np.load(npz_path, allow_pickle=True)
    weights = {k: raw[k] for k in raw.files}
    load_time = time.time() - t0
    ChatUI.system_msg(f"Weights loaded in {load_time:.1f}s")

    # Initialize engine
    ChatUI.system_msg("Quantizing weights to Q8.8 fixed-point..." if not args.no_q88 else "Using float32 precision")
    relu_everywhere = (args.relu_mode == 'all')
    use_ffn_relu = (args.relu_mode in ('ffn', 'all'))
    use_q88 = not args.no_q88
    engine = OPTEngine(weights, relu_everywhere=relu_everywhere, 
                       use_ffn_relu=use_ffn_relu, use_q88=use_q88)
    
    mode_label = {'none': 'No ReLU', 'ffn': 'ReLU in FFN (native)', 'all': 'ReLU EVERYWHERE'}[args.relu_mode]
    q88_label = 'Q8.8' if use_q88 else 'float32'
    ChatUI.system_msg(f"Engine: {mode_label} + {q88_label}")
    if use_q88:
        ChatUI.system_msg("★ Q8.8: small values quantize to exact zero!")
    if args.relu_mode in ('ffn', 'all'):
        ChatUI.system_msg("★ ReLU: negative activations become exact zero!")
    ChatUI.system_msg(f"GPU features: Zero-Skip | {q88_label} Fixed-Point | Pipelined")
    print()
    ChatUI.system_msg("Type your message and press Enter. Type 'quit' to exit.")
    ChatUI.system_msg("Type '/stats' for session statistics.")

    total_tokens_generated = 0
    total_gpu_cycles = 0

    while True:
        try:
            user_input = input(f"\n  {ChatUI.GREEN}You > {ChatUI.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {ChatUI.DIM}Goodbye!{ChatUI.RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "/quit"):
            print(f"\n  {ChatUI.DIM}Goodbye!{ChatUI.RESET}\n")
            break

        if user_input == "/stats":
            print(f"\n  {ChatUI.YELLOW}{ChatUI.BOLD}📊 Session Statistics{ChatUI.RESET}")
            print(f"  {ChatUI.DIM}├─ Total tokens: {total_tokens_generated}{ChatUI.RESET}")
            print(f"  {ChatUI.DIM}└─ Total GPU cycles: {total_gpu_cycles:,}{ChatUI.RESET}")
            continue

        # Tokenize
        input_ids = tokenizer.encode(user_input)

        # Generate with streaming
        print(f"\n  {ChatUI.CYAN}OPT-125M > {ChatUI.RESET}", end="", flush=True)
        generated_tokens = []

        def on_token(tok_id, step, stats):
            text = tokenizer.decode([tok_id])
            print(text, end="", flush=True)
            generated_tokens.append(tok_id)

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
            import traceback; traceback.print_exc()
            continue

        gen_time = time.time() - t_start
        num_tokens = len(generated_tokens)
        total_tokens_generated += num_tokens
        total_gpu_cycles += num_tokens * CYCLES_PER_TOKEN_768

        if num_tokens > 0:
            ChatUI.gpu_metrics(engine.stats, gen_time, num_tokens)


if __name__ == "__main__":
    main()
