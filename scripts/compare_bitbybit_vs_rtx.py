import torch
import torch.nn as nn
import time
import numpy as np

class SimpleTransformerBlock(nn.Module):
    def __init__(self, embd_dim, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = nn.MultiheadAttention(embd_dim, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Linear(4 * embd_dim, embd_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, n_layer=12, embd_dim=768, n_head=12, vocab_size=50257):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embd_dim)
        self.blocks = nn.Sequential(*[SimpleTransformerBlock(embd_dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embd_dim)
        self.head = nn.Linear(embd_dim, vocab_size)

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

def benchmark_rtx():
    print("--- Benchmarking Local GPU (Simulated Workload) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # GPT-2 Small Config (approx 85M-124M params)
    model = MiniGPT(n_layer=12, embd_dim=768, n_head=12).to(device)
    model.eval()

    # Input: Batch size 1, Sequence length 128
    input_ids = torch.randint(0, 50257, (1, 128)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)

    # Latency Measurement
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            start = time.perf_counter()
            _ = model(input_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    avg_latency_ms = np.mean(latencies) * 1000
    throughput = 128 / (avg_latency_ms / 1000) # Tokens per second

    print(f"Avg Latency (128 tokens): {avg_latency_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    
    return {
        "latency_ms": avg_latency_ms,
        "throughput": throughput,
        "device": device if device == "cpu" else torch.cuda.get_device_name(0)
    }

def generate_comparison(rtx_data):
    # BitbyBit Constants
    bbb_freq_mhz = 200
    bbb_latency_cycles = 112
    bbb_throughput_tokens_sec = 2670000
    
    # Calculation for BitbyBit Latency in MS
    bbb_latency_ms = (bbb_latency_cycles / (bbb_freq_mhz * 1e6)) * 1000
    
    # Energy Calculation
    rtx_tdp = 250 # Estimated for high-end consumer GPU
    bbb_est_power = 10 # Estimated for specialized Ternary Hardware
    
    rtx_energy_per_token = rtx_tdp / rtx_data["throughput"]
    bbb_energy_per_token = bbb_est_power / bbb_throughput_tokens_sec
    
    improvement_throughput = bbb_throughput_tokens_sec / rtx_data["throughput"]
    improvement_energy = rtx_energy_per_token / bbb_energy_per_token

    print("\n--- BitbyBit vs GPU Comparison ---")
    print(f"Throughput Improvement: {improvement_throughput:.2f}x")
    print(f"Energy Efficiency Gain: {improvement_energy:.2f}x")
    
    return {
        "rtx": rtx_data,
        "bbb": {
            "latency_ms": bbb_latency_ms,
            "throughput": bbb_throughput_tokens_sec,
            "energy_per_token": bbb_energy_per_token
        },
        "metrics": {
            "throughput_gain": improvement_throughput,
            "energy_gain": improvement_energy
        }
    }

if __name__ == "__main__":
    rtx_results = benchmark_rtx()
    comparison = generate_comparison(rtx_results)
