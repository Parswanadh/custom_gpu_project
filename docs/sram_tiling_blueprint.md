# SRAM Tiling and Interconnect Blueprint: 128-Banked 20MB System

## 1. Bank Configuration
To achieve a total of 20MB of on-chip SRAM, we utilize a 128-banked structure.
- **Total Capacity:** 20MB (20,971,520 bytes)
- **Number of Banks:** 128
- **Size per Bank:** 160KB (163,840 bytes) per bank.

## 2. Crossbar Interconnect Strategy
The crossbar maps 16 Compute Units (CUs) to 128 SRAM banks. To minimize contention and maintain high throughput:

### Mapping Scheme
- **Interleaved Addressing:** Use low-order interleaving (bit-slicing) to distribute contiguous memory blocks across all 128 banks. This ensures that parallel memory access patterns (e.g., SIMD vector loads) are distributed uniformly across banks.
- **Access Routing:** A 16x128 non-blocking crossbar switch is employed.
- **Contention Management:**
    - **Buffered Arbiters:** Each bank entry has a local arbiter to queue requests.
    - **Bank Conflicts:** If multiple CUs target the same bank, a round-robin arbitration policy is applied to ensure fairness and prevent starvation.

## 3. Latency Estimates
The bank-arbitration logic is critical path. Based on our 16x128 scale:

| Logic Stage | Estimated Latency (Cycles) |
| :--- | :--- |
| CU Request Issue | 1 |
| Crossbar Traversal | 1 |
| Bank Arbitration/Conflict Check | 2 |
| SRAM Read Access | 3 |
| **Total** | **7 cycles (pipelined)** |
