# BitbyBit Detailed Architecture Diagrams

This document contains detailed, component-level architectural diagrams for the BitbyBit custom Transformer GPU. These diagrams are intended for RTL engineers, compiler writers, and architects needing to understand the exact datapath, memory flow, and hardware accelerator interactions.

---

## 1. Top-Level System-on-Chip (SoC) Topology

This diagram shows the complete integration of the BitbyBit accelerator within the host environment, highlighting the command interface, DMA memory pathways, and multi-banked SRAM structure.

```mermaid
graph TD
    classDef host fill:#37474F,stroke:#CFD8DC,stroke-width:2px,color:#fff;
    classDef interface fill:#1A237E,stroke:#8C9EFF,stroke-width:2px,color:#fff;
    classDef control fill:#004D40,stroke:#69F0AE,stroke-width:2px,color:#fff;
    classDef compute fill:#B71C1C,stroke:#FF8A80,stroke-width:2px,color:#fff;
    classDef memory fill:#E65100,stroke:#FFD180,stroke-width:2px,color:#fff;

    subgraph Host ["Host Device"]
        CPU[Host CPU / Driver]:::host
        DRAM[(System DDR)]:::host
    end

    subgraph BitbyBit ["BitbyBit Accelerator Top"]
        direction TB
        
        %% Interfaces
        AXIS[AXI4-Lite Target Interface]:::interface
        AXIM[AXI4 Master DMA Engine]:::interface
        
        %% Control
        CFG[Configuration & Status Registers]:::control
        CMD[Command Processor FSM]:::control
        Perf[Performance & Telemetry Counters]:::control
        
        %% Central Memory
        SRAM[(Multi-Bank Scratchpad SRAM 4KB)]:::memory
        
        %% Compute Clusters
        SA[Systolic Array Engine]:::compute
        TF[Tiled FlashAttention Controller]:::compute
        VEC[Vector / Activation Unit]:::compute
        
        %% Routing
        CPU -- "MMIO / Config" --> AXIS
        AXIS --> CFG
        CFG -- "Start / PC" --> CMD
        CMD -- "Fetch Instr" --> SRAM
        CMD -- "Trigger DMA" --> AXIM
        
        AXIM -- "Bursts" --> DRAM
        AXIM -- "Fill/Spill" --> SRAM
        
        CMD -- "Execute Matrix" --> SA
        CMD -- "Execute Attention" --> TF
        CMD -- "Execute Vector" --> VEC
        
        SA <--> SRAM
        TF <--> SRAM
        VEC <--> SRAM
    end
```

---

## 2. Transformer NanoGPT Datapath Pipeline

This diagram traces the exact, cycle-accurate flow of a token passing through a single Transformer block in the BitbyBit engine. It highlights where specific hardware accelerators (like the Online Softmax and LUTs) sit within the pipeline.

```mermaid
flowchart LR
    classDef data fill:#424242,stroke:#fff,color:#fff;
    classDef compute fill:#B71C1C,stroke:#FF8A80,color:#fff;
    classDef memory fill:#E65100,stroke:#FFD180,color:#fff;
    classDef math fill:#4A148C,stroke:#EA80FC,color:#fff;

    TokenIn([Token & Pos ID]):::data --> EmbLookup[Embedding Lookup]:::memory
    EmbLookup --> ResAdd1((+)):::math
    EmbLookup --> LN1[LayerNorm 1]:::compute
    
    subgraph AttentionBlock ["Hardware Accelerated Attention"]
        direction TB
        LN1 --> QKV[Q, K, V Linear Projections]:::compute
        QKV --> TiledCtrl[Tiled FlashAttention Controller]:::compute
        
        TiledCtrl -- "O(Tile) K/V Reads" --> KVCache[(Paged KV Cache)]:::memory
        
        TiledCtrl --> |"Q_tile, K_tile, V_tile"| OSM[Streaming Online Softmax]:::compute
        
        subgraph SoftmaxInner ["Online Softmax Fused Core"]
            direction LR
            Max[Running Max]:::math --> ExpCorr[EXP LUT: Correction]:::math
            Score[Current Score] --> ExpNew[EXP LUT: New Term]:::math
            ExpCorr & ExpNew --> VAcc[V Vector Accumulator]:::math
        end
        OSM -.-> SoftmaxInner
        
        OSM --> |"Fused Attn_Out"| WO[Output Projection Wo]:::compute
    end
    
    WO --> ResAdd1
    
    ResAdd1 --> LN2[LayerNorm 2]:::compute
    
    subgraph FFNBlock ["Feed Forward Network"]
        direction TB
        LN2 --> W1[Linear W1]:::compute
        W1 --> GELU[GELU Activation LUT]:::math
        GELU --> W2[Linear W2]:::compute
    end
    
    W2 --> ResAdd2((+)):::math
    ResAdd1 --> ResAdd2
    
    ResAdd2 --> FinalLN[Final LayerNorm]:::compute
    FinalLN --> Logits([Logits Projection & Argmax]):::data
```

---

## 3. Paged KV Cache Memory Subsystem

To break the memory wall, BitbyBit treats the KV Cache like a virtual memory system. This diagram details the MMU-like translation process that allows zero-fragmentation sliding-window attention.

```mermaid
flowchart TD
    classDef input fill:#37474F,color:#fff;
    classDef mmu fill:#004D40,stroke:#69F0AE,color:#fff;
    classDef ram fill:#E65100,stroke:#FFD180,color:#fff;

    LogicalID([Logical Token ID]):::input --> PagedMMU[Paged Attention MMU]:::mmu
    
    subgraph PageManagement ["Page Management Unit (Hardware)"]
        direction LR
        Alloc[Stack-based Page Allocator]:::mmu
        PageTable[Virtual-to-Physical Page Table]:::mmu
        
        Alloc -- "Pops Free Physical Page" --> PageTable
        Alloc -. "Pushes Evicted Page" .-> Alloc
    end
    
    PagedMMU --> |"Write Req (New Token)"| Alloc
    PagedMMU --> |"Read Req (Context)"| PageTable
    
    PageTable -- "Translated Physical Page ID" --> AddrCalc[Address Calculator Hardware]:::mmu
    
    AddrCalc -- "Phy_Addr = (PageID * Size) + Offset" --> KVSRAM[(Physical KV Cache SRAM Banks)]:::ram
```

---

## 4. Compute Microarchitecture Accelerators

A deep dive into the two major hardware innovations that drastically reduce multiplier logic area and power consumption.

### 4A. Multiplier-Free Ternary MAC (BitNet b1.58)
Replaces power-hungry 8x8 or 16x16 silicon multipliers with a simple multiplexer, shifting the paradigm from multiplication to conditional addition.

```mermaid
flowchart TB
    classDef input fill:#37474F,color:#fff;
    classDef logic fill:#01579B,stroke:#40C4FF,color:#fff;
    classDef math fill:#4A148C,stroke:#EA80FC,color:#fff;

    Act([8-bit Activation]):::input --> PosNeg[Two's Complement Negator]:::logic
    Act --> MUX
    PosNeg -- "-Activation" --> MUX{2-bit Multiplexer}:::logic
    
    Weight([2-bit Packed Weight]):::input --> |"00, 01, 10"| MUX
    
    MUX -- "If 00" --> Zero[0]:::math
    MUX -- "If 01" --> Pos[+Activation]:::math
    MUX -- "If 10" --> Neg[-Activation]:::math
    
    Zero & Pos & Neg --> AccAdd((Accumulator Add)):::math
```

### 4B. 2:4 Structured Sparsity PE
Enforces a 50% sparsity pattern, doubling the effective throughput of the matrix math engine by skipping predetermined zero-weights.

```mermaid
flowchart TB
    classDef input fill:#37474F,color:#fff;
    classDef logic fill:#01579B,stroke:#40C4FF,color:#fff;
    classDef math fill:#4A148C,stroke:#EA80FC,color:#fff;

    subgraph Offline ["Offline / Memory Load Phase"]
        DenseW([4x Dense Weights]):::input --> Encoder[Sparsity Encoder]:::logic
        Encoder -- "Drops 2 Smallest" --> NonZeroW([2x Non-Zero Weights]):::input
        Encoder -- "Generates Mask" --> Mask([4-bit Sparsity Mask]):::input
    end
    
    subgraph ActiveInference ["Active Inference Pipeline"]
        Acts([4x 8-bit Activations]):::input --> SMUX{Activation Selector MUX}:::logic
        Mask --> |"Control Signal"| SMUX
        
        SMUX -- "Routes 2 Corresponding Acts" --> Mult1[*]:::math & Mult2[*]:::math
        NonZeroW --> Mult1 & Mult2
        
        Mult1 & Mult2 --> SAcc((Partial Sum Adder)):::math
    end
```
