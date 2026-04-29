# nano-vllm Architecture Graph

## The Full System — End to End

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER APPLICATION                                  │
│                                                                             │
│   from nanovllm import LLM, SamplingParams                                  │
│   llm = LLM("Qwen/Qwen3-0.6B")                                              │
│   llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256))    │
│                                                                             │
│   llm.py ← trivial subclass of LLMEngine                                    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LLM ENGINE  (llm_engine.py)                           │
│                                                                              │
│  The orchestrator. Owns the Scheduler, ModelRunner, and Tokenizer.           │
│                                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────────────┐   │
│  │  Tokenizer  │   │    Config    │   │  Worker Processes (rank 1,2,..)  │   │
│  │ (HuggingFace│   │  (config.py) │   │  Spawned via mp.Process          │   │
│  │  AutoToken.)│   │              │   │  Run ModelRunner.loop()          │   │
│  └──────┬──────┘   └──────────────┘   │  Wait on SharedMemory barrier    │   │
│         │                             └──────────────────────────────────┘   │
│         │                                                                    │
│  ┌──────▼──────────────────────────────────────────────────────────────────┐ │
│  │                     generate() main loop                                │ │
│  │                                                                         │ │
│  │  1. add_request(): tokenize prompt → create Sequence → enqueue          │ │
│  │  2. while not finished:                                                 │ │
│  │       step()  ─┐                                                        │ │
│  │                │                                                        │ │
│  │       ┌────────▼────────┐      ┌──────────────────┐                     │ │
│  │       │   Scheduler     │────▶│   ModelRunner    │                     │ │
│  │       │   .schedule()   │      │   .run()         │                     │ │
│  │       └────────┬────────┘      └────────┬─────────┘                     │ │
│  │                │                        │                               │ │
│  │                │  seqs, is_prefill      │  token_ids                    │ │
│  │                │                        │                               │ │
│  │       ┌────────▼────────────────────────▼─────────┐                     │ │
│  │       │   Scheduler.postprocess()                 │                     │ │
│  │       │   append token, check EOS, hash blocks    │                     │ │
│  │       └───────────────────────────────────────────┘                     │ │
│  │  3. decode token_ids → text via tokenizer                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Scheduler — The Traffic Cop (scheduler.py + block_manager.py)

```
                    ┌──────────────────────────────────────────┐
                    │              SCHEDULER                   │
                    │                                          │
  add_request() ──▶ │  ┌────────────┐      ┌───────────────┐  │
                    │  │   WAITING   │────▶│   RUNNING     │  │
                    │  │   (deque)   │      │   (deque)     │  │
                    │  │             │      │               │  │
                    │  │  Sequences  │      │  Sequences    │  │
                    │  │  needing    │      │  generating   │  │
                    │  │  prefill    │      │  tokens       │  │
                    │  └──────┬──────┘      └───────┬───────┘  │
                    │         │  allocate           │  check   │
                    │         │  KV blocks          │  memory  │
                    │         ▼                     ▼          │
                    │  ┌────────────────────────────────────┐  │
                    │  │         BLOCK MANAGER              │  │
                    │  │        (block_manager.py)          │  │
                    │  │                                    │  │
                    │  │  ┌─────┬─────┬─────┬─────┬─────┐   │  │
                    │  │  │ Blk │ Blk │ Blk │ Blk │ ... │   │  │
                    │  │  │  0  │  1  │  2  │  3  │     │   │  │
                    │  │  └─────┴─────┴─────┴─────┴─────┘   │  │
                    │  │  free_blocks: [4, 5, 6, 7, ...]    │  │
                    │  │  hash_to_block: {hash→id}          │  │
                    │  │         (prefix cache)             │  │
                    │  └────────────────────────────────────┘  │
                    │                                          │
                    │  .schedule() decides:                    │
                    │    Prefills waiting? → run prefill       │
                    │    Only running?     → run decode        │
                    │    Out of memory?    → preempt (evict    │
                    │                        back to WAITING)  │
                    │                                          │
                    │              ┌───────────┐               │
                    │              │ FINISHED  │               │
                    │              │ (removed) │               │
                    │              └───────────┘               │
                    └──────────────────────────────────────────┘

Sequence lifecycle:

  WAITING ──(prefill)──▶ RUNNING ──(decode loop)──▶ FINISHED
     ▲                      │
     └──────(preempt)───────┘   (if out of KV cache memory)
```

---

## Sequence — One Request's State (sequence.py)

```
┌─────────────────────────────────────────────────────────────┐
│  Sequence                                                   │
│                                                             │
│  token_ids: [1542, 389, 7201, | 4821, 992, ...]             │
│              ────────────────   ────────────────            │
│              prompt tokens      generated tokens            │
│              (num_prompt_tokens) (grows each step)          │
│                                                             │
│  block_table: [3, 7, 12]  ← physical KV cache block IDs     │
│                               (like an OS page table)       │
│                                                             │
│  status: WAITING → RUNNING → FINISHED                       │
│  temperature: 0.6                                           │
│  max_tokens: 256                                            │
│  num_cached_tokens: 512  ← prefix cache hit (skip these)    │
└─────────────────────────────────────────────────────────────┘
```

---

## ModelRunner — GPU Execution (model_runner.py)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MODEL RUNNER                                   │
│                                                                         │
│  Owns: Model (Qwen3ForCausalLM), Sampler, KV Cache, CUDA Graphs         │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  run(seqs, is_prefill)                                        │      │
│  │                                                               │      │
│  │  1. PREPARE                                                   │      │
│  │     ┌─────────────────────┐  ┌──────────────────────┐         │      │
│  │     │  prepare_prefill()  │  │  prepare_decode()    │         │      │
│  │     │                     │  │                      │         │      │
│  │     │  • Pack all prompt  │  │  • Extract last_token│         │      │
│  │     │    tokens together  │  │    from each seq     │         │      │
│  │     │  • Build cu_seqlens │  │  • Build block_tables│         │      │
│  │     │  • Compute slots    │  │  • Compute slots     │         │      │
│  │     │  • set_context()    │  │  • set_context()     │         │      │
│  │     └─────────────────────┘  └──────────────────────┘         │      │
│  │                                                               │      │
│  │  2. RUN MODEL                                                 │      │
│  │     ┌─────────────────────────────────────────────────┐       │      │
│  │     │  Prefill: normal forward pass                   │       │      │
│  │     │  Decode:  CUDA graph replay (pre-recorded)      │       │      │
│  │     │           at batch sizes 1,2,4,8,16,32,...      │       │      │
│  │     └──────────────────────┬──────────────────────────┘       │      │
│  │                            │ logits                           │      │
│  │  3. SAMPLE                 ▼                                  │      │
│  │     ┌─────────────────────────────────────────────────┐       │      │
│  │     │  Sampler (rank 0 only)                          │       │      │
│  │     │  logits → temperature → softmax → Gumbel-max    │       │      │
│  │     │  → token_ids                                    │       │      │
│  │     └─────────────────────────────────────────────────┘       │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  KV Cache (pre-allocated on GPU):                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Shape: [2, num_layers, num_blocks, block_size, kv_heads, dim]  │    │
│  │          K        V                                             │    │
│  │  Block 0: [████████████]  [████████████]   ← used by seq A      │    │
│  │  Block 1: [████████████]  [████████████]   ← used by seq B      │    │
│  │  Block 2: [............]  [............]   ← free               │    │
│  │  Block 3: [████████████]  [████████████]   ← shared (prefix)    │    │
│  │  ...                                                            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Forward Pass — Inside the GPU (models/qwen3.py + layers/)

```
input_ids [N]          positions [N]
    │                      │
    ▼                      │
┌──────────────────────┐   │
│ VocabParallelEmbed   │   │    embed_head.py
│ token ID → [N, 1024] │   │    (lookup table, TP via all_reduce)
└──────────┬───────────┘   │
           │               │
           ▼               │
    ╔══════════════════════════════════════════════════════════╗
    ║           DECODER LAYER  ×28                             ║
    ║                                                          ║
    ║  hidden_states ──▶ RMSNorm (+ residual add)              ║  layernorm.py
    ║       │                                                  ║
    ║       ▼                                                  ║
    ║  ┌─────────────────────────────────────────────────┐     ║
    ║  │            ATTENTION BLOCK                      │     ║
    ║  │                                                 │     ║
    ║  │  [N,1024] ──QKVParallelLinear──▶ [N,2048]       │     ║  linear.py
    ║  │              (ColumnParallel)                   │    ║
    ║  │                    │                            │    ║
    ║  │              split Q, K, V                      │    ║
    ║  │         Q[N,16,64] K[N,8,64] V[N,8,64]          │    ║
    ║  │              │         │                        │    ║
    ║  │         Q/K Norm (RMSNorm on head dim)          │    ║  layernorm.py
    ║  │              │         │                        │    ║
    ║  │         RoPE(positions, Q, K)                   │    ║  rotary_embedding.py
    ║  │              │         │       │                │    ║
    ║  │              ▼         ▼       ▼                │    ║
    ║  │  ┌─────────────────────────────────────┐        │    ║
    ║  │  │  store_kvcache (Triton kernel)      │        │    ║  attention.py
    ║  │  │  K,V → paged cache via slot_mapping │        │    ║
    ║  │  ├─────────────────────────────────────┤        │    ║
    ║  │  │  Flash Attention                    │        │    ║
    ║  │  │  prefill: flash_attn_varlen_func    │        │    ║
    ║  │  │  decode:  flash_attn_with_kvcache   │        │    ║
    ║  │  └────────────────┬────────────────────┘        │    ║
    ║  │                   │ [N,16,64]                   │    ║
    ║  │              flatten → [N,1024]                 │    ║
    ║  │                   │                             │    ║
    ║  │  [N,1024] ◀──o_proj (RowParallel + all_reduce) │    ║  linear.py
    ║  └───────────────────┬─────────────────────────────┘    ║
    ║                      │                                  ║
    ║  hidden_states ──▶ RMSNorm (+ residual add)            ║  layernorm.py
    ║       │                                                 ║
    ║       ▼                                                 ║
    ║  ┌─────────────────────────────────────────────────┐    ║
    ║  │              MLP BLOCK                          │    ║
    ║  │                                                 │    ║
    ║  │  [N,1024] ──gate_up_proj──▶ [N,5632]           │    ║  linear.py
    ║  │              (MergedColumnParallel)             │    ║
    ║  │                    │                            │    ║
    ║  │              SiluAndMul                         │    ║  activation.py
    ║  │         split gate,up → SiLU(gate)*up           │    ║
    ║  │                    │ [N,2816]                   │    ║
    ║  │                    │                            │    ║
    ║  │  [N,1024] ◀──down_proj (RowParallel+all_reduce)│    ║  linear.py
    ║  └───────────────────┬─────────────────────────────┘    ║
    ║                      │                                  ║
    ║              hidden_states, residual                    ║
    ╚══════════════════════╪══════════════════════════════════╝
                           │
                           ▼
                    Final RMSNorm                               layernorm.py
                           │
                           ▼
                  ┌─────────────────┐
                  │ ParallelLMHead  │                           embed_head.py
                  │ [1,1024]→       │  (prefill: only last token per seq)
                  │   [1,151936]    │  (decode: all tokens, one per seq)
                  └────────┬────────┘
                           │ logits
                           ▼
                  ┌─────────────────┐
                  │    Sampler      │                           sampler.py
                  │ temp → softmax  │
                  │ → Gumbel-max    │
                  │ → argmax        │
                  └────────┬────────┘
                           │
                           ▼
                      token_id [batch]
```

---

## Tensor Parallelism — Multi-GPU Communication (linear.py)

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  Per-layer communication: only 2 all_reduce calls                │
 │                                                                  │
 │  ATTENTION:                                                      │
 │  ┌──────────────────────┐          ┌───────────────────────┐     │
 │  │ qkv_proj             │          │ o_proj                │     │
 │  │ (ColumnParallel)     │          │ (RowParallel)         │     │
 │  │                      │          │                       │     │
 │  │ GPU 0: heads 0-7     │  ──────▶ │ GPU 0: partial sum    │    │
 │  │ GPU 1: heads 8-15    │  no comm │ GPU 1: partial sum    │     │
 │  │                      │          │         │             │     │
 │  └──────────────────────┘          │    all_reduce(sum)  ──┼──▶ synced
 │                                    └───────────────────────┘     │
 │  MLP:                                                            │
 │  ┌──────────────────────┐          ┌───────────────────────┐     │
 │  │ gate_up_proj         │          │ down_proj             │     │
 │  │ (MergedColumn)       │          │ (RowParallel)         │     │
 │  │                      │          │                       │     │
 │  │ GPU 0: half features │  ──────▶ │ GPU 0: partial sum    │    │
 │  │ GPU 1: half features │  no comm │ GPU 1: partial sum    │     │
 │  │                      │          │         │             │     │
 │  └──────────────────────┘          │    all_reduce(sum)  ──┼──▶ synced
 │                                    └───────────────────────┘     │
 └──────────────────────────────────────────────────────────────────┘
```

---

## Weight Loading — Checkpoint to Fused Layers (utils/loader.py)

```
HuggingFace Checkpoint (.safetensors)         nano-vllm Model
─────────────────────────────────────         ──────────────────

layers.0.self_attn.q_proj  ──┐
layers.0.self_attn.k_proj  ──┼──▶  qkv_proj (QKVParallelLinear)
layers.0.self_attn.v_proj  ──┘     [Q Q Q Q Q Q Q Q|K K K K|V V V V]
                                    TP shard: each GPU takes its heads

layers.0.mlp.gate_proj  ──┐
layers.0.mlp.up_proj    ──┴──▶  gate_up_proj (MergedColumnParallel)
                                [GATE GATE GATE|UP UP UP]
                                TP shard: each GPU takes half

layers.0.self_attn.o_proj  ──▶  o_proj (RowParallel, split input cols)
layers.0.mlp.down_proj    ──▶  down_proj (RowParallel, split input cols)

model.embed_tokens  ──▶  VocabParallelEmbedding (split vocab rows)
lm_head             ──▶  ParallelLMHead (may share embed weights)

Mapping defined by: Qwen3ForCausalLM.packed_modules_mapping
Loading done by:    utils/loader.py → calls param.weight_loader(param, data, shard_id)
```

---

## Prefill vs Decode — Two Modes of Operation

```
┌─────────────────────────────────────┬────────────────────────────────────┐
│           PREFILL                   │           DECODE                   │
│  "Process the whole prompt"         │  "Generate one token at a time"    │
├─────────────────────────────────────┼────────────────────────────────────┤
│                                     │                                    │
│  Input: all prompt tokens           │  Input: just the last token        │
│  [token₀, token₁, ..., tokenₙ]       │  [tokenₙ₊₁]                        │
│                                     │                                    │
│  Attention: flash_attn_varlen_func  │  Attention: flash_attn_with_kvcache│
│  Q,K,V all from current input       │  Q from new token only             │
│                                     │  K,V read from cache               │
│                                     │                                    │
│  KV Cache: WRITE all K,V to cache   │  KV Cache: WRITE 1 new K,V         │
│                                     │            READ all historical K,V │
│                                     │                                    │
│  Compute: O(n²) — all tokens        │  Compute: O(n) — 1 query vs cache  │
│           attend to each other      │                                    │
│                                     │                                    │
│  LM Head: only last token's logits  │  LM Head: the one token's logits   │
│  (cu_seqlens_q[1:] - 1)             │                                    │
│                                     │                                    │
│  Happens: ONCE per request          │  Happens: MANY times (one per      │
│                                     │           generated token)         │
│                                     │                                    │
│  Optimization: variable-length      │  Optimization: CUDA graph replay   │
│  batching packs multiple prompts    │  (pre-recorded at batch sizes      │
│                                     │   1, 2, 4, 8, 16, 32, ...)         │
└─────────────────────────────────────┴────────────────────────────────────┘
```

---

## Paged KV Cache & Block Manager — Memory Management

```
 Physical KV Cache (one giant GPU tensor, pre-allocated):

 Block 0  ┌────────────────────────────┐
          │ 256 token slots            │  ← used by Seq A (prompt prefix)
          │ K: [256, 8 heads, 64 dim]  │     ref_count=2 (shared via prefix cache!)
          │ V: [256, 8 heads, 64 dim]  │
 Block 1  ├────────────────────────────┤
          │ 256 token slots            │  ← used by Seq A (continued)
          │                            │     ref_count=1
 Block 2  ├────────────────────────────┤
          │ 256 token slots            │  ← used by Seq B
          │                            │     ref_count=1
 Block 3  ├────────────────────────────┤
          │ [empty]                    │  ← free
          │                            │
 Block 4  ├────────────────────────────┤
          │ 256 token slots            │  ← used by Seq B (shares Block 0!)
          │                            │     this IS Block 0 (same prefix)
          └────────────────────────────┘

 Seq A block_table: [0, 1]        ← "my pages are blocks 0 and 1"
 Seq B block_table: [0, 2]        ← "my pages are blocks 0 and 2"
                     ▲                  (Block 0 shared — prefix cache hit!)

 slot_mapping: token index → block_id × block_size + offset_within_block
               token 300 of Seq A → block 1, slot 44  → physical slot 256+44=300
```

---

## File Map — What Lives Where

```
nano-vllm/
├── example.py                  Entry point: demo usage
├── bench.py                    Entry point: throughput benchmark
│
├── nanovllm/
│   ├── __init__.py             Public API: exports LLM, SamplingParams
│   ├── llm.py                  LLM class (thin wrapper over LLMEngine)
│   ├── config.py               Config dataclass (model path, batch limits, TP size)
│   ├── sampling_params.py      SamplingParams (temperature, max_tokens)
│   │
│   ├── engine/                 ─── SCHEDULING & ORCHESTRATION ───
│   │   ├── llm_engine.py       Main loop: tokenize → schedule → run → postprocess
│   │   ├── scheduler.py        Decides what to run: prefill vs decode, preemption
│   │   ├── block_manager.py    KV cache allocation: paging, prefix cache, ref counting
│   │   ├── sequence.py         One request's state: tokens, block_table, status
│   │   └── model_runner.py     GPU execution: prepare tensors, forward pass, CUDA graphs
│   │
│   ├── models/                 ─── MODEL DEFINITION ───
│   │   └── qwen3.py            Qwen3: Attention + MLP + DecoderLayer + CausalLM
│   │                           (new models go here: llama.py, deepseek.py, etc.)
│   │
│   ├── layers/                 ─── REUSABLE NEURAL NETWORK BLOCKS ───
│   │   ├── attention.py        Paged KV cache (Triton) + Flash Attention
│   │   ├── linear.py           TP linear layers: Column, Row, Merged, QKV
│   │   ├── layernorm.py        RMSNorm (with fused residual add)
│   │   ├── rotary_embedding.py RoPE: position → rotation of Q,K
│   │   ├── activation.py       SiLU gated activation
│   │   ├── embed_head.py       Embedding (entry) + LM Head (exit)
│   │   ├── sampler.py          Logits → token ID (Gumbel-max)
│   │   └── README.md           Layer-level reference notes
│   │
│   └── utils/                  ─── SHARED UTILITIES ───
│       ├── context.py          Global metadata (prefill/decode, block_tables, slots)
│       └── loader.py           Load HF checkpoint → fused/sharded layers
```

---

## One Request's Complete Journey

```
 "Hello, how are you?"
         │
    ① TOKENIZE
         │  [15339, 11, 1268, 527, 499, 30]
         ▼
    ② ENQUEUE → Sequence(status=WAITING)
         │
    ③ SCHEDULE
         │  Scheduler: "prefill this sequence"
         │  BlockManager: allocate blocks [0, 1]
         │  Sequence: status=RUNNING, block_table=[0,1]
         ▼
    ④ PREPARE PREFILL
         │  input_ids = [15339, 11, 1268, 527, 499, 30]
         │  positions = [0, 1, 2, 3, 4, 5]
         │  slot_mapping = [0, 1, 2, 3, 4, 5]
         │  set_context(is_prefill=True, ...)
         ▼
    ⑤ GPU FORWARD PASS
         │  Embed → 28 layers → Norm → LM Head
         │  → logits [1, 151936]  (only last token)
         ▼
    ⑥ SAMPLE → token_id = 40  ("I")
         │
    ⑦ POSTPROCESS
         │  hash blocks for prefix cache
         │  seq.append_token(40)
         │  not EOS → continue
         ▼
    ⑧ SCHEDULE (decode this time)
         │  input_ids = [40]  (just the new token)
         │  positions = [6]
         ▼
    ⑨ GPU FORWARD (CUDA graph replay)
         │  Q from token "I"
         │  K,V read from cache (positions 0-5)
         │  → logits → sample → token_id = 2846 ("'m")
         ▼
    ⑩ REPEAT ⑧-⑨ until EOS or max_tokens
         │
         │  tokens: [40, 2846, 1695, 11, 9901, 499, 0]
         │           I   'm   fine  ,   thank  you  <eos>
         ▼
    ⑪ FINISH
         │  status = FINISHED
         │  deallocate KV cache blocks [0, 1]
         │  decode token_ids → "I'm fine, thank you"
         ▼
    ⑫ RETURN to user
```
