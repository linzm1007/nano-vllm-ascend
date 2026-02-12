# ModelRunner 流程图

## ModelRunner 整体架构

```mermaid
flowchart TD
    subgraph ModelRunner["ModelRunner 模型运行器"]
        model["model: 加载的模型"]
        sampler["sampler: 采样器"]
        kv_cache["kv_cache: KV缓存"]
        shm["shm: 共享内存"]
        compiler["compile_decode: 编译后的模型"]
    end

    subgraph MultiProcess["多进程通信"]
        rank0["rank=0: 主进程"]
        rankN["rank>0: 工作进程"]
        events["events: 进程同步事件"]
    end
```

## 初始化流程 (__init__)

```mermaid
flowchart TD
    A["__init__(config, rank, event)"] --> B["设置配置参数"]
    B --> C["初始化分布式进程组 HCCL"]
    C --> D["设置 NPU 设备"]
    D --> E["设置默认数据类型"]
    E --> F["加载模型"]
    F --> G{"模型类型?"}
    G -->|"qwen3_vl"| H["_load_qwen3_vl_strategy"]
    G -->|"其他"| I["_load_default_strategy"]
    H --> J["is_multimodal = True"]
    I --> K["is_multimodal = False"]
    J --> L["初始化 sampler"]
    K --> L
    L --> M["分配 KV Cache"]
    M --> N{"enforce_eager?"}
    N -->|"否"| O["decode_compile()"]
    N -->|"是"| P["跳过编译"]
    O --> Q["设置共享内存"]
    P --> Q
```

## KV Cache 分配流程

```mermaid
flowchart TD
    A["allocate_kv_cache()"] --> B["获取 NPU 内存信息"]
    B --> C["计算 block_bytes"]
    C --> D["计算可用内存"]
    D --> E["计算 num_kvcache_blocks"]
    E --> F["创建 KV Cache tensor"]
    F --> G["shape: (2, num_layers, num_blocks, block_size, num_kv_heads*head_dim)"]
    G --> H["zero_() 初始化为0"]
    H --> I["遍历模型模块"]
    I --> J{"有 k_cache 和 v_cache?"}
    J -->|"是"| K["绑定 k_cache/v_cache"]
    J -->|"否"| L["继续"]
    K --> I
    L --> I
```

## 模型编译流程

```mermaid
flowchart TD
    A["decode_compile()"] --> B{"graph_mode?"}
    B -->|"MAX_AUTOTUNE"| C["max-autotune 模式"]
    B -->|"REDUCE_OVERHEAD"| D["reduce-overhead 模式"]
    C --> E{"use_graph_cache?"}
    E -->|"是"| F["torchair.inference.cache_compile"]
    E -->|"否"| G["torch.compile + npu_backend"]
    D --> H["设置 aclgraph 模式"]
    H --> I["torch.compile + npu_backend"]
```

## 运行流程 (run)

```mermaid
flowchart TD
    A["run(seqs, is_prefill)"] --> B{"is_prefill?"}
    B -->|"是"| C["prepare_prefill(seqs)"]
    B -->|"否"| D{"enforce_eager?"}
    D -->|"是"| E["prepare_decode(seqs)"]
    D -->|"否"| F["prepare_decode_padding(seqs)"]
    C --> G["计算 sequence_lengths"]
    E --> H["获取 vision_slices_per_seq"]
    F --> H
    G --> I["_get_vision_slices_per_seq"]
    H --> J["prepare_sample(seqs)"]
    I --> J
    J --> K["run_model()"]
    K --> L["_advance_vision_offsets()"]
    L --> M["sampler(logits, temperatures)"]
    M --> N["reset_context()"]
    N --> O["返回 token_ids"]
```

## Prefill 准备流程

```mermaid
flowchart TD
    A["prepare_prefill(seqs)"] --> B["初始化列表"]
    B --> C["遍历 seqs"]
    C --> D["input_ids.extend(seq.token_ids)"]
    D --> E["positions.extend(range(0, len(seq)))"]
    E --> F["计算 cu_seqlens_q/k"]
    F --> G["更新 max_seqlen_q/k"]
    G --> H{"有 block_table?"}
    H -->|"是"| I["计算 slot_mapping"]
    H -->|"否"| J["继续"]
    I --> C
    J --> C
    C -->|"遍历完成"| K["prepare_block_tables"]
    K --> L["转换为 tensor 并移至设备"]
    L --> M["set_context(True, ...)"]
    M --> N["返回 (input_ids, positions)"]
```

## Decode 准备流程

```mermaid
flowchart TD
    A["prepare_decode(seqs)"] --> B["遍历 seqs"]
    B --> C["input_ids.append(last_token)"]
    C --> D["positions.append(len(seq)-1)"]
    D --> E["context_lens.append(len(seq))"]
    E --> F["slot_mapping.append([last_block, offset])"]
    F --> B
    B -->|"完成"| G["转换为 tensor"]
    G --> H["准备 block_tables"]
    H --> I["set_context(False, ...)"]
    I --> J["返回 (input_ids, positions)"]
```

## Decode Padding 准备流程

```mermaid
flowchart TD
    A["prepare_decode_padding(seqs)"] --> B["获取 real_bs"]
    B --> C["遍历 seqs 收集数据"]
    C --> D["padding_size = max_compile_bs - real_bs"]
    D --> E{"padding_size > 0?"}
    E -->|"是"| F["填充 input_ids 为 0"]
    F --> G["填充 positions 为 0"]
    G --> H["填充 context_lens 为 0"]
    H --> I["填充 slot_mapping 为 dummy_slot"]
    E -->|"否"| J["继续"]
    I --> K["构造静态 block_tables"]
    J --> K
    K --> L["set_context(False, real_bs=real_bs)"]
```

## 模型执行流程 (run_model)

```mermaid
flowchart TD
    A["run_model(input_ids, positions, is_prefill, ...)"] --> B{"is_multimodal<br/>且 is_prefill?"}
    B -->|"是"| C["设置 model_kwargs"]
    C --> D["获取 execute_tokens"]
    B -->|"否"| D
    D --> E{"条件判断"}
    E -->|"is_prefill<br/>或 enforce_eager<br/>或 tokens > 512"| F["model.compute_logits<br/>(model(input_ids, positions))"]
    E -->|"其他"| G["model.compute_logits<br/>(compile_decode(input_ids, positions))"]
```

## 视觉特征处理流程

```mermaid
flowchart TD
    A["_ensure_vision_cache(seq)"] --> B{"已有缓存?"}
    B -->|"是"| C["返回"]
    B -->|"否"| D["检查 pixel_values 和 grid"]
    D --> E["移至 GPU"]
    E --> F["model.visual(pixel, grid)"]
    F --> G["image_embeds, deepstack_features"]
    G --> H["缓存到 CPU"]
    H --> I["清空原始数据"]

    J["_get_vision_slices_per_seq"] --> K{"is_prefill 且 is_multimodal?"}
    K -->|"否"| L["返回 None"]
    K -->|"是"| M["遍历 seqs"]
    M --> N["_ensure_vision_cache"]
    N --> O["计算 window_start/end"]
    O --> P["遍历 vision_placeholders"]
    P --> Q{"计算重叠区域"}
    Q -->|"有重叠"| R["计算 slice"]
    R --> S["_get_token_slice"]
    R --> T["_get_deepstack_slice"]
    S --> U["添加到 slices_for_seq"]
    T --> U
    U --> P
```

## 多进程通信流程

```mermaid
flowchart TD
    A["call(method_name, *args)"] --> B{"world_size > 1 且 rank == 0?"}
    B -->|"是"| C["write_shm(method_name, *args)"]
    B -->|"否"| D["直接调用方法"]
    C --> D
```

### 写共享内存 (write_shm)

```mermaid
flowchart TD
    A["write_shm"] --> B["pickle.dumps 序列化"]
    B --> C["写入共享内存"]
    C --> D["设置所有 event"]
    D --> E["通知工作进程"]
```

### 读共享内存 (read_shm)

```mermaid
flowchart TD
    A["read_shm"] --> B["event.wait() 等待信号"]
    B --> C["读取数据长度"]
    C --> D["pickle.loads 反序列化"]
    D --> E["event.clear()"]
    E --> F["返回 method_name, args"]
```

### 多进程架构

```mermaid
flowchart TB
    subgraph Rank0["Rank 0 (主进程)"]
        A1["调用 call()"]
        A2["write_shm()"]
        A3["等待执行结果"]
    end
    
    subgraph RankN["Rank N (工作进程)"]
        B1["loop() 监听"]
        B2["read_shm()"]
        B3["执行方法"]
    end
    
    subgraph SharedMem["共享内存"]
        C1["共享内存缓冲区"]
        C2["同步事件"]
    end
    
    A1 --> A2
    A2 --> C1
    C1 --> B2
    C2 --> B2
    B1 --> B2
    B2 --> B3
```
