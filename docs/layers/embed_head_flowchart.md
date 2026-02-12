# Embed Head 流程图

## 整体架构

```mermaid
flowchart TD
    A["nn.Module"] --> B["VocabParallelEmbedding"]
    B --> C["ParallelLMHead"]
```

## VocabParallelEmbedding

```mermaid
flowchart TD
    A["VocabParallelEmbedding.__init__"] --> B["num_embeddings, embedding_dim"]
    B --> C["获取 tp_rank, tp_size"]
    C --> D["assert num_embeddings % tp_size == 0"]
    D --> E["num_embeddings_per_partition = num_embeddings // tp_size"]
    E --> F["计算 vocab_start_idx, vocab_end_idx"]
    F --> G["创建 weight Parameter"]
    G --> H["设置 weight_loader"]
```

## VocabParallelEmbedding 前向传播

```mermaid
flowchart TD
    A["forward(x)"] --> B{"tp_size > 1?"}
    B -->|Yes| C["创建 mask: (x >= vocab_start) & (x < vocab_end)"]
    C --> D["x = mask * (x - vocab_start_idx)"]
    B -->|No| E["直接使用 x"]
    D --> F["y = F.embedding(x, weight)"]
    E --> F
    F --> G{"tp_size > 1?"}
    G -->|Yes| H["y = mask.unsqueeze(1) * y"]
    H --> I["dist.all_reduce(y)"]
    G -->|No| J["return y"]
    I --> J
```

## VocabParallelEmbedding 权重加载

```mermaid
flowchart TD
    A["weight_loader(param, loaded_weight)"] --> B["param_data = param.data"]
    B --> C["shard_size = param_data.size(0)"]
    C --> D["start_idx = tp_rank * shard_size"]
    D --> E["loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)"]
    E --> F["param_data.copy_(loaded_weight)"]
```

## ParallelLMHead

```mermaid
flowchart TD
    A["ParallelLMHead.__init__"] --> B["num_embeddings, embedding_dim"]
    B --> C["assert not bias"]
    C --> D["调用父类 VocabParallelEmbedding"]
```

## ParallelLMHead 前向传播

```mermaid
flowchart TD
    A["forward(x)"] --> B["获取 context"]
    B --> C{"is_prefill?"}
    C -->|Yes| D["last_indices = cu_seqlens_q[1:] - 1"]
    D --> E["x = x[last_indices].contiguous()"]
    C -->|No| F["直接使用 x"]
    E --> G["logits = F.linear(x, weight)"]
    F --> G
    G --> H{"tp_size > 1?"}
    H -->|Yes| I{"tp_rank == 0?"}
    I -->|Yes| J["创建 all_logits 列表"]
    J --> K["dist.gather(logits, all_logits, 0)"]
    K --> L["logits = torch.cat(all_logits, -1)"]
    I -->|No| M["logits = None"]
    H -->|No| N["return logits"]
    L --> N
    M --> N
```

## 词嵌入并行策略

```mermaid
flowchart LR
    subgraph Embedding["VocabParallelEmbedding"]
        A1["按词汇表切分"]
        A2["每个rank负责部分vocab"]
        A3["需要all_reduce聚合"]
    end
    
    subgraph LMHead["ParallelLMHead"]
        B1["转置权重用于输出"]
        B2["Prefill时取last token"]
        B3["Gather到rank 0"]
    end
```

## 输入处理流程

```mermaid
flowchart TD
    subgraph Input["输入 token ids"]
        A["[batch_size, seq_len]"]
    end
    
    subgraph Masking["掩码处理"]
        B["检查是否在范围内"]
        C["范围外设为0"]
        D["范围内减去offset"]
    end
    
    subgraph Embedding["嵌入查找"]
        E["F.embedding"]
        F["得到 embeddings"]
    end
    
    subgraph AllReduce["AllReduce"]
        G["聚合所有rank结果"]
    end
    
    Input --> Masking
    Masking --> Embedding
    Embedding --> AllReduce
```

## 输出投影流程

```mermaid
flowchart TD
    subgraph Hidden["隐藏状态"]
        A["[batch_size, hidden_dim]"]
    end
    
    subgraph Slice["切片(Prefill)"]
        B["取最后位置"]
        C["[batch_size, hidden_dim]"]
    end
    
    subgraph Linear["线性变换"]
        D["F.linear(x, weight.T)"]
        E["[batch_size, vocab_size/tp_size]"]
    end
    
    subgraph Gather["Gather"]
        F["仅在rank 0收集"]
        G["[batch_size, vocab_size]"]
    end
    
    Hidden --> Slice
    Slice --> Linear
    Linear --> Gather
```
