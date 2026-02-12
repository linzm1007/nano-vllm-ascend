# Attention Torch Native 流程图

## 整体架构

### 类结构概览

```mermaid
flowchart TB
    subgraph Main["Attention 类"]
        A["核心方法"] 
        B["内部工具方法"]
        C["辅助函数"]
    end
```

### 核心方法

```mermaid
flowchart LR
    A["__init__"] --> B["forward"]
    B --> C["Prefill阶段"] 
    B --> D["Decode阶段"]
```

### 内部方法分类

```mermaid
flowchart TD
    subgraph Init["初始化"]
        A1["__init__"]
        A2["_init_cache"]
    end
    
    subgraph Compute["计算核心"]
        B1["_prefill_attn"]
        B2["_decode_attn"]
    end
    
    subgraph Tools["工具方法"]
        C1["store_kvcache"]
        C2["repeat_kv"]
        C3["_check_numeric_range"]
    end
    
    Init --> Compute
    Tools --> Compute
```

## 初始化流程

```mermaid
flowchart TD
    A["Attention.__init__"] --> B["设置 num_heads, num_kv_heads, head_dim"]
    B --> C["assert num_heads % num_kv_heads == 0"]
    C --> D["计算 num_rep = num_heads // num_kv_heads"]
    D --> E["计算 scale = 1/sqrt(head_dim)"]
    E --> F["初始化空 k_cache, v_cache"]
```

## KV缓存存储流程

```mermaid
flowchart TD
    A["store_kvcache"] --> B["输入: key[N,G,D], value[N,G,D]"]
    B --> C["验证维度匹配"]
    C --> D["筛选有效slot (slot != -1)"]
    D --> E["计算 block_idx, block_inner_idx"]
    E --> F["筛选在缓存范围内的元素"]
    F --> G["生成静态索引"]
    G --> H["提取有效数据"]
    H --> I["批量更新 k_cache, v_cache"]
```

## GQA扩展 (repeat_kv)

```mermaid
flowchart TD
    A["repeat_kv(hidden_states, n_rep)"] --> B{"n_rep == 1?"}
    B -->|Yes| C["return hidden_states"]
    B -->|No| D["seq_len, G, D = shape"]
    D --> E["assert G * n_rep <= 32"]
    E --> F["unsqueeze(2).expand(seq_len, G, n_rep, D)"]
    F --> G["reshape(seq_len, G*n_rep, D)"]
```

## Prefill注意力计算

```mermaid
flowchart TD
    A["_prefill_attn(q, k, v, cu_seqlens_q, cu_seqlens_k)"] --> B["batch_size = len(cu_seqlens_q) - 1"]
    B --> C["outputs = []"]
    C --> D["for i in range(batch_size)"]
    D --> E["切片: start_q/end_q, start_k/end_k"]
    E --> F["提取 q_seq, k_seq, v_seq"]
    F --> G["_check_numeric_range"]
    G --> H["repeat_kv 扩展 K,V"]
    H --> I["转置计算注意力"]
    I --> J["matmul(q_trans, k_trans) * scale"]
    J --> K["添加因果掩码"]
    K --> L["softmax"]
    L --> M["matmul(attn, v_trans)"]
    M --> N["转置恢复并添加到outputs"]
    N --> D
    D -->|done| O["return torch.cat(outputs)"]
```

## Decode注意力计算

```mermaid
flowchart TD
    A["_decode_attn(q, context_lens, block_tables)"] --> B["batch_size = q.shape[0]"]
    B --> C["reshape q: [B, H, D]"]
    C --> D["for i in range(batch_size)"]
    D --> E["验证缓存索引"]
    E --> F["读取历史KV缓存"]
    F --> G["concat并限制长度"]
    G --> H["repeat_kv 扩展"]
    H --> I["unsqueeze q_i"]
    I --> J["计算注意力分数"]
    J --> K["softmax"]
    K --> L["加权求和"]
    L --> M["squeeze并添加到outputs"]
    M --> D
    D -->|done| N["return torch.cat(outputs)"]
```

## 主前向传播流程

```mermaid
flowchart TD
    A["forward(q, k, v)"] --> B["获取 context"]
    B --> C{"k_cache为空?"}
    C -->|Yes| D["_init_cache"]
    C -->|No| E["跳过"]
    D --> F["store_kvcache"]
    E --> F
    F --> G["维度校验"]
    G --> H{"is_prefill?"}
    H -->|Yes| I["_prefill_attn"]
    H -->|No| J["_decode_attn"]
    I --> K["reshape输出"]
    J --> K
    K --> L["数值范围检查"]
    L --> M["return output"]
```

## Prefill vs Decode 对比

```mermaid
flowchart LR
    subgraph Prefill["Prefill阶段"]
        A1["处理完整序列"]
        A2["使用cu_seqlens"]
        A3["因果掩码"]
        A4["批量处理多个seq"]
    end
    
    subgraph Decode["Decode阶段"]
        B1["单token推理"]
        B2["使用block_tables"]
        B3["读取KV缓存"]
        B4["逐个seq处理"]
    end
```

## 数值范围检查

```mermaid
flowchart TD
    A["_check_numeric_range(x, name)"] --> B["max_val = x.max()"]
    B --> C["min_val = x.min()"]
    C --> D{"max_val > 1e4 or min_val < -1e4?"}
    D -->|Yes| E["logger.warning 数值溢出"]
    D -->|No| F["正常"]
    E --> G["return x"]
    F --> G
```

## 因果掩码生成

```mermaid
flowchart TD
    A["生成因果掩码"] --> B["Lq, Lk = q_seq.shape[0], k_seq.shape[0]"]
    B --> C["mask = triu(full((Lq, Lk), -inf), diagonal=1)"]
    C --> D["attn = attn + mask.unsqueeze(0)"]
    D --> E["结果: 上三角为-inf, 下三角为0"]
```
