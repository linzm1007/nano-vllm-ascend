# Qwen3 模型流程图

## Qwen3 整体架构

### 顶层结构

```mermaid
flowchart TB
    A["Qwen3ForCausalLM"] --> B["模型主体"]
    A --> C["输出头"]
    B --> D["Qwen3Model"]
    C --> E["ParallelLMHead"]
```

### 模型主体层次

```mermaid
flowchart TB
    A["Qwen3Model"] --> B["词嵌入层"]
    A --> C["Transformer层堆叠"]
    A --> D["最终归一化"]
    
    B --> E["VocabParallelEmbedding"]
    C --> F["Qwen3DecoderLayer x N"]
    D --> G["RMSNorm"]
```

### Decoder Layer 内部

```mermaid
flowchart TB
    subgraph Decoder["Qwen3DecoderLayer"]
        direction TB
        A1["输入归一化"] --> B1["注意力模块"]
        B1 --> C1["后注意力归一化"]
        C1 --> D1["MLP模块"]
    end
    
    A1 --> E["RMSNorm"]
    B1 --> F["Qwen3Attention"]
    C1 --> G["RMSNorm"]
    D1 --> H["Qwen3MLP"]
```

## Qwen3Attention 前向传播

```mermaid
flowchart TD
    A["forward(positions, hidden_states)"] --> B["qkv_proj(hidden_states)"]
    B --> C["split into q, k, v"]
    C --> D["view reshape"]
    D --> E{"qkv_bias?"}
    E -->|"No"| F["q_norm(q), k_norm(k)"]
    E -->|"Yes"| G["skip norm"]
    F --> H["rotary_emb(positions, q, k)"]
    G --> H
    H --> I["attn(q, k, v)"]
    I --> J["flatten + o_proj"]
    J --> K["return output"]
```

## Qwen3MLP 前向传播

```mermaid
flowchart TD
    A["forward(x)"] --> B["gate_up_proj(x)"]
    B --> C["SiluAndMul activation"]
    C --> D["down_proj(x)"]
    D --> E["return output"]
```

## Qwen3DecoderLayer 前向传播

```mermaid
flowchart TD
    A["forward(positions, hidden_states, residual)"] --> B{"residual is None?"}
    B -->|"Yes"| C["input_layernorm(hidden_states), hidden_states"]
    B -->|"No"| D["input_layernorm(hidden_states, residual)"]
    C --> E["self_attn(positions, hidden_states)"]
    D --> E
    E --> F["post_attention_layernorm(hidden_states, residual)"]
    F --> G["mlp(hidden_states)"]
    G --> H["return hidden_states, residual"]
```

## Qwen3Model 前向传播

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["embed_tokens(input_ids)"]
    B --> C["hidden_states = embeddings"]
    C --> D["residual = None"]
    D --> E["for layer_idx, layer in enumerate(layers)"]
    E --> F["layer(positions, hidden_states, residual)"]
    F --> G["update hidden_states, residual"]
    G --> E
    E -->|"done"| H["norm(hidden_states, residual)"]
    H --> I["return hidden_states"]
```

## Qwen3ForCausalLM 前向传播

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["model(input_ids, positions)"]
    B --> C["return hidden_states"]

    D["compute_logits(hidden_states)"] --> E["lm_head(hidden_states)"]
    E --> F["return logits"]
```

## Packed Modules Mapping

```mermaid
flowchart LR
    subgraph Mapping["Weight Mapping"]
        q_proj["q_proj"] --> qkv_q["qkv_proj.q"]
        k_proj["k_proj"] --> qkv_k["qkv_proj.k"]
        v_proj["v_proj"] --> qkv_v["qkv_proj.v"]
        gate_proj["gate_proj"] --> gate_up_0["gate_up_proj[0]"]
        up_proj["up_proj"] --> gate_up_1["gate_up_proj[1]"]
    end
```

## Qwen3 vs Llama 关键区别

```mermaid
flowchart TD
    subgraph Llama["Llama"]
        L1["Standard RMSNorm"]
        L2["Optional bias"]
        L3["Standard RoPE"]
    end

    subgraph Qwen3["Qwen3"]
        Q1["Conditional Q/K Norm"]
        Q2["Configurable bias"]
        Q3["Enhanced RoPE"]
        Q4["Default rope_theta=1000000"]
    end
```

## 配置参数继承

```mermaid
flowchart TD
    A["Qwen3Config"] --> B["hidden_size"]
    A --> C["num_attention_heads"]
    A --> D["num_key_value_heads"]
    A --> E["intermediate_size"]
    A --> F["rms_norm_eps"]
    A --> G["attention_bias"]
    A --> H["head_dim"]
    A --> I["rope_theta"]
    A --> J["rope_scaling"]
    A --> K["max_position_embeddings"]
    A --> L["hidden_act"]
```
