# Llama 模型流程图

## Llama 整体架构

### 顶层结构

```mermaid
flowchart TB
    A["LlamaForCausalLM"] --> B["模型主体"]
    A --> C["输出头"]
    B --> D["LlamaModel"]
    C --> E["ParallelLMHead"]
```

### 模型主体层次

```mermaid
flowchart TB
    A["LlamaModel"] --> B["词嵌入层"]
    A --> C["Transformer层堆叠"]
    A --> D["最终归一化"]
    
    B --> E["VocabParallelEmbedding"]
    C --> F["LlamaDecoderLayer x N"]
    D --> G["RMSNorm"]
```

### Decoder Layer 内部

```mermaid
flowchart TB
    subgraph Decoder["LlamaDecoderLayer"]
        direction TB
        A1["输入归一化"] --> B1["注意力模块"]
        B1 --> C1["后注意力归一化"]
        C1 --> D1["MLP模块"]
    end
    
    A1 --> E["RMSNorm"]
    B1 --> F["LlamaAttention"]
    C1 --> G["RMSNorm"]
    D1 --> H["LlamaMLP"]
```

## LlamaAttention 前向传播

```mermaid
flowchart TD
    A["forward(positions, hidden_states)"] --> B["qkv_proj(hidden_states)"]
    B --> C["split into q, k, v"]
    C --> D["view reshape for attention"]
    D --> E["rotary_emb(positions, q, k)"]
    E --> F["attn(q, k, v)"]
    F --> G["o_proj(output)"]
    G --> H["return output"]
```

## LlamaMLP 前向传播

```mermaid
flowchart TD
    A["forward(x)"] --> B["gate_up_proj(x)"]
    B --> C["act_fn: SiluAndMul"]
    C --> D["down_proj(x)"]
    D --> E["return output"]
```

## LlamaDecoderLayer 前向传播

```mermaid
flowchart TD
    A["forward(positions, hidden_states, residual)"] --> B{"residual is None?"}
    B -->|"Yes"| C["residual = hidden_states"]
    C --> D["hidden_states = input_layernorm(hidden_states)"]
    B -->|"No"| E["input_layernorm(hidden_states, residual)"]
    E --> F["return hidden_states, residual"]
    D --> G["self_attn(positions, hidden_states)"]
    F --> G
    G --> H["post_attention_layernorm(hidden_states, residual)"]
    H --> I["mlp(hidden_states)"]
    I --> J["return hidden_states, residual"]
```

## LlamaModel 前向传播

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["embed_tokens(input_ids)"]
    B --> C["hidden_states = embeddings"]
    C --> D["residual = None"]
    D --> E["for each layer in layers"]
    E --> F["layer(positions, hidden_states, residual)"]
    F --> G["update hidden_states, residual"]
    G --> E
    E -->|"done"| H["norm(hidden_states, residual)"]
    H --> I["return hidden_states"]
```

## LlamaForCausalLM 前向传播

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["model(input_ids, positions)"]
    B --> C["return hidden_states"]

    D["compute_logits(hidden_states)"] --> E["lm_head(hidden_states)"]
    E --> F["return logits"]
```

## 张量并行切分

```mermaid
flowchart TD
    subgraph TP["Tensor Parallel"]
        A["total_num_heads"] -->|"/ tp_size"| B["num_heads per device"]
        C["total_num_kv_heads"] -->|"/ tp_size"| D["num_kv_heads per device"]
    end

    subgraph QKVProj["QKVParallelLinear"]
        E["hidden_size"] --> F["q_size, kv_size"]
        F --> G["concat(q, k, v)"]
    end

    subgraph Attention["Attention Pattern"]
        H["Q"] --> I["Scaled Dot-Product"]
        J["K"] --> I
        K["V"] --> L["Weighted Sum"]
        I --> M["Softmax"]
        M --> L
    end
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
