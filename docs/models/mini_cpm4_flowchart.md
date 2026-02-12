# MiniCPM4 模型流程图

## MiniCPM4 整体架构

### 顶层结构

```mermaid
flowchart TB
    A["MiniCPMForCausalLM"] --> B["模型主体"]
    A --> C["输出头"]
    B --> D["Cpm4Model"]
    C --> E["ParallelLMHead"]
```

### 模型主体层次

```mermaid
flowchart TB
    A["Cpm4Model"] --> B["词嵌入层"]
    A --> C["嵌入缩放"]
    A --> D["Transformer层堆叠"]
    A --> E["最终归一化"]
    
    B --> F["VocabParallelEmbedding"]
    C --> G["embed_scale"]
    D --> H["Cpm4DecoderLayer x N"]
    E --> I["RMSNorm"]
```

### Decoder Layer 内部

```mermaid
flowchart TB
    subgraph Decoder["Cpm4DecoderLayer"]
        direction TB
        A1["输入归一化"] --> B1["注意力模块"]
        B1 --> C1["后注意力归一化"]
        C1 --> D1["MLP模块"]
    end
    
    A1 --> E["RMSNorm"]
    B1 --> F["Cpm4Attention"]
    C1 --> G["RMSNorm"]
    D1 --> H["Cpm4MLP"]
    
    style Decoder fill:#f9f,stroke:#333,stroke-width:2px
```

### MiniCPM4 特有组件

```mermaid
flowchart TB
    subgraph Unique["MiniCPM4特有"]
        A["LongRoPE"]
        B["深度缩放"]
        C["宽度缩放"]
        D["嵌入缩放"]
    end
```

## LongRoPE Implementation

```mermaid
flowchart TD
    A["MiniCPMLongRoPE.__init__"] --> B["head_size, rotary_dim, max_position"]
    B --> C["base inverse frequencies"]
    C --> D["short_factor/long_factor"]
    D --> E["calculate scaling_factor"]
    E --> F["_set_cos_sin_cache(max_position)"]
    F --> G["t = arange(seq_len)"]
    G --> H{"seq_len > original_max_position?"}
    H -->|"Yes"| I["use long_factor"]
    H -->|"No"| J["use short_factor"]
    I --> K["freqs = outer(t, 1/ext_factors) * inv_freq"]
    J --> K
    K --> L["emb = cat(freqs, freqs)"]
    L --> M["cos_cached = emb.cos() * scaling_factor"]
    M --> N["sin_cached = emb.sin() * scaling_factor"]
```

## MiniCPMLongRoPE Forward

```mermaid
flowchart TD
    A["forward(positions, query, key)"] --> B["num_tokens = positions.size(0)"]
    B --> C["max_pos = positions.max()"]
    C --> D{"max_pos >= max_seq_len_cached?"}
    D -->|"Yes"| E["_set_cos_sin_cache(max_pos + 1)"]
    D -->|"No"| F["use cached cos/sin"]
    E --> G["cos = cos_cached[positions]"]
    F --> G
    G --> H["sin = sin_cached[positions]"]
    H --> I["query = _apply_rotary_emb(query, cos, sin)"]
    I --> J["key = _apply_rotary_emb(key, cos, sin)"]
    J --> K["return query, key"]
```

## Apply Rotary Embedding

```mermaid
flowchart TD
    A["_apply_rotary_emb(x, cos, sin)"] --> B["x: [num_tokens, num_heads, head_dim]"]
    B --> C["cos/sin: [num_tokens, head_dim]"]
    C --> D["unsqueeze cos/sin for broadcasting"]
    D --> E["convert to float32"]
    E --> F["chunk x into x1, x2"]
    F --> G["rotate_half = cat(-x2, x1)"]
    G --> H["result = x * cos + rotate_half * sin"]
    H --> I["return result.to(orig_dtype)"]
```

## Cpm4Attention Forward

```mermaid
flowchart TD
    A["forward(positions, hidden_states)"] --> B["qkv_proj(hidden_states)"]
    B --> C["split into q, k, v"]
    C --> D{"apply_qk_norm?"}
    D -->|"Yes"| E["q_norm(q), k_norm(k)"]
    D -->|"No"| F["skip norm"]
    E --> G["rotary_emb(positions, q, k)"]
    F --> G
    G --> H["attn(q, k, v)"]
    H --> I["o_proj(output)"]
    I --> J["return output"]
```

## Cpm4DecoderLayer Forward

```mermaid
flowchart TD
    A["forward(positions, hidden_states, residual)"] --> B["residual = hidden_states"]
    B --> C["hidden_states = input_layernorm(hidden_states)"]
    C --> D["attn_out = self_attn(positions, hidden_states)"]
    D --> E["scale = scale_depth / sqrt(num_hidden_layers)"]
    E --> F["hidden_states = residual + attn_out * scale"]
    F --> G["residual = hidden_states"]
    G --> H["hidden_states = post_attention_layernorm(hidden_states)"]
    H --> I["mlp_out = mlp(hidden_states)"]
    I --> J["hidden_states = residual + mlp_out * scale"]
    J --> K["return hidden_states, residual"]
```

## Cpm4Model Forward

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["embed_tokens(input_ids)"]
    B --> C["hidden_states = embeddings * embed_scale"]
    C --> D["residual = None"]
    D --> E["for layer in layers"]
    E --> F["hidden_states, residual = layer(positions, hidden_states, residual)"]
    F --> E
    E -->|"done"| G["hidden_states = norm(hidden_states)"]
    G --> H["return hidden_states"]
```

## MiniCPMForCausalLM Forward

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["model(input_ids, positions)"]
    B --> C["return hidden_states"]

    D["compute_logits(hidden_states)"] --> E["scale_width = hidden_size / dim_model_base"]
    E --> F["logits = lm_head(hidden_states / scale_width)"]
    F --> G["return logits"]
```

## MiniCPM vs Standard Transformer

```mermaid
flowchart TD
    subgraph Standard["Standard Transformer"]
        A1["PreNorm"] --> B1["Attention"]
        B1 --> C1["Add"]
        C1 --> D1["PreNorm"]
        D1 --> E1["MLP"]
        E1 --> F1["Add"]
    end

    subgraph MiniCPM["MiniCPM4"]
        A2["PreNorm"] --> B2["Attention"]
        B2 --> C2["Add with depth scaling"]
        C2 --> D2["PreNorm"]
        D2 --> E2["MLP"]
        E2 --> F2["Add with depth scaling"]
        G2["embedding scaling"] --> A2
        H2["width scaling before logits"]
    end
```

## Scaling Factors

```mermaid
flowchart TD
    subgraph DepthScaling["Depth Scaling"]
        A["scale_depth"] --> B["default = 1.0"]
        B --> C["per-layer scale = scale_depth / sqrt(num_hidden_layers)"]
        C --> D["applied to both attn and mlp outputs"]
    end

    subgraph WidthScaling["Width Scaling"]
        E["scale_width"] --> F["hidden_size / dim_model_base"]
        F --> G["applied before lm_head"]
        G --> H["hidden_states = hidden_states / scale_width"]
    end

    subgraph EmbedScaling["Embedding Scaling"]
        I["scale_emb"] --> J["default = 1.0"]
        J --> K["embed_tokens * scale_emb"]
    end
```

## get_cpm4_rope

```mermaid
flowchart TD
    A["get_cpm4_rope(head_size, rotary_dim, max_position, base, rope_scaling)"] --> B["extract short_factor"]
    B --> C["extract long_factor"]
    C --> D["extract original_max_position_embeddings"]
    D --> E["create MiniCPMLongRoPE"]
    E --> F["return rotary_emb"]
```

## Key Features

```mermaid
flowchart LR
    A["MiniCPM4 Features"] --> B["LongRoPE for extended context"]
    A --> C["Non-uniform scaling factors"]
    A --> D["Depth-wise scaling"]
    A --> E["Width-wise scaling"]
    A --> F["Embedding scaling"]
    A --> G["Optional Q/K normalization"]
```
