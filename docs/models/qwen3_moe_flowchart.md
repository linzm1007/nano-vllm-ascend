# Qwen3-MoE 模型流程图

## Qwen3-MoE 整体架构

### 顶层结构

```mermaid
flowchart TB
    A["Qwen3MoeForCausalLM"] --> B["模型主体"]
    A --> C["输出头"]
    B --> D["Qwen3MoeModel"]
    C --> E["ParallelLMHead"]
```

### 模型主体层次

```mermaid
flowchart TB
    A["Qwen3MoeModel"] --> B["词嵌入层"]
    A --> C["Transformer层堆叠"]
    A --> D["最终归一化"]
    
    B --> E["VocabParallelEmbedding"]
    C --> F["Qwen3MoeDecoderLayer x N"]
    D --> G["RMSNorm"]
```

### Decoder Layer 内部

```mermaid
flowchart TB
    subgraph Decoder["Qwen3MoeDecoderLayer"]
        direction TB
        A1["输入归一化"] --> B1["注意力模块"]
        B1 --> C1["后注意力归一化"]
        C1 --> D1["MLP/MoE模块"]
    end
    
    A1 --> E["RMSNorm"]
    B1 --> F["Qwen3MoeAttention"]
    C1 --> G["RMSNorm"]
    D1 --> H["MLP 或 SparseMoE"]
```

## MoE Layer Selection

```mermaid
flowchart TD
    A["Qwen3MoeDecoderLayer.__init__"] --> B["create self_attn"]
    B --> C{"layer_idx in mlp_only_layers?"}
    C -->|"Yes"| D["use Qwen3MoeMLP"]
    C -->|"No"| E{"num_experts > 0 and (layer_idx+1) % decoder_sparse_step == 0?"}
    E -->|"Yes"| F["use Qwen3MoeSparseMoeBlock"]
    E -->|"No"| D
    F --> G["self.mlp = selected module"]
    D --> G
```

## Sparse MoE Block 前向传播

```mermaid
flowchart TD
    A["Qwen3MoeSparseMoeBlock.forward(hidden_states)"] --> B["sequence_length, hidden_dim = shape"]
    B --> C["router_logits = gate(hidden_states)"]
    C --> D["routing_weights = softmax(router_logits, dim=1)"]
    D --> E["routing_weights, selected_experts = topk(routing_weights, top_k)"]
    E --> F["routing_weights /= sum(routing_weights, keepdim=True)"]
    F --> G["final_hidden_states = zeros"]
    G --> H["expert_mask = one_hot(selected_experts).permute"]
    H --> I["expert_hitted = nonzero(expert_mask.sum)"]
    I --> J["for expert_idx in expert_hitted"]
    J --> K["expert_layer = experts[expert_idx]"]
    K --> L["idx, top_x = where(expert_mask[expert_idx])"]
    L --> M["current_state = hidden_states[top_x].reshape"]
    M --> N["current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]"]
    N --> O["final_hidden_states.index_add_(0, top_x, current_hidden_states)"]
    O --> J
    J -->|"done"| P["return final_hidden_states"]
```

## Qwen3MoeAttention 前向传播

```mermaid
flowchart TD
    A["forward(positions, hidden_states)"] --> B["qkv_proj(hidden_states)"]
    B --> C["split into q, k, v"]
    C --> D["view reshape for norm"]
    D --> E["q_norm(q_by_head), k_norm(k_by_head)"]
    E --> F["view back"]
    F --> G["rotary_emb(positions, q, k)"]
    G --> H["attn(q, k, v)"]
    H --> I["o_proj(output)"]
    I --> J["return output"]
```

## Standard MLP vs MoE MLP

```mermaid
flowchart TD
    subgraph Standard["Qwen3MoeMLP (Standard)"]
        A["forward(x)"] --> B["gate_up_proj(x)"]
        B --> C["SiluAndMul"]
        C --> D["down_proj"]
        D --> E["return output"]
    end

    subgraph MoE["Qwen3MoeSparseMoeBlock (Sparse)"]
        F["forward(hidden_states)"] --> G["gate(hidden_states)"]
        G --> H["softmax -> top_k"]
        H --> I["select top_k experts"]
        I --> J["route to selected experts"]
        J --> K["weighted aggregation"]
        K --> L["return output"]
    end
```

## Decoder Layer Forward

```mermaid
flowchart TD
    A["forward(positions, hidden_states, residual)"] --> B{"residual is None?"}
    B -->|"Yes"| C["residual = hidden_states"]
    C --> D["hidden_states = input_layernorm(hidden_states)"]
    B -->|"No"| E["input_layernorm(hidden_states, residual)"]
    E --> F["return h, r"]
    D --> G["self_attn(positions, hidden_states)"]
    F --> G
    G --> H["post_attention_layernorm(hidden_states, residual)"]
    H --> I["mlp(hidden_states)"]
    I --> J["return hidden_states, residual"]
```

## Model Forward

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["embed_tokens(input_ids)"]
    B --> C["hidden_states = embeddings"]
    C --> D["residual = None"]
    D --> E["for layer in layers"]
    E --> F["hidden_states, residual = layer(positions, hidden_states, residual)"]
    F --> E
    E -->|"done"| G["hidden_states, _ = norm(hidden_states, residual)"]
    G --> H["return hidden_states"]
```

## CausalLM Forward

```mermaid
flowchart TD
    A["forward(input_ids, positions)"] --> B["model(input_ids, positions)"]
    B --> C["return hidden_states"]

    D["compute_logits(hidden_states)"] --> E["lm_head(hidden_states)"]
    E --> F["return logits"]
```

## MoE Configuration

```mermaid
flowchart TD
    subgraph Config["Qwen3MoeConfig"]
        A["num_experts"] --> B["number of expert modules"]
        C["num_experts_per_tok"] --> D["top_k selection"]
        E["moe_intermediate_size"] --> F["expert hidden dim"]
        G["decoder_sparse_step"] --> H["MoE layer interval"]
        I["mlp_only_layers"] --> J["layers forced to use dense MLP"]
    end
```

## Expert Selection Process

```mermaid
flowchart LR
    A["Input<br/>[seq_len, hidden_dim]"] --> B["Router<br/>Linear(hidden_dim, num_experts)"]
    B --> C["Softmax"]
    C --> D["Top-K Selection"]
    D --> E["Expert 0"]
    D --> F["Expert 1"]
    D --> G["Expert N"]
    E --> H["Weighted Sum"]
    F --> H
    G --> H
    H --> I["Output<br/>[seq_len, hidden_dim]"]
```

## Mask-Based Expert Routing

```mermaid
flowchart TD
    A["expert_mask = one_hot(selected_experts, num_experts)"] --> B["permute to [num_experts, top_k, seq_len]"]
    B --> C["expert_hitted = nonzero(expert_mask.sum)"]
    C --> D["for expert_idx in expert_hitted"]
    D --> E["idx, top_x = where(expert_mask[expert_idx])"]
    E --> F["select tokens for this expert"]
    F --> G["process with expert layer"]
    G --> H["index_add to output"]
```
