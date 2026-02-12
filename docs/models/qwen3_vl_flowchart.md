# Qwen3-VL 多模态模型流程图

## Qwen3-VL 整体架构

### 顶层结构

```mermaid
flowchart TB
    A["Qwen3VLForConditionalGeneration"] --> B["视觉编码器"]
    A --> C["语言模型"]
    B --> D["Qwen3VisionEncoder"]
    C --> E["Qwen3VLTextForCausalLM"]
```

### 视觉编码器结构

```mermaid
flowchart TB
    A["Qwen3VisionEncoder"] --> B["视觉模型"]
    B --> C["Qwen3VLVisionModel"]
    
    C --> D["图像块嵌入"]
    C --> E["位置编码"]
    C --> F["旋转位置编码"]
    C --> G["Transformer Blocks"]
    C --> H["Patch合并层"]
    C --> I["DeepStack合并层"]
```

### 视觉模型组件

```mermaid
flowchart TB
    subgraph VisionComponents["视觉模型组件"]
        direction TB
        A1["PatchEmbed"] 
        A2["Position Embed"]
        A3["Rotary Pos Embed"]
        A4["Vision Blocks x depth"]
        A5["Patch Merger"]
        A6["DeepStack Mergers"]
    end
```

### 语言模型结构

```mermaid
flowchart TB
    A["Qwen3VLTextModel"] --> B["词嵌入"]
    A --> C["文本解码层"]
    A --> D["归一化层"]
    
    B --> E["embed_tokens"]
    C --> F["Decoder Layers"]
    D --> G["RMSNorm"]
```

## Vision Encoder 前向传播

```mermaid
flowchart TD
    A["Qwen3VLVisionModel.forward(pixel_values, grid_thw)"] --> B["patch_embed(pixel_values)"]
    B --> C["hidden_states = tokens.reshape"]
    C --> D["pos_embeds = fast_pos_embed_interpolate"]
    D --> E["hidden_states += pos_embeds"]
    E --> F["rotary_pos_emb = rot_pos_emb"]
    F --> G["position_embeddings = (cos, sin)"]
    G --> H["seq_lengths = grid_thw products"]
    H --> I["deepstack_features = []"]
    I --> J["for layer_idx, block in blocks"]
    J --> K["block(hidden_states, seq_lengths, position_embeddings)"]
    K --> L{"layer_idx in deepstack_visual_indexes?"}
    L -->|"Yes"| M["merger = deepstack_merger_list"]
    M --> N["deepstack_features.append(merger)"]
    L -->|"No"| O["continue"]
    N --> J
    O --> J
    J -->|"done"| P["hidden_states = merger"]
    P --> Q["return hidden_states, deepstack_features"]
```

## Vision Patch Embed

```mermaid
flowchart TD
    A["Qwen3VLVisionPatchEmbed.forward(inputs)"] --> B{"inputs.dim() == 4?"}
    B -->|"Yes"| C["unsqueeze temporal dim"]
    B -->|"No"| D["keep as is"]
    C --> E["proj: Conv3d"]
    D --> E
    E --> F["flatten + transpose"]
    F --> G["return hidden_states"]
```

## Vision Block

```mermaid
flowchart TD
    A["Qwen3VLVisionBlock.forward(hidden_states, seq_lengths, position_embeddings)"] --> B["residual = hidden_states"]
    B --> C["hidden_states = norm1(hidden_states)"]
    C --> D["hidden_states = residual + attn(hidden_states, seq_lengths, position_embeddings)"]
    D --> E["residual = hidden_states"]
    E --> F["hidden_states = norm2(hidden_states)"]
    F --> G["hidden_states = residual + mlp(hidden_states)"]
    G --> H["return hidden_states"]
```

## Vision Attention

```mermaid
flowchart TD
    A["Qwen3VLVisionAttention.forward(hidden_states, seq_lengths, position_embeddings)"] --> B["outputs = []"]
    B --> C["offset = 0"]
    C --> D["cos, sin = position_embeddings"]
    D --> E["for length in seq_lengths"]
    E --> F["chunk = hidden_states[offset:offset+length]"]
    F --> G["qkv = qkv_proj(chunk)"]
    G --> H["q, k, v = qkv.chunk(3, dim=-1)"]
    H --> I["reshape for multi-head"]
    I --> J["q, k = apply_rotary_pos_emb_vision"]
    J --> K["attn_scores = matmul(q, k^T) * scale"]
    K --> L["attn_weights = softmax(attn_scores)"]
    L --> M["attn_output = matmul(attn_weights, v)"]
    M --> N["reshape + proj"]
    N --> O["outputs.append(attn_output)"]
    O --> P["offset += length"]
    P --> E
    E -->|"done"| Q["return torch.cat(outputs)"]
```

## Multimodal Forward

```mermaid
flowchart TD
    A["Qwen3VLForConditionalGeneration.forward(input_ids, positions, pixel_values, ...)"] --> B{"inputs_embeds?"}
    B -->|"None"| C["get_input_embeddings(input_ids)"]
    B -->|"Provided"| D["use inputs_embeds"]
    C --> E["clone inputs_embeds"]
    D --> E
    E --> F["visual_pos_mask = zeros"]
    F --> G{"vision_slices_per_seq?"}
    G -->|"Yes"| H["process_vision_slices"]
    G -->|"No"| I{"pixel_values?"}
    I -->|"Yes"| J["process_pixel_values"]
    I -->|"No"| K["no vision"]
    H --> L["inject vision tokens"]
    J --> L
    K --> M["positions = arange"]
    L --> M
    M --> N["language_model(inputs_embeds, positions, ...)"]
    N --> O["return hidden_states"]
```

## Process Vision Slices

```mermaid
flowchart TD
    A["process_vision_slices"] --> B["check sequence_lengths match"]
    B --> C["calculate offsets"]
    C --> D["for seq_idx in range(len(vision_slices_per_seq))"]
    D --> E["seq_slices = vision_slices_per_seq[seq_idx]"]
    E --> F["for slice_info in seq_slices"]
    F --> G["token_slice = slice_info['tokens']"]
    G --> H["target_start = start + target_offset"]
    H --> I["inputs_embeds[target_start:target_end] = token_slice"]
    I --> J["visual_pos_mask[target_start:target_end] = True"]
    J --> K{"deepstack_slice?"}
    K -->|"Yes"| L["collect deepstack features"]
    K -->|"No"| F
    L --> F
    F -->|"done"| D
    D -->|"done"| M["concat deepstack_layers"]
    M --> N["return vision_token_count, visual_pos_mask, deepstack_layers"]
```

## Load Model

```mermaid
flowchart TD
    A["load_qwen3_vl_model(model_path, config)"] --> B["create Qwen3VLForConditionalGeneration"]
    B --> C["define name_mapping function"]
    C --> D["if weight_name starts with 'model.language_model.'"]
    D --> E["map to language_model.model.*"]
    D --> F["map to language_model.lm_head.*"]
    C --> G["if weight_name starts with 'model.visual.'"]
    G --> H["map to visual.vision.*"]
    C --> I["load_model with name_mapping"]
    I --> J["return model"]
```

## Vision Position Interpolation

```mermaid
flowchart TD
    A["fast_pos_embed_interpolate(grid_thw)"] --> B["grid_ts, grid_hs, grid_ws = grid_thw"]
    B --> C["for t, h, w in zip"]
    C --> D["h_idxs = linspace(0, num_grid_per_side-1, h)"]
    D --> E["w_idxs = linspace(0, num_grid_per_side-1, w)"]
    E --> F["calculate floor/ceil indices"]
    F --> G["calculate interpolation weights"]
    G --> H["4-corner bilinear interpolation"]
    H --> I["C -->|"done"| J["sum weighted embeddings"]
    J --> K["permute and flatten"]
    K --> L["return patch_pos_embeds"]
```

## Text Decoder Layer with DeepStack

```mermaid
flowchart TD
    A["Qwen3VLTextModel.forward with deepstack"] --> B["embed_tokens or inputs_embeds"]
    B --> C["residual = None"]
    C --> D["for layer_idx, layer in layers"]
    D --> E["hidden_states, residual = layer(positions, hidden_states, residual)"]
    E --> F{"deepstack_visual_embeds and layer_idx < len?"}
    F -->|"Yes"| G["ds = deepstack_visual_embeds[layer_idx]"]
    G --> H{"visual_pos_mask?"}
    H -->|"Yes"| I["hidden_states[mask] += ds"]
    H -->|"No"| J["hidden_states[:vision_token_count] += ds"]
    I --> D
    J --> D
    F -->|"No"| D
    D -->|"done"| K["norm(hidden_states, residual)"]
    K --> L["return hidden_states"]
```

## Rotational Position Embedding (Vision)

```mermaid
flowchart TD
    A["rot_pos_emb(grid_thw)"] --> B["merge_size = spatial_merge_size"]
    B --> C["max_hw = max of grid_thw[:, 1:]"]
    C --> D["freq_table = rotary_pos_emb(max_hw)"]
    D --> E["total_tokens = sum(prod(grid_thw, dim=1))"]
    E --> F["pos_ids = empty(total_tokens, 2)"]
    F --> G["for num_frames, height, width in grid_thw"]
    G --> H["calculate merged h/w"]
    H --> I["generate row/col indices"]
    I --> J["coords = stack(row_idx, col_idx)"]
    J --> K{"num_frames > 1?"}
    K -->|"Yes"| L["coords = coords.repeat(num_frames, 1)"]
    K -->|"No"| M["fill pos_ids[offset:offset+num_tokens]"]
    L --> M
    M --> N["offset += num_tokens"]
    N --> G
    G -->|"done"| O["embeddings = freq_table[pos_ids]"]
    O --> P["return embeddings.flatten(1)"]
```
