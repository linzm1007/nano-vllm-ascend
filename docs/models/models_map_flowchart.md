# Models Map 流程图

## 模型映射关系

### 架构名称到实现类的映射

```mermaid
flowchart LR
    subgraph DecoderOnly["Decoder-only 模型"]
        A["LlamaForCausalLM"] --> B["llama.LlamaForCausalLM"]
        C["Qwen3ForCausalLM"] --> D["qwen3.Qwen3ForCausalLM"]
        E["Qwen2ForCausalLM"] --> D
        F["MiniCPMForCausalLM"] --> G["mini_cpm4.MiniCPMForCausalLM"]
    end
```

```mermaid
flowchart LR
    subgraph SpecialModels["特殊架构模型"]
        A["Qwen3MoeForCausalLM"] --> B["qwen3_moe.Qwen3MoeForCausalLM"]
        C["Qwen3VLForConditionalGeneration"] --> D["qwen3_vl.Qwen3VLForConditionalGeneration"]
    end
```

### 映射表

| 架构名称 | 实现类 | 类型 |
|---------|--------|------|
| LlamaForCausalLM | llama.LlamaForCausalLM | Decoder-only |
| Qwen2ForCausalLM | qwen3.Qwen3ForCausalLM | Decoder-only |
| Qwen3ForCausalLM | qwen3.Qwen3ForCausalLM | Decoder-only |
| Qwen3MoeForCausalLM | qwen3_moe.Qwen3MoeForCausalLM | Sparse MoE |
| Qwen3VLForConditionalGeneration | qwen3_vl.Qwen3VLForConditionalGeneration | Multimodal |
| MiniCPMForCausalLM | mini_cpm4.MiniCPMForCausalLM | Decoder-only |

## 模型导入关系

```mermaid
flowchart LR
    subgraph Imports["Import Statements"]
        A["from .llama import LlamaForCausalLM"]
        B["from .mini_cpm4 import MiniCPMForCausalLM"]
        C["from .qwen3 import Qwen3ForCausalLM"]
        D["from .qwen3_moe import Qwen3MoeForCausalLM"]
        E["from .qwen3_vl import Qwen3VLForConditionalGeneration"]
    end
```

## 模型选择流程

```mermaid
flowchart TD
    A["加载模型"] --> B["读取配置文件"]
    B --> C["获取 architectures"]
    C --> D["arch = architectures[0]"]
    D --> E{"arch in model_dict?"}
    E -->|"Yes"| F["model_class = model_dict[arch]"]
    E -->|"No"| G["Raise KeyError"]
    F --> H["model = model_class(config)"]
    H --> I["返回模型实例"]
```

## 支持的模型架构

```mermaid
flowchart LR
    subgraph Supported["支持的模型"]
        direction TB
        Llama["Llama<br/>Decoder-only"]
        Qwen3["Qwen3<br/>Decoder-only"]
        Qwen3MoE["Qwen3-MoE<br/>Sparse MoE"]
        Qwen3VL["Qwen3-VL<br/>Multimodal"]
        MiniCPM4["MiniCPM4<br/>LongRoPE"]
    end
```

## 模型特点对比

```mermaid
flowchart TD
    subgraph Llama["Llama"]
        L1["Standard attention"]
        L2["RMSNorm"]
        L3["RoPE"]
    end

    subgraph Qwen3["Qwen3"]
        Q1["Conditional Q/K norm"]
        Q2["High theta RoPE"]
        Q3["Gated MLP"]
    end

    subgraph Qwen3MoE["Qwen3-MoE"]
        M1["Sparse MoE layers"]
        M2["Top-K routing"]
        M3["Expert selection"]
    end

    subgraph Qwen3VL["Qwen3-VL"]
        V1["Vision encoder"]
        V2["DeepStack features"]
        V3["Multimodal fusion"]
    end

    subgraph MiniCPM4["MiniCPM4"]
        C1["LongRoPE"]
        C2["Depth scaling"]
        C3["Width scaling"]
    end
```

## 使用示例

```mermaid
flowchart TD
    A["from nanovllm.models.models_map import model_dict"]
    B["arch = 'Qwen3ForCausalLM'"]
    C["model_class = model_dict[arch]"]
    D["model = model_class(config)"]
    E["load_model_weights(model, path)"]

    A --> B --> C --> D --> E
```

## 模型类别分布

```mermaid
pie
    title 模型架构分布
    "Llama": 1
    "Qwen3": 1
    "Qwen3-MoE": 1
    "Qwen3-VL": 1
    "MiniCPM4": 1
```

## 架构别名映射

```mermaid
flowchart LR
    subgraph Alias["架构别名"]
        Qwen2["Qwen2ForCausalLM"] --> Qwen3["Qwen3ForCausalLM"]
    end

    subgraph Note["说明"]
        direction TB
        N1["Qwen2和Qwen3架构相同"]
        N2["复用Qwen3实现"]
    end
```
