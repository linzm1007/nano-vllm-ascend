# Linear Layers 流程图

## 类继承关系

```mermaid
flowchart TD
    A["nn.Module"] --> B["LinearBase"]
    B --> C["ReplicatedLinear"]
    B --> D["ColumnParallelLinear"]
    B --> E["RowParallelLinear"]
    D --> F["MergedColumnParallelLinear"]
    D --> G["QKVParallelLinear"]
```

## LinearBase 初始化流程

```mermaid
flowchart TD
    A["LinearBase.__init__"] --> B["获取 tp_rank, tp_size"]
    B --> C["创建 weight Parameter"]
    C --> D["设置 weight_loader"]
    D --> E{"bias?"}
    E -->|Yes| F["创建 bias Parameter"]
    E -->|No| G["register_parameter=None"]
    F --> H["设置 weight_loader"]
```

## ReplicatedLinear

```mermaid
flowchart TD
    subgraph Replicated["ReplicatedLinear (复制线性层)"]
        A["weight_loader: param.copy_(loaded_weight)"]
        B["forward: F.linear(x, weight, bias)"]
    end
```

## ColumnParallelLinear

```mermaid
flowchart TD
    A["ColumnParallelLinear.__init__"] --> B["output_size = output_size // tp_size"]
    B --> C["tp_dim = 0 (按列切分)"]
    
    D["weight_loader"] --> E["计算 shard_size"]
    E --> F["start_idx = tp_rank * shard_size"]
    F --> G["narrow(loaded_weight, tp_dim, start_idx, shard_size)"]
    G --> H["param_data.copy_(loaded_weight)"]
    
    I["forward"] --> J["F.linear(x, weight, bias)"]
```

## RowParallelLinear

```mermaid
flowchart TD
    A["RowParallelLinear.__init__"] --> B["input_size = input_size // tp_size"]
    B --> C["tp_dim = 1 (按行切分)"]
    
    D["weight_loader"] --> E["同 ColumnParallelLinear"]
    
    F["forward"] --> G["y = F.linear(x, weight, bias if tp_rank==0 else None)"]
    G --> H{"tp_size > 1?"}
    H -->|Yes| I["dist.all_reduce(y)"]
    H -->|No| J["return y"]
    I --> J
```

## MergedColumnParallelLinear

```mermaid
flowchart TD
    A["MergedColumnParallelLinear.__init__"] --> B["接收 output_sizes: list[int]"]
    B --> C["output_size = sum(output_sizes)"]
    C --> D["调用父类 __init__"]
    
    E["weight_loader(param, loaded_weight, loaded_shard_id)"] --> F["计算 shard_offset"]
    F --> G["shard_offset = sum(output_sizes[:loaded_shard_id]) // tp_size"]
    G --> H["shard_size = output_sizes[loaded_shard_id] // tp_size"]
    H --> I["param_data = param.narrow(tp_dim, shard_offset, shard_size)"]
    I --> J["loaded_weight = loaded_weight.chunk(tp_size, tp_dim)[tp_rank]"]
    J --> K["param_data.copy_(loaded_weight)"]
```

## QKVParallelLinear

```mermaid
flowchart TD
    A["QKVParallelLinear.__init__"] --> B["参数: hidden_size, head_size, total_num_heads, total_num_kv_heads"]
    B --> C["num_heads = total_num_heads // tp_size"]
    C --> D["num_kv_heads = total_num_kv_heads // tp_size"]
    D --> E["output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size"]
    E --> F["调用父类 ColumnParallelLinear"]
    
    G["weight_loader(param, loaded_weight, loaded_shard_id)"] --> H{"loaded_shard_id?"}
    H -->|"q"| I["shard_offset = 0"]
    H -->|"k"| J["shard_offset = num_heads * head_size"]
    H -->|"v"| K["shard_offset = num_heads * head_size + num_kv_heads * head_size"]
    I --> L["shard_size = num_heads * head_size"]
    J --> M["shard_size = num_kv_heads * head_size"]
    K --> M
    L --> N["param_data = param.narrow(tp_dim, shard_offset, shard_size)"]
    M --> N
    N --> O["加载权重并复制"]
```

## 张量并行策略对比

```mermaid
flowchart LR
    subgraph Column["Column Parallel"]
        A1["按输出维度切分"]
        A2["tp_dim = 0"]
        A3["每个rank计算部分输出"]
    end
    
    subgraph Row["Row Parallel"]
        B1["按输入维度切分"]
        B2["tp_dim = 1"]
        B3["需要 all_reduce"]
    end
    
    subgraph Replicated["Replicated"]
        C1["不切分"]
        C2["每个rank有完整副本"]
    end
```

## 权重加载流程

```mermaid
sequenceDiagram
    participant Loader as Model Loader
    participant Param as Parameter
    participant WeightLoader as weight_loader
    
    Loader->>Param: 加载 checkpoint weight
    Param->>WeightLoader: 调用 weight_loader(param, loaded_weight)
    WeightLoader->>WeightLoader: 计算 shard_offset/size
    WeightLoader->>WeightLoader: narrow 切片
    WeightLoader->>Param: copy_ 到 param.data
```
