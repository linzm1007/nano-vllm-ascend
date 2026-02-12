# BlockManager 流程图

## Block 结构

```mermaid
flowchart TD
    subgraph Block["Block 块结构"]
        block_id["block_id: 块ID"]
        ref_count["ref_count: 引用计数"]
        hash["hash: 哈希值"]
        token_ids["token_ids: Token列表"]
    end
```

## BlockManager 初始化

```mermaid
flowchart TD
    A["__init__"] --> B["初始化 block_size"]
    B --> C["创建 blocks 列表<br/>[Block(0), Block(1), ...]"]
    C --> D["初始化 hash_to_block_id<br/>空字典"]
    D --> E["初始化 free_block_ids<br/>deque([0, 1, 2, ...])"]
    E --> F["初始化 used_block_ids<br/>空集合"]
    F --> G["初始化 non_cache_token_ids<br/>多模态token不缓存"]
```

## 块分配流程 (allocate)

```mermaid
flowchart TD
    A["allocate(seq)"] --> B{"seq.block_table 为空?"}
    B -->|"否"| C["报错: 已有块表"]
    B -->|"是"| D["初始化 h=-1, cache_miss=false"]
    D --> E["遍历 seq 的所有块"]
    E --> F["获取 token_ids = seq.block(i)"]
    F --> G{"包含 non_cache_token?"}
    G -->|"是"| H["cache_miss = true"]
    G -->|"否"| I{"块大小等于 block_size?"}
    H --> I
    I -->|"是"| J["计算哈希 h"]
    I -->|"否"| K["h = -1"]
    J --> L["查找 hash_to_block_id.get(h)"]
    K --> L
    L --> M{"找到 block_id 且<br/>token_ids 匹配?"}
    M -->|"否"| N["cache_miss = true"]
    M -->|"是"| O["命中缓存"]
    N --> P["从 free_block_ids 取新块"]
    P --> Q["_allocate_block(block_id)"]
    O --> R["seq.num_cached_tokens += block_size"]
    R --> S{"block_id 在 used 中?"}
    S -->|"是"| T["ref_count++"]
    S -->|"否"| U["_allocate_block(block_id)"]
    Q --> V{"h != -1?"}
    T --> V
    U --> V
    V -->|"是"| W["更新 block 的 hash 和 token_ids"]
    V -->|"否"| X["添加到 seq.block_table"]
    W --> Y["更新 hash_to_block_id[h]"]
    Y --> X
    X --> Z{"还有下一块?"}
    Z -->|"是"| E
    Z -->|"否"| AA["结束"]
```

## 块释放流程 (deallocate)

```mermaid
flowchart TD
    A["deallocate(seq)"] --> B["逆序遍历 seq.block_table"]
    B --> C["获取 block = blocks[block_id]"]
    C --> D["block.ref_count--"]
    D --> E{"ref_count == 0?"}
    E -->|"是"| F["_deallocate_block(block_id)"]
    E -->|"否"| G{"还有下一块?"}
    F --> G
    G -->|"是"| B
    G -->|"否"| H["清空 seq.block_table"]
```

## _allocate_block 内部流程

```mermaid
flowchart TD
    A["_allocate_block(block_id)"] --> B["获取 block = blocks[block_id]"]
    B --> C{"验证 ref_count == 0"}
    C -->|"失败"| D["断言错误"]
    C -->|"通过"| E["block.reset()<br/>ref_count=1, hash=-1, token_ids=[]"]
    E --> F["从 free_block_ids 移除 block_id"]
    F --> G["添加到 used_block_ids"]
    G --> H["返回 block"]
```

## _deallocate_block 内部流程

```mermaid
flowchart TD
    A["_deallocate_block(block_id)"] --> B{"验证 blocks[block_id].ref_count == 0"}
    B -->|"失败"| C["断言错误"]
    B -->|"通过"| D["从 used_block_ids 移除"]
    D --> E["添加到 free_block_ids 末尾"]
```

## 哈希计算 (compute_hash)

```mermaid
flowchart TD
    A["compute_hash(token_ids, prefix=-1)"] --> B["创建 xxhash.xxh64()"]
    B --> C{"prefix != -1?"}
    C -->|"是"| D["更新哈希: prefix"]
    C -->|"否"| E["转换 token_ids 为 numpy 数组"]
    D --> E
    E --> F["更新哈希: token_ids 字节"]
    F --> G["返回 h.intdigest()"]
```

## 是否能分配 (can_allocate)

```mermaid
flowchart TD
    A["can_allocate(seq)"] --> B["比较 free_block_ids 数量"]
    B --> C["与 seq.num_blocks"]
    C --> D["返回 >= 结果"]
```

## 是否能追加 (can_append)

```mermaid
flowchart TD
    A["can_append(seq)"] --> B["计算条件:<br/>len(seq) % block_size == 1"]
    B --> C{"需要新块?"}
    C -->|"是"| D["检查 free_block_ids >= 1"]
    C -->|"否"| E["不需要新块,返回 True"]
```

## 可能追加 (may_append)

```mermaid
flowchart TD
    A["may_append(seq)"] --> B["获取 last_block"]
    B --> C{"len(seq) % block_size?"}
    C -->|"== 1"| D{"验证 hash != -1"}
    D -->|"通过"| E["取 free_block_ids[0]"]
    E --> F["_allocate_block(block_id)"]
    F --> G["添加到 block_table"]
    C -->|"== 0"| H{"验证 hash == -1"}
    H -->|"通过"| I["获取 token_ids"]
    I --> J["计算 prefix hash"]
    J --> K["计算当前块 hash"]
    K --> L["更新 block 的 hash 和 token_ids"]
    L --> M["更新 hash_to_block_id"]
    C -->|"其他"| N{"验证 hash == -1"}
    N -->|"通过"| O["无操作"]
```

## 缓存命中机制

```mermaid
flowchart TD
    subgraph 缓存查找
        A["输入: token_ids"] --> B["计算 hash"]
        B --> C["查找 hash_to_block_id"]
        C --> D{"找到 block_id?"}
        D -->|"是"| E["比较 token_ids"]
        D -->|"否"| F["缓存未命中"]
        E -->|"匹配"| G["缓存命中!"]
        E -->|"不匹配"| F
    end

    subgraph 非缓存Token
        H["检查 token_ids"] --> I{"包含 image_token<br/>或 vision_token?"}
        I -->|"是"| J["强制 cache_miss"]
        I -->|"否"| K["正常缓存流程"]
    end
```
