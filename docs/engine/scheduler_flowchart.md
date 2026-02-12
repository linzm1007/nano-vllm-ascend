# Scheduler 流程图

## Scheduler 整体架构

```mermaid
flowchart TD
    subgraph Scheduler["Scheduler 调度器"]
        waiting["waiting: deque[Sequence]<br/>等待队列"]
        running["running: deque[Sequence]<br/>运行队列"]
        block_mgr["block_manager<br/>块管理器"]
    end

    subgraph Limits["限制参数"]
        max_num_seqs["max_num_seqs<br/>最大序列数"]
        max_num_batched_tokens["max_num_batched_tokens<br/>最大批处理token数"]
        max_model_len["max_model_len<br/>最大模型长度"]
    end
```

## 调度主流程 (schedule)

```mermaid
flowchart TD
    A["schedule()"] --> B["初始化:<br/>scheduled_seqs=[], num_seqs=0, num_batched_tokens=0"]
    B --> C{"waiting 队列有数据<br/>且 num_seqs < max_num_seqs?"}
    C -->|"是"| D["获取 seq = waiting[0]"]
    D --> E{"检查约束条件"}
    E -->|"通过"| F["从 waiting 弹出 seq"]
    F --> G["block_manager.allocate(seq)"]
    G --> H["seq.status = RUNNING"]
    H --> I["添加到 running 队列"]
    I --> J["添加到 scheduled_seqs"]
    J --> K["num_seqs++"]
    K --> L["num_batched_tokens += len(seq) - cached"]
    L --> C
    E -->|"不通过"| M["跳出循环"]
    C -->|"否"| N{"scheduled_seqs 不为空?"}
    N -->|"是"| O["返回 (scheduled_seqs, True)<br/>True表示是prefill"]
    N -->|"否"| P["进入 decode 阶段"]
    M --> P
    P --> Q{"running 队列有数据<br/>且 num_seqs < max_num_seqs?"}
    Q -->|"是"| R["从 running 弹出 seq"]
    R --> S["检查 can_append"]
    S -->|"失败"| T{"running 还有数据?"}
    T -->|"是"| U["preempt(running.pop())"]
    T -->|"否"| V["preempt(seq), seq=None"]
    U --> S
    V --> Q
    S -->|"通过"| W["num_seqs++"]
    W --> X["may_append(seq)"]
    X --> Y["添加到 scheduled_seqs"]
    Y --> Q
    Q -->|"否"| Z{"scheduled_seqs 不为空?"}
    Z -->|"是"| AA["running.extendleft(reversed)<br/>保持顺序"]
    AA --> AB["返回 (scheduled_seqs, False)<br/>False表示是decode"]
    Z -->|"否"| AC["返回 ([], False)"]
```

## Prefill 调度流程

```mermaid
flowchart TD
    A["开始 Prefill 调度"] --> B["检查 waiting 队列"]
    B --> C["取 waiting[0]"]
    C --> D["检查 can_allocate"]
    D -->|"失败"| E["停止调度"]
    D -->|"通过"| F["检查 token 数量限制"]
    F -->|"超过 max_num_batched_tokens"| E
    F -->|"通过"| G["分配块"]
    G --> H["状态改为 RUNNING"]
    H --> I["移到 running 队列"]
    I --> J{"还有空间?"}
    J -->|"是"| B
    J -->|"否"| K["返回调度结果"]
```

## Decode 调度流程

```mermaid
flowchart TD
    A["开始 Decode 调度"] --> B["从 running 取 seq"]
    B --> C{"can_append(seq)?"}
    C -->|"是"| D["准备 decode"]
    D --> E["返回调度列表"]
    C -->|"否"| F{"running 还有 seq?"}
    F -->|"是"| G["preempt 最后一个"]
    G --> C
    F -->|"否"| H["preempt 当前 seq"]
    H --> I["seq = None"]
    I --> J{"还有更多 seq?"}
    J -->|"是"| B
    J -->|"否"| K["返回空列表"]
```

## 抢占流程 (preempt)

```mermaid
flowchart TD
    A["preempt(seq)"] --> B["seq.status = WAITING"]
    B --> C["seq.finish_reason = PREEMPTED"]
    C --> D["block_manager.deallocate(seq)"]
    D --> E["添加到 waiting 队列头部"]
```

## 后处理流程 (postprocess)

```mermaid
flowchart TD
    A["postprocess(seqs, token_ids)"] --> B["遍历 seqs 和 token_ids"]
    B --> C["seq.append_token(token_id)"]
    C --> D{"检查完成条件"}
    D -->|"is_eos"| E["free_seq(seq, EOS)"]
    D -->|"is_max_tokens"| F["free_seq(seq, LENGTH)"]
    D -->|"is_max_model_len"| G["free_seq(seq, LENGTH)"]
    D -->|"未完成"| H["继续"]
    E --> I["从 running 移除"]
    F --> I
    G --> I
```

## 添加请求流程 (add)

```mermaid
flowchart TD
    A["add(seq)"] --> B["添加到 waiting 队列尾部"]
```

## 取消请求流程 (abort_seq_group)

```mermaid
flowchart TD
    A["abort_seq_group(request_id)"] --> B["遍历 waiting 和 running"]
    B --> C{"匹配 request_id?"}
    C -->|"是"| D["从队列移除"]
    D --> E["free_seq(seq, ABORTED)"]
    C -->|"否"| F["下一个"]
```

## 释放序列 (free_seq)

```mermaid
flowchart TD
    A["free_seq(seq, reason)"] --> B["seq.status = FINISHED"]
    B --> C["seq.finish_reason = reason"]
    C --> D["block_manager.deallocate(seq)"]
```

## 队列状态流转

```mermaid
flowchart LR
    subgraph 状态转换
        W["WAITING<br/>等待队列"] -->|"schedule()<br/>调度"| R["RUNNING<br/>运行队列"]
        R -->|"preempt()<br/>抢占"| W
        R -->|"完成<br/>EOS/max_tokens"| F["FINISHED<br/>完成"]
        W -->|"abort<br/>取消"| F
    end
```
