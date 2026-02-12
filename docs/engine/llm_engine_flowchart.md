# LLMEngine 流程图

## LLMEngine 整体架构

```mermaid
flowchart TD
    subgraph LLMEngine["LLMEngine 大语言模型引擎"]
        config["config: 配置"]
        tokenizer["tokenizer: 分词器"]
        scheduler["scheduler: 调度器"]
        model_runner["model_runner: 模型运行器"]
        ps["ps: 多进程列表"]
        events["events: 同步事件"]
    end
```

## 初始化流程 (__init__)

```mermaid
flowchart TD
    A["__init__(model, **kwargs)"] --> B["创建 Config"]
    B --> C["保存 config 和 block_size"]
    C --> D["创建进程列表和事件列表"]
    D --> E["获取多进程上下文 spawn"]
    E --> F["启动 tensor_parallel_size-1 个工作进程"]
    F --> G["每个进程运行 ModelRunner"]
    G --> H["主进程创建 ModelRunner"]
    H --> I["加载 Tokenizer"]
    I --> J["设置 eos_token"]
    J --> K["创建 Scheduler"]
    K --> L["warmup_model()"]
    L --> M["注册 exit 退出处理"]
```

## Warmup 流程

```mermaid
flowchart TD
    A["warmup_model()"] --> B["记录开始时间"]
    B --> C["prefill_warmup()"]
    C --> D["decode_warmup()"]
    D --> E["记录结束时间"]
    E --> F["计算耗时"]
    F --> G["输出 warmup 完成日志"]

    C --> H["计算 num_seqs"]
    H --> I["生成随机 prompt_token_ids"]
    I --> J["创建 SamplingParams"]
    J --> K["调用 generate()<br/>prefill 模式"]

    D --> L["计算 num_seqs"]
    L --> M["生成随机 prompts"]
    M --> N{"enforce_eager?"}
    N -->|"是"| O["直接 warmup"]
    N -->|"否"| P{"有 graph cache?"}
    P -->|"是"| Q["加载缓存日志"]
    P -->|"否"| O
    Q --> O
```

## 添加请求 (add_request)

```mermaid
flowchart TD
    A["add_request(prompt, sampling_params, request_id, ...)"] --> B{"prompt 是 str?"}
    B -->|"是"| C["tokenizer.encode(prompt)"]
    B -->|"否"| D["保持 token_ids"]
    C --> E["创建 Sequence 对象"]
    D --> E
    E --> F["设置所有参数"]
    F --> G["scheduler.add(seq)"]
```

## Step 执行流程

```mermaid
flowchart TD
    A["step()"] --> B["scheduler.schedule()"]
    B --> C["返回 (seqs, is_prefill)"]
    C --> D["model_runner.call('run', seqs, is_prefill)"]
    D --> E["返回 token_ids"]
    E --> F["scheduler.postprocess(seqs, token_ids)"]
    F --> G["收集完成的序列"]
    G --> H["计算 num_tokens"]
    H --> I["返回 (outputs, num_tokens)"]
```

## Generate 主流程

```mermaid
flowchart TD
    A["generate(prompts, sampling_params, use_tqdm=True)"] --> B{"use_tqdm?"}
    B -->|"是"| C["创建 tqdm 进度条"]
    B -->|"否"| D["无进度条"]
    C --> E["sampling_params 转列表"]
    D --> E
    E --> F["遍历 prompts"]
    F --> G["add_request(prompt, sp)"]
    G --> F
    F -->|"完成"| H["初始化 outputs 字典"]
    H --> I{"scheduler.is_finished()?"}
    I -->|"否"| J["记录开始时间"]
    J --> K["step()"]
    K --> L["返回 (output, num_tokens)"]
    L --> M{"use_tqdm?"}
    M -->|"是"| N["计算吞吐量"]
    N --> O["更新进度条"]
    M -->|"否"| P
    O --> P["处理 outputs"]
    P --> Q["添加到 outputs 字典"]
    Q --> I
    I -->|"是"| R["排序 outputs"]
    R --> S["解码 token_ids"]
    S --> T["构建结果字典"]
    T --> U{"use_tqdm?"}
    U -->|"是"| V["关闭进度条"]
    U -->|"否"| W
    V --> W["返回结果列表"]
```

## 多模态生成流程

```mermaid
flowchart TD
    A["generate_multimodal(requests, sampling_params, processor)"] --> B{"use_tqdm?"}
    B -->|"是"| C["创建 tqdm 进度条"]
    B -->|"否"| D
    C --> E["_mm_add_request()"]
    D --> E
    E --> F["初始化 outputs"]
    F --> G{"is_finished()?"}
    G -->|"否"| H["step()"]
    H --> I["处理 outputs"]
    I --> J["更新进度条"]
    J --> G
    G -->|"是"| K["排序 outputs"]
    K --> L["get_mm_results()"]
    L --> M{"use_tqdm?"}
    M -->|"是"| N["关闭进度条"]
    M -->|"否"| O
    N --> O["返回结果"]

    E --> P["遍历 requests"]
    P --> Q["处理 messages/text/images"]
    Q --> R["processor.apply_chat_template"]
    R --> S["提取 images"]
    S --> T["processor() 处理"]
    T --> U["获取 input_ids, pixel_values, grid"]
    U --> V{"有 image_grid_thw?"}
    V -->|"是"| W["_expand_vision_placeholders"]
    V -->|"否"| X
    W --> Y["移到 CPU"]
    X --> Z["add_request()"]
    Y --> Z
```

## 视觉占位符扩展流程

```mermaid
flowchart TD
    A["_expand_vision_placeholders(input_ids, image_grid_thw)"] --> B["获取 vision_config"]
    B --> C["获取 merge_size"]
    C --> D["检查 token_ids"]
    D --> E["验证 image_grid_thw 形状"]
    E --> F["计算 expected_counts"]
    F --> G["遍历 input_ids"]
    G --> H{"遇到 vision_start_token?"}
    H -->|"是"| I["添加 start_token"]
    I --> J["跳过原内容直到 end_token"]
    J --> K["计算 required 数量"]
    K --> L["扩展 image_token_ids"]
    L --> M["添加 end_token"]
    M --> N["记录 placeholder_ranges"]
    H -->|"否"| O["添加当前 token"]
    N --> G
    O --> G
    G -->|"完成"| P{"image_idx == total_images?"}
    P -->|"否"| Q["报错: 不匹配"]
    P -->|"是"| R["返回结果"]
```

## 退出流程 (exit)

```mermaid
flowchart TD
    A["exit()"] --> B["model_runner.call('exit')"]
    B --> C["删除 model_runner"]
    C --> D["join 所有工作进程"]
```

## 请求取消流程

```mermaid
flowchart TD
    A["abort_request(request_id)"] --> B["scheduler.abort_seq_group(request_id)"]
    B --> C["从 waiting/running 移除"]
    C --> D["标记为 ABORTED 状态"]
```

## LLMEngine 数据流向

```mermaid
flowchart LR
    subgraph Input["输入层"]
        prompt["prompt<br/>文本/图片"]
        params["sampling_params<br/>采样参数"]
    end

    subgraph Process["处理层"]
        tokenize["Tokenizer<br/>编码"]
        schedule["Scheduler<br/>调度"]
        run["ModelRunner<br/>推理"]
    end

    subgraph Output["输出层"]
        tokens["token_ids<br/>生成的token"]
        decode["Tokenizer<br/>解码"]
        text["text<br/>最终文本"]
    end

    prompt --> tokenize
    params --> schedule
    tokenize --> schedule
    schedule --> run
    run --> tokens
    tokens --> decode
    decode --> text
```

## 主循环时序图

```mermaid
sequenceDiagram
    participant Client
    participant LLMEngine
    participant Scheduler
    participant ModelRunner
    participant Sampler

    Client->>LLMEngine: add_request(prompt, params)
    LLMEngine->>LLMEngine: tokenizer.encode
    LLMEngine->>Scheduler: add(seq)
    Scheduler->>Scheduler: waiting.append(seq)

    loop while not finished
        Client->>LLMEngine: step()
        LLMEngine->>Scheduler: schedule()
        Scheduler->>Scheduler: prefill or decode
        Scheduler-->>LLMEngine: (seqs, is_prefill)
        
        LLMEngine->>ModelRunner: call('run', seqs, is_prefill)
        ModelRunner->>ModelRunner: prepare input
        ModelRunner->>ModelRunner: model forward
        ModelRunner->>Sampler: sample(logits)
        Sampler-->>ModelRunner: token_ids
        ModelRunner-->>LLMEngine: token_ids
        
        LLMEngine->>Scheduler: postprocess(seqs, token_ids)
        Scheduler->>Scheduler: append_token
        Scheduler->>Scheduler: check finish
        Scheduler-->>LLMEngine: completed seqs
    end

    LLMEngine->>LLMEngine: tokenizer.decode
    LLMEngine-->>Client: results
```
