# LLM 推理优化技术大纲

本文档整理了 LLM（大语言模型）推理领域的关键技术点，按重要性和技术深度分类，帮助开发者系统性地了解推理优化的完整技术栈。

---

## 🔥 第一层：核心技术优化（基础必备）

### 1. KV Cache 管理

KV Cache 是 LLM 推理中占用内存最大的部分，高效的 KV Cache 管理是推理优化的基础。

#### 1.1 PageAttention
- **核心思想**：操作系统虚拟内存分页机制
- **关键技术**：
  - Block 管理（固定大小 block，如 16/32 tokens）
  - Block Table（逻辑位置到物理 block 的映射）
  - 非连续存储
  - Copy-on-Write（写时复制）
- **效果**：内存浪费从 75% 降低到 < 4%
- **实现**：vLLM、本项目 `attention.py`
- **详细文档**：[PageAttention 对比分析](pageattention_comparison.md)

#### 1.2 Prefix Caching（前缀缓存）
- **场景**：多轮对话、System Prompt 共享
- **原理**：缓存相同前缀的 KV Cache，避免重复计算
- **效果**：共享 prompt 的场景吞吐量提升 2-3x
- **挑战**：缓存命中策略、内存管理

#### 1.3 KV Cache 压缩/量化
- **方法**：
  - Key/Value 量化到 INT8/INT4
  - 重要性采样（保留重要 token）
  - 窗口压缩（Sliding Window）
- **代表工作**：H2O、StreamingLLM

---

### 2. Attention 优化

Attention 计算复杂度为 O(N²)，是推理的主要计算瓶颈。

#### 2.1 FlashAttention
- **核心思想**：IO 感知算法，分块计算减少 HBM 访问
- **关键技术**：
  - Tiling（分块）
  - Online Softmax
  - Recomputation
- **效果**：内存从 O(N²) 降至 O(N)，速度提升 2-4x
- **版本演进**：FlashAttention → FlashAttention-2 → FlashAttention-3
- **实现**：`flash-attn` 库、本项目 `attention_ori.py`

#### 2.2 GQA/MQA（注意力分组）
- **GQA**（Group Query Attention）：Query 多组共享同一组 Key/Value
- **MQA**（Multi-Query Attention）：所有 Query 共享同一组 Key/Value
- **效果**：减少 KV Cache 内存，降低带宽压力
- **应用**：Llama-2/3、Qwen、Mistral 等主流模型

#### 2.3 稀疏注意力
- **方法**：
  - Local Attention（局部注意力）
  - Strided Attention（稀疏模式）
  - Blockwise Attention
- **代表工作**：Longformer、BigBird、SparQ Attention

---

### 3. 批处理策略

高效处理多个并发请求是提升吞吐量的关键。

#### 3.1 Continuous Batching（连续批处理）
- **原理**：动态将新请求加入正在运行的 batch
- **vs Static Batching**：避免等待最慢请求，GPU 利用率更高
- **实现**：vLLM、TGI、本项目已实现

#### 3.2 Dynamic Batching（动态批处理）
- **原理**：根据请求到达时间动态组 batch
- **权衡**：延迟 vs 吞吐量

#### 3.3 Inflight Batching
- **原理**：在推理过程中插入新请求
- **场景**：在线服务、高并发场景

---

## 🚀 第二层：性能优化技术（进阶）

### 4. 量化技术（Quantization）

量化通过降低模型精度来减少内存占用和提升计算速度。

#### 4.1 训练后量化（PTQ）
- **INT8 量化**：
  - SmoothQuant：平衡激活和权重的量化难度
  - LLM.int8()：混合精度，大数值保持 FP16
- **INT4 量化**：
  - GPTQ：基于 OBS 的逐层量化
  - AWQ：激活感知权重量化（保护重要权重）
  - GGUF/GGML：llama.cpp 格式，CPU/GPU 通用

#### 4.2 量化感知训练（QAT）
- **方法**：训练时模拟量化误差
- **效果**：比 PTQ 更高的精度
- **代价**：需要重新训练

#### 4.3 FP8 精度
- **硬件**：Hopper（H100）、Ascend 等支持
- **优势**：接近 INT8 的速度，FP16 的精度
- **应用**：训练和推理统一精度

#### 4.4 混合精度推理
- **策略**：
  - Attention 用 FP16，FFN 用 INT8
  - 敏感层 FP16，其他层 INT8/INT4

---

### 5. 投机采样（Speculative Decoding）

使用小模型预测多个 token，大模型并行验证，显著降低延迟。

#### 5.1 Draft Model（草稿模型）
- **架构**：小模型（如 7B）生成候选，大模型（如 70B）验证
- **接受率**：通常 60-80% 的 token 被接受
- **效果**：延迟降低 2-3x，吞吐量提升

#### 5.2 Medusa Decoding
- **原理**：在模型头上添加多个解码头，并行预测未来 N 个 token
- **优势**：无需额外模型
- **训练**：需要微调

#### 5.3 Lookahead Decoding
- **原理**：使用 n-gram 缓存和 Jacobi 迭代
- **优势**：无需草稿模型，零额外内存

#### 5.4 Prompt Lookup Decoding
- **原理**：在 prompt 中查找可复制的 n-gram
- **场景**：文档问答、长文本生成

---

### 6. 解码优化

#### 6.1 Parallel Decoding
- **方法**：同时解码多个候选，选择最优
- **权衡**：计算量增加，但接受率提升

#### 6.2 Token Tree Verification
- **原理**：将多个候选组织成树结构，共享前缀计算
- **应用**：Medusa、Lookahead 等场景

---

## 🏗️ 第三层：系统架构与调度

### 7. 请求调度策略

#### 7.1 基础策略
- **FCFS**（先来先服务）：简单公平，但可能有 head-of-line 阻塞
- **SJF**（最短作业优先）：最小化平均延迟
- **Maximal Scheduling**：最大化 batch size

#### 7.2 高级策略
- **Priority-based**：VIP 用户优先
- **Deadline-aware**：满足延迟 SLA
- **Cost-aware**：考虑计算成本

#### 7.3 Preemption（抢占）
- **场景**：高优先级请求到达，或 KV Cache 不足
- **策略**：
  - Swapping：将 KV Cache 换出到 CPU
  - Recomputation：丢弃后重新计算
  - Chunked Prefill：分块处理长序列

---

### 8. 内存优化

#### 8.1 Memory Pool Management
- **原理**：预分配内存池，避免动态分配开销
- **管理**：分配、回收、碎片整理

#### 8.2 Swapping/Paging
- **场景**：KV Cache 超过 GPU 内存
- **策略**：将不活跃的序列换出到 CPU 内存或磁盘

#### 8.3 Offloading
- **方法**：将部分计算卸载到 CPU
- **场景**：超大模型，单卡放不下
- **框架**：DeepSpeed-Inference、HuggingFace Accelerate

#### 8.4 Activation Checkpointing
- **原理**：前向时不保存中间激活，反向时重新计算
- **权衡**：内存 vs 计算
- **应用**：训练场景为主

---

### 9. 并行策略

#### 9.1 Tensor Parallelism（张量并行）✅ 已实现
- **原理**：将层内计算切分到多卡
- **切分**：Column Parallel（按列切）+ Row Parallel（按行切）
- **通信**：AllReduce、AllGather
- **适用**：单层放不下，如大 hidden_size

#### 9.2 Pipeline Parallelism（流水线并行）
- **原理**：将不同层放到不同卡
- **气泡**：Pipeline Bubble（空闲等待）
- **优化**：GPipe、PipeDream、Interleaved Pipeline

#### 9.3 Expert Parallelism（专家并行 - MoE）
- **场景**：Mixture of Experts 模型
- **原理**：不同 expert 放在不同卡
- **通信**：All-to-All（路由时的 token 交换）

#### 9.4 Sequence Parallelism（序列并行）
- **原理**：长序列切分到多卡
- **应用**：长上下文训练/推理
- **代表**：DeepSpeed Ulysses、Ring Attention

---

## 🧠 第四层：长上下文与特殊场景

### 10. 长上下文处理（Long Context）

#### 10.1 位置编码外推
- **RoPE Scaling**：线性/NTK/动态插值
- **ALiBi**：注意力偏置，天然支持外推
- **xPos**：新的位置编码方法

#### 10.2 上下文压缩
- **StreamingLLM**：保留初始 token 和滑动窗口
- **H2O**：保留 Heavy Hitters（高频 token）
- **SnapKV**：基于重要性的 KV 选择

#### 10.3 Ring Attention
- **原理**：分块循环计算注意力
- **优势**：理论上支持无限长序列
- **局限**：实现复杂，通信开销

---

### 11. 多模态推理优化

#### 11.1 Vision-Language Models（VLM）✅ 已支持
- **架构**：视觉编码器 + 投影层 + 语言模型
- **优化**：
  - 视觉特征缓存
  - 跨模态注意力优化
- **代表**：LLaVA、Qwen-VL、CLIP

#### 11.2 Audio-Language Models
- **场景**：语音对话、音乐理解
- **架构**：音频编码器（如 Whisper）+ LLM

#### 11.3 统一多模态架构
- **趋势**：单一模型处理多种模态
- **代表**：GPT-4o、Gemini、Qwen2.5-Omni

---

### 12. MoE 模型优化 ✅ 已部分支持

#### 12.1 Expert Routing
- **负载均衡**：防止某些 expert 过载
- **辅助损失**：Balance Loss、Router Z-Loss
- **Top-K 选择**：K=1, 2, 4 等

#### 12.2 通信优化
- **All-to-All**：token 在不同 expert 卡间传输
- **EP+DP**：Expert Parallel + Data Parallel
- **细粒度调度**：重叠通信和计算

#### 12.3 Expert Pruning/Clustering
- **方法**：合并相似 expert，剪枝不重要的 expert
- **效果**：减少参数量，提升推理速度

---

## ⚡ 第五层：硬件与底层优化

### 13. 图编译优化

#### 13.1 PyTorch 生态
- **Torch.compile**：PyTorch 2.0 的编译模式
- **TorchInductor**：默认编译后端
- **Triton**：Python 编写 GPU kernel

#### 13.2 厂商专用 ✅ 已支持
- **TorchAir/Ascend IR**：华为昇腾图编译
- **TensorRT-LLM**：NVIDIA 推理引擎
- **XLA**：Google 的线性代数编译器
- **ONNX Runtime**：跨平台推理

#### 13.3 图优化技术
- **算子融合**：减少 kernel 启动开销
- **常量折叠**：编译时预计算
- **死代码消除**：移除无用计算

---

### 14. 算子融合（Operator Fusion）

#### 14.1 常见融合模式
- **QKV Fusion**：合并 Q、K、V 投影 ✅ 已实现
- **Attention-MLP Fusion**：Attention 后接 MLP
- **Norm-Activation**：LayerNorm + Activation
- **Bias-Add**：线性层 + Bias + 激活

#### 14.2 自定义 Kernel
- **CUDA Kernel**：手写高性能 kernel
- **Triton**：Python 编写 GPU kernel，易于开发
- **CUTLASS**：NVIDIA 的模板库

---

### 15. 通信优化

#### 15.1 通信库
- **NCCL**：NVIDIA Collective Communication Library
- **HCCL**：Huawei Collective Communication Library
- **Gloo**：Facebook 的 CPU 通信库

#### 15.2 优化技术
- **RDMA**：远程直接内存访问，低延迟
- **GPUDirect**：GPU 间直接通信
- **Zero-Copy**：零拷贝数据传输
- **Communication Overlapping**：重叠通信和计算

---

## 📊 第六层：评估与可观测性

### 16. 性能分析与 Profiling

#### 16.1 内存分析
- **Memory Profiler**：PyTorch Memory Profiler、Nsight
- **内存占用**：模型权重、KV Cache、激活值
- **内存带宽**：HBM 带宽利用率

#### 16.2 计算分析
- **Compute Utilization**：GPU 计算单元利用率
- **Roofline Model**：分析计算瓶颈（内存 bound 或计算 bound）
- **Kernel Timeline**：算子执行时间线

#### 16.3 瓶颈识别
- **Bottleneck Analysis**：定位性能瓶颈
- **Amdahl's Law**：优化收益评估

---

### 17. 关键性能指标

#### 17.1 延迟指标
- **TTFT**（Time To First Token）：首 token 延迟
- **TPOT**（Time Per Output Token）：每个输出 token 的延迟
- **TBT**（Time Between Tokens）：流式输出间隔

#### 17.2 吞吐量指标
- **Throughput**：tokens/s、requests/s
- **Goodput**：满足延迟约束的吞吐量
- **Batch Size**：并发请求数

#### 17.3 资源指标
- **GPU Utilization**：GPU 利用率
- **Memory Usage**：显存占用
- **Power Consumption**：功耗

---

## 🔮 第七层：前沿技术趋势

### 18. 模型架构创新

#### 18.1 State Space Models（SSM）
- **Mamba**：线性复杂度注意力替代
- **RWKV**：RNN 和 Transformer 的结合
- **优势**：长序列、低内存、快速推理

#### 18.2 Mixture of Depths
- **原理**：不同 token 使用不同计算深度
- **效果**：动态分配计算资源

#### 18.3 RetNet
- **原理**：保留机制替代注意力
- **优势**：并行训练，常数复杂度推理

---

### 19. 服务化与部署

#### 19.1 Disaggregated Serving
- **原理**：Prefill 和 Decode 阶段分离部署
- **优势**：分别优化两个阶段（Prefill 计算密集，Decode 内存密集）
- **代表**：Splitwise、DistServe

#### 19.2 Elastic Scaling
- **自动扩缩容**：根据负载自动调整实例数
- **Serverless**：按需付费，自动管理

#### 19.3 A/B Testing
- **模型版本对比**：新模型效果验证
- **流量分配**：灰度发布

---

### 20. 新兴优化方向

#### 20.1 推理蒸馏
- **原理**：大模型生成数据，小模型学习
- **应用**：降低部署成本

#### 20.2 Early Exit
- **原理**：简单样本提前退出，不深推理
- **方法**：置信度阈值、学习退出点

#### 20.3 Hardware-Aware NAS
- **原理**：针对特定硬件设计模型架构
- **目标**：最大化硬件利用率

---

## 🎯 学习路径建议

### 初学者（1-3 个月）
1. ✅ 理解 PageAttention 和 Continuous Batching
2. ✅ 掌握 FlashAttention 原理
3. ✅ 学习基础量化（INT8/INT4）
4. ✅ 熟悉张量并行和流水线并行

### 进阶（3-6 个月）
5. 📚 深入投机采样（Speculative Decoding）
6. 📚 研究长上下文优化（Long Context）
7. 📚 掌握图编译和算子融合
8. 📚 学习 MoE 模型优化

### 专家（6 个月以上）
9. 🔬 自定义 CUDA/Triton Kernel 开发
10. 🔬 系统级性能调优
11. 🔬 前沿技术跟进（Mamba、Disaggregated Serving 等）
12. 🔬 硬件协同设计

---

## 📚 参考资源

### 论文
- **vLLM**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- **FlashAttention-2**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- **Speculative Decoding**: "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- **Medusa**: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024)

### 项目
- **vLLM**: https://github.com/vllm-project/vllm
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Text Generation Inference**: https://github.com/huggingface/text-generation-inference

### 文档
- **本项目 PageAttention 分析**: [pageattention_comparison.md](pageattention_comparison.md)
- **PyTorch Performance Tuning**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **NVIDIA Performance Guide**: https://docs.nvidia.com/deeplearning/performance/index.html



