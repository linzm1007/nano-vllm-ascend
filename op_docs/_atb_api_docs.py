import torch_npu


def _add_torch_npu_atb_docstr(method, docstr):
    """Add doc to atb operator API.
    """
    func = getattr(torch_npu.atb, method, None)
    if not func:
        return
    getattr(torch_npu.atb, method).__doc__ = docstr


def _add_torch_npu_atb_api_docstr():
    _add_torch_npu_atb_docstr(
        "npu_multi_head_latent_attention",
        """
torch_npu.atb.npu_multi_head_latent_attention(q_nope, q_rope, ctkv, k_rope, block_tables, context_lens, q_headnum, qk_scale, kv_headnum, *, mask=None, qseqlen=None, qk_descale=None, pv_descale=None, mask_type=None, calc_type=None, cache_mode=None, output=None) -> Tensor
torch_npu.atb.npu_multi_head_latent_attention(q_nope, q_rope, ctkv, k_rope, block_tables, context_lens, q_headnum, qk_scale, kv_headnum, return_lse, *, mask=None, qseqlen=None, qk_descale=None, pv_descale=None, mask_type=None, calc_type=None, cache_mode=None, output=None, lse=None) -> (Tensor, Tensor)
功能描述
MLA场景，使用分页管理的kvcache计算attention score，额外支持分离qnope/qrope、ctkv/krope的输入。第一个接口不支持返回lse输出，第二个接口支持根据return_lse值判断是否返回lse输出。

参数说明
位置参数：
q_nope: Device Tensor, 无位置编码query, 支持float16/bf16/int8， shape为[num_tokens, num_heads, 512]
q_rope: Device Tensor, 旋转位置编码query, 支持float16/bf16， shape为[num_tokens, num_heads, 64]
ctkv: Device Tensor, 无位置编码ctkv, 支持float16/bf16/int8数据类型，cache_mode为'krope_ctkv'时，shape为[num_blocks , block_size, kv_heads, 512]，cache_mode为'int8_nzcache'时，shape为[blockNum, kv_heads*512/32,block_size, 32]，数据类型为int8, 格式NZ。
k_rope: Device Tensor, 旋转位置编码k, 支持float16/bf16，cache_mode为'krope_ctkv', shape为[num_blocks , block_size, kv_heads, 64]，cache_mode为'int8_nzcache'时， shape为[blockNum, kv_heads*64 / 16 ,block_size, 16]，格式为NZ。
block_tables: Device Tensor, 每个query的kvcache的block映射表，shape为[batch, max_num_blocks_per_query]，数据类型为int32。
context_lens：int类型数组，每个query对应的上下文长度，kseqlen。
q_headnum：int类型，query头数量。
qk_scale: float类型，Q*K^T后乘以的缩放系数
kv_headnum：int类型，kv头数量。
return_lse：bool类型，是否返回lse输出。
关键字参数：
mask: 可选Device Tensor,  mask_type为默认场景时，可不传。
qseqlen: 可选Device Tensor, cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。  
qk_descale: 可选Device Tensor,  cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。
qk_descale: 可选Device Tensor,  cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。
mask_type：可选字符串，设置mask类型，缺省值为'undefined', 当前支持'undefined'，无mask.
calc_type：可选字符串，设置mask类型，缺省值为'calc_type_undefined', 当前支持'calc_type_undefined'（默认的decoder场景。）.
cache_mode: 可选字符串，输入query和kcache的类型, 缺省值为'krope_ctkv'(输入的q拆分为qNope和qRope，输入的kcache拆分为ctKV和kRope), 当前支持'krope_ctkv', 'int8_nzcache'(性能cache，在KROPE_CTKV的基础上：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。)。
output: 可选Device Tensor，attention输出，shape为[num_tokens, num_heads, head_size_vo]，数据类型为float16/bf16。
lse: 可选Device Tensor，lse输出，shape为[num_tokens, num_heads, 1]，数据类型为float16/bf16/float。
规格约束
block_size ＜= 128，建议为128。
batch <= 8192
cache_mode为int8_nzcache, 不支持num_heads = 128

参考样例：
import torch,torch_npu
block_size = 128
num_tokens = 32
num_heads = 32
kv_heads =  1
head_size_qk = 576
head_size_vo = 512
batch = num_tokens
num_blocks = 64
max_num_blocks_per_query = 16

q_node = torch.randn((num_tokens, num_heads, head_size_vo), dtype=torch.float16).npu()
q_rope = torch.randn((num_tokens, num_heads, head_size_qk - head_size_vo), dtype=torch.float16).npu()
ctkv = torch.randn((num_blocks, block_size, kv_heads, 512), dtype=torch.float16).npu()
k_rope = torch.randn((num_blocks , block_size, kv_heads, 64), dtype=torch.float16).npu()
block_tables = torch.randint(0, 10, (batch, max_num_blocks_per_query), dtype=torch.int32).npu()
contextLens = [10] * batch

attenOut = torch_npu.atb.npu_multi_head_latent_attention(q_node, q_rope, ctkv, k_rope, block_tables, contextLens,32,1.0, 1)
    """
    )

    _add_torch_npu_atb_docstr(
        "npu_self_attention_prefix_encoder",
        """
torch_npu.atb.npu_self_attention_prefix_encoder(query, key, value, block_tables, seqlen, kv_seqlen, q_headnum, qk_scale, kv_headnum, *, mask=None, slopes=None, mask_type=None, output=None) -> Tensor
功能描述
MLA场景，使用分页管理的kvcache计算attention score，额外支持分离qnope/qrope、ctkv/krope的输入。

参数说明
位置参数：
query: Device Tensor, query矩阵, 支持float16/bf16， shape为[batch*qSeqLen, qHiddenSize]和[batch*qSeqLen, headNum, headSize]
key: Device Tensor, key矩阵, 支持float16/bf16， shape为[numBlocks, blockSize, kvHiddenSize]和[numBlocks, blockSize, headNum, headSize]
value: Device Tensor, 无位置编码ctkv, 支持float16/bf16数据类型，shape为[numBlocks, blockSize, kvHiddenSize]和[numBlocks, blockSize, headNum, headSize]	
block_tables: Device Tensor, 必选，每个query的kvcache的block table，第一维是token索引，第二维表示block索引
seqlen：int类型数组，query对应的每个batch的序列长度
kv_seqlen：int类型，key, value对应的每个batch的序列长度
q_headnum: int类型，query头数量。
qk_scale: float类型，Q*K^T后乘以的缩放系数
kv_headnum：int类型，kv头数量。
关键字参数：
mask: 可选Device Tensor,  默认场景下不需要输入
slopes: 可选Device Tensor,  默认场景下不需要输入
mask_type：可选字符串，设置mask类型，缺省值为'mask_type_causal_mask', 当前支持'mask_type_causal_mask'.
output: 可选Device Tensor，attention输出，shape为[batch*qSeqLen, qHiddenSize]或[batch*seqLen, headNum, headSize]，数据类型为float16/bf16。

约束：
qseqlen <= kv_seqlen
(kv_seqlen - q_seqlen) % 128 == 0

参考样例：
import torch,torch_npu
batch = 4
qseqlen = 128
headnum = 28
headsize = 128
numblocks = 64
block_size = 128
q_seqlens = [32] * batch
kv_seqLen = [32] * batch
dtype = torch.float16

query = torch.randn((batch*qseqlen, headnum, headsize), dtype=dtype).npu()
key = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
value = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
block_tables = torch.randint(0, 10, (batch, 64), dtype=torch.int32).npu()

output = torch_npu.atb.npu_self_attention_prefix_encoder(query, key, value, block_tables, q_seqlens, kv_seqLen, 28, 0.0883,4)
    """
    )
    _add_torch_npu_atb_docstr(
        "npu_mla_preprocess",
        """
torch_npu.atb.npu_mla_preprocess(input, gamma0, beta0, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2, cos, sin, wuk, kv_cache, kv_cache_rope,
                                slotmapping, *,quant_scale0=None, quant_offset0=None, bias0=None, quant_scale1=None, quant_offset1=None, bias1=None,
                                ctkv_scale=None, q_nope_scale=None, cache_mode=None, quant_mode=None) -> (Tensor q_out0, Tensor kv_cache_out0, Tensor q_out1, Tensor kv_cache_out1)
功能描述
融合了MLA场景下PagedAttention输入数据处理的全过程，包括从隐状态输入开始经过rmsnorm、反量化、matmul、rope、reshapeAndCache的一系列计算。
默认场景：cache_mode为krope_ctkv，quant_mode为per_token_quant_symm

参数说明：下属参数说明基于融合的各个小算子输入进行说明。
位置参数：
input: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[tokenNum, 7168]
gamma0: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[7168], 数据类型与input一致
beta0: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[7168], 数据类型与input一致
wdqkv: Device Tensor, matmul_0算子输入, 支持int8，shape为[2112,7168]
descale0: Device Tensor, matmul_0算子输入, 支持int8，shape为[2112]
gamma1: Device Tensor, rmsNormQuant_1算子输入, 支持float16/bf16，shape为[1536], 数据类型与input一致
beta1: Device Tensor, rmsNormQuant_1算子输入, 支持float16/bf16，shape为[1536], 数据类型与input一致
wuq: Device Tensor, matmul_1算子输入, 支持int8，shape为[headNum * 192, 1536]
descale1: Device Tensor, matmul_1算子输入, input为float16时为int64，input为bf16时为float，shape为[headNum * 192]
gamma2: Device Tensor, rmsNorm算子输入, 支持float16/bf16，shape为[512], 数据类型与input一致
cos: Device Tensor, rope算子输入, 支持float16/bf16，shape为[tokenNum,64], 数据类型与input一致
sin: Device Tensor, rope算子输入, 支持float16/bf16，shape为[tokenNum,64], 数据类型与input一致
wuk: Device Tensor, matmulEin算子输入, 支持float16/bf16, 数据类型与input一致, ND格式时shape为[headNum,128,512], NZ格式时shape为[headNum,32,128,16]
kv_cache:  Device Tensor, reshapeAndCache算子输入, 支持float16/bf16/int8, cache_mode为'krope_ctkv'时，数据类型与input一致, shape为[blockNum,blockSize,1,512]， cache_mode为'int8_nzcache'时，需为NZ格式， int8数据类型，shape为[blockNum, headNum*512/32,block_size, 32]
kv_cache_rope: Device Tensor, reshapeAndCache算子输入, 支持float16/bf16 ，数据类型与input一致cache_mode为'krope_ctkv'时, shape为[blockNum,blockSize,1,64]， cache_mode为'int8_nzcache'时，需为NZ格式，shape为[blockNum, headNum*64 / 16 ,block_size, 16]
slotmapping: Device Tensor, reshapeAndCache算子输入, 支持int32, shape为[tokenNum]
关键字参数：
quant_scale0: 可选Device Tensor, rmsNormQuant_0算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持float16/bf16，shape为[1], 数据类型与input一致
quant_offset0: 可选Device Tensor, rmsNormQuant_0算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int8，shape为[1]
bias0: 可选Device Tensor, matmul_0算子输入, 支持int8，shape为[2112]
quant_scale1: 可选Device Tensor, rmsNormQuant_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持float16/bf16，shape为[1], 数据类型与input一致
quant_offset1: 可选Device Tensor, rmsNormQuant_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int8，shape为[1]
bias1: 可选Device Tensor, matmul_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int32，shape为[headNum * 192]
ctkv_scale: 可选Device Tensor,  默认场景下不需要输入, quant算子输入, shape为[1], 数据类型与input一致支持float16/bf16, cache_mode为'int8_nzcache'时,传入此tensor。
q_nope_scale: 可选Device Tensor,  默认场景下不需要输入, quant算子输入, shape为[headNum], 数据类型与input一致支持float16/bf16, cache_mode为'int8_nzcache'时,传入此tensor。
cache_mode: 可选字符串，输入query和kcache的类型, 缺省值为'krope_ctkv', 当前支持'krope_ctkv'（输入输出的kvCcache拆分为krope和ctkv，q拆分为qrope和qnope）, 'int8_nzcache'（krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型）。
quant_mode: 可选字符串，表示RmsNorm量化类型, 缺省值为'per_token_quant_symm'（per_token动态对称量化）, 当前支持'per_token_quant_symm', 'per_tensor_quant_asymm'（per_tensor静态非对称量化）。

约束：
tokenNum＜=1024。
blockSize <= 128 或 blockSize = 256。
cache_mode为int8_nzcache时，blockSize = 128。

参考样例（cache_mode为int8_nzcache，quant_mode为per_tensor_quant_asymm场景）：
import torch,torch_npu
token_num = 8
head_num = 128
N_7168 = 7168
block_num = 192
block_size = 128
dtype = torch.float16
device = 'npu'

input1 = torch.randn((token_num, N_7168), dtype=dtype, device=device)

gamma0 = torch.randn((N_7168), dtype=dtype, device=device)
beta0 = torch.randn((N_7168), dtype=dtype, device=device)
quant_scale0 = torch.randn((1,), dtype=dtype, device=device)
quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device=device)
wdqkv = torch_npu.npu_format_cast(wdqkv, 29)
de_scale0 = torch.randint(0, 7, (2112, ), dtype=torch.int64, device=device)
bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32, device=device)

gamma1 = torch.randn((1536), dtype=dtype, device=device)
beta1 = torch.randn((1536), dtype=dtype, device=device)
quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device=device)
wuq = torch_npu.npu_format_cast(wuq, 29)
de_scale1 = torch.randint(0, 7, (head_num * 192, ), dtype=torch.int64, device=device)
bias1 = torch.randint(0, 7, (head_num * 192, ), dtype=torch.int32, device=device)

gamma2 = torch.randn((512), dtype=dtype, device=device)

cos = torch.randn((token_num, 64), dtype=dtype, device=device)
sin = torch.randn((token_num, 64), dtype=dtype, device=device)

wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)
wuk = torch_npu.npu_format_cast(wuk, 29)

kv_cache = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
kv_cache = torch_npu.npu_format_cast(kv_cache, 29)
kv_cache_rope = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
kv_cache_rope = torch_npu.npu_format_cast(kv_cache_rope, 29)

slotmapping = torch.randint(0, 7, (token_num,), dtype=torch.int32, device=device)

ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
qnope_scale = torch.randn((head_num), dtype=dtype, device=device)

q_out0 = torch.randint(0, 7, (token_num, head_num, 512), dtype=torch.int8, device=device)
kv_cache_out0 = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
kv_cache_out0 = torch_npu.npu_format_cast(kv_cache_out0, 29)
q_out1 = torch.randn((token_num, head_num, 64), dtype=dtype, device=device)
kv_cache_out1 = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
kv_cache_out1 = torch_npu.npu_format_cast(kv_cache_out1, 29)

torch_npu.atb.npu_mla_preprocess(
    input1, gamma0, beta0, wdqkv, de_scale0,
    gamma1, beta1, wuq, de_scale1,
    gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping,
    quant_scale0=quant_scale0,
    quant_offset0=quant_offset0,
    bias0=bias0,
    quant_scale1=quant_scale0,
    quant_offset1=quant_offset1,
    bias1=bias1,
    ctkv_scale=ctkv_scale,
    q_nope_scale=qnope_scale,
    cache_mode="int8_nzcache",
    quant_mode="per_tensor_quant_asymm",
    q_out0=q_out0,
    kv_cache_out0=kv_cache_out0,
    q_out1=q_out1,
    kv_cache_out1=kv_cache_out1,
)
    """
    )
    _add_torch_npu_atb_docstr(
        "_npu_paged_attention_v2",
        """
torch_npu.atb._npu_paged_attention_v2(Tensor query, Tensor key_cache, Tensor block_table, SymInt[] context_lens, *, Tensor? value_cache=None, Tensor? mask=None, int num_kv_heads=0, int num_heads=0, float scale_value=1.0, int mask_type=0, Tensor(a!) out) -> Tensor(a!)
功能描述：
基于paged attention 对于kvcache分页管理的机制计算attention score。支持基础场景、alibi场景。
基础场景(默认场景): mask_type=0 (mask_type=UNDEFINED)。
alibi场景: mask_type=2 (mask_type=MASK_TYPE_ALIBI)。

参数说明：
位置参数：
query: Device Tensor, 各batch的query在num_tokens轴合并, 支持float16/bf16/int8,支持ND格式, shape为[num_tokens, num_heads, head_size]。
key_cache: Device Tensor, cache好的key,支持float16/bf16/int8。
           Atlas 800I A2 推理产品/Atlas A2 训练系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品：[num_blocks, block_size, kv_head_num, head_size_k],支持ND格式。
           Atlas 推理系列产品：[num_blocks, head_size *num_heads / 16, block_size, 16],支持NZ格式。
block_table: Device Tensor, 每个query的kvcache的block table,第一维是token索引,第二维表示block索引。 支持int32,shape为[num_tokens, max_num_blocks_per_query],支持ND格式。
context_lens: int类型数组, 每个query对应的key/value的token数量(每个query对应的上下文长度)。 支持int32,shape为[batch],支持ND格式。

关键字参数：
value_cache: 可选Device Tensor, cache好的value, 支持float16/bf16/int8。
             Atlas 800I A2 推理产品/Atlas A2 训练系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品：[num_blocks, block_size, kv_head_num, head_size_v]，支持ND格式。
             Atlas 推理系列产品：[num_blocks, head_size *num_heads / 16 ,block_size, 16]，支持NZ格式。
mask: 可选Device Tensor, 支持不传mask(mask_type=UNDEFINED)、支持传入alibi mask (mask_type=MASK_TYPE_ALIBI)。
      Atlas 800I A2 推理产品/Atlas A2 训练系列产品、Atlas A3 推理系列产品/Atlas A3 训练系列产品支持的维度为[batch, num_head, 1, max_seq_len] 或 [num_head, 1, max_seq_len]，支持ND格式。
      Atlas 推理系列产品支持的维度为[batch * num_head, max_seq_len / 16, 16, 16] 或 [num_head, max_seq_len / 16, 16, 16],支持NZ格式。
num_kv_heads: int类型，kv头数量。
num_heads: int类型，query头数量。此默认值不可用，用户需配置此项参数。
scale_value: float类型，算子tor值。
mask_type: int类型枚举, mask类型。当前接口支持两种类型：mask_type=0 (mask_type=UNDEFINED)、mask_type=2 (mask_type=MASK_TYPE_ALIBI)。缺省值为mask_type=0。
out: Device Tensor，经过计算输出的query。数据类型为float16/bf16，shape为[num_tokens, num_head, head_size_v]。

规格约束
因硬件限制，block_size %16 == 0，推荐 block_size = 128。
因硬件限制，block_size <= 128。
block_tables中元素的值须在[0, num_blocks)之间。
head_size范围为（0, 256]，key_cache和value_cache的格式为NZ时，head_size必须是16倍数。
在Atlas 推理系列产品上 ：query、key_cache、value_cache、mask 四个入参只支持float16。0<batch<=2048。
多头自适应压缩、并行解码场景、量化场景、以及Atlas 推理系列产品上，key_cache，value_cache的head_size等长，范围为（0, 256]，且block_size * head_size ≤ 128 * 128，否则key_cache，value_cache的head_size可以不相同，范围为（0, 576]，当key_cache或value_cache的head_size > 256时，block_size小于等于128。
针对mask类型，如果原模型非alibi，且mask没有做padding，可选择“mask_type”为UNDEFINED，即不传mask；其他情况则依据原生模型的配置，选择“mask_type”是MASK_TYPE_NORM或MASK_TYPE_ALIBI。
BNSD不支持PA_ENCODER。
Atlas 训练系列产品与Atlas 推理系列产品的输入输出约束相同，Atlas 训练系列产品只支持“mask_type”为UNDEFINED、MASK_TYPE_NORM或MASK_TYPE_ALIBI的基础功能场景，其余特性参数需为默认值。
若想使用GQA模式，需满足head_num > kv_head_num且head_num%kvhead_num == 0。
当block_size = head_size = 256时，在Atlas 800I A2 推理产品/Atlas A2 训练系列产品上，输入格式需满足BSND、非量化、decoder场景。

参考样例：
import torch,torch_npu

num_tokens = 1
num_heads = 32
num_kv_heads = 32
block_size = 256
head_size = 256
head_size_k = 256
head_size_v = 38
num_blocks = 64
k_seqlen = 128
scale_value = 1.0 / (head_size ** 0.5)
max_num_blocks_per_query = 2
mask_type=0

query = torch.randn((num_tokens, num_heads, head_size), dtype=torch.float16).npu()
key_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_size_k), dtype=torch.float16).npu()
block_table = torch.randint(0, 10, (num_tokens, max_num_blocks_per_query), dtype=torch.int32).npu()
context_lens = [1024] * num_tokens
mask = None
value_cache = torch.rand((num_blocks, block_size, num_kv_heads, head_size_v), dtype=torch.float16).npu()
out = torch.rand((num_tokens, num_heads, head_size_v), dtype=torch.float16).npu()

torch_npu.atb._npu_paged_attention_v2(query, key_cache, block_table, context_lens, value_cache=value_cache, mask=mask, num_kv_heads=num_kv_heads, num_heads=num_heads, scale_value=scale_value, mask_type=0, out=out)
    """
    )
    _add_torch_npu_atb_docstr(
        "_npu_flash_attention_v2",
        """
torch_npu.atb._npu_flash_attention_v2(Tensor query, Tensor key, Tensor value, SymInt[] seq_len, *, Tensor? mask=None, Tensor? slopes=None, int kernel_type=0, int mask_type=2, float scale_value=1, int num_heads=0, int num_kv_heads=0, Tensor(a!) out) -> Tensor(a!)
功能描述：
基于传统flash attention按照layerId管理kvcache的机制计算attention score。
支持基础场景、倒三角mask、alibi mask、alibi压缩mask、alibi压缩开平方mask、alibi压缩mask左对齐(只支持Atlas 800I A2推理产品)。
支持mask传mask_type=0(mask_type=MASK_TYPE_UNDEFINED)、mask_type=1(mask_type=MASK_TYPE_NORM)、mask_type=2(mask_type=MASK_TYPE_ALIBI)、mask_type=4(mask_type=MASK_TYPE_ALIBI_COMPRESS)、mask_type=5(mask_type=MASK_TYPE_ALIBI_COMPRESS_SQRT)、mask_type=6(mask_type=MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN)。
当前默认场景为alibi mask。此接口默认calc_type为PA_ENCODER。

参数说明：
位置参数：
query: Device Tensor, query矩阵, 支持float16/bf16,支持ND格式, shape为[num_tokens, num_heads, head_size]。
key: Device Tensor, key矩阵, 支持float16/bf16,支持ND格式, shape为[num_tokens, num_heads, head_size]。
value：Device Tensor, key矩阵, 支持float16/bf16,支持ND格式, shape为[num_tokens, num_heads, head_size]。
seq_len：int类型数组, 序列长度, 支持ND格式, 等于1时，为增量或全量；大于1时，为全量。

关键字参数：
mask: 可选Device Tensor。
1.所有batch相同,方阵。shape为[max_seq_len, max_seq_len]。
2.batch不同时的方阵。shape为[batch, max_seq_len, max_seq_len]。
3.alibi场景。shape为[batch, head_num, max_seq_len, max_seq_len]。、
4.alibi压缩mask只有“calcType”置为PA_ENCODER和PREFIX_ENCODER时生效。
“mask_type”为MASK_TYPE_ALIBI_COMPRESS或MASK_TYPE_ALIBI_COMPRESS_SQRT时，mask的维度：
    在 Atlas 800I A2 推理产品 / Atlas A2 训练系列产品 和 Atlas A3 推理系列产品 / Atlas A3 训练系列产品 上为[head_num, seq_len, 128]或[256, 256]。
    在 Atlas 推理系列产品 上为[head_num,128//16,max_seq_len,16]或[1,256//16,256,16]。
“mask_type”为MASK_TYPE_NORM_COMPRESS时，mask的维度：
    在 Atlas 800I A2 推理产品 / Atlas A2 训练系列产品 和 Atlas A3 推理系列产品 / Atlas A3 训练系列产品 上为[128,128]。
    在 Atlas 推理系列产品 上为[1,128//16,128,16]。
使用alibi压缩mask时，qkv的headSize要小于等于128。
MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN仅支持 Atlas 800I A2 推理产品 / Atlas A2 训练系列产品 和 Atlas A3 推理系列产品 / Atlas A3 训练系列产品 。此时mask的维度为[256, 256]。
Atlas 800I A2 推理产品 / Atlas A2 训练系列产品 和 Atlas A3 推理系列产品 / Atlas A3 训练系列产品上，当数据类型为float16时，alibi压缩mask只有开启高精度才有效。
slopes：可选Device Tensor, ALiBi位置编码，为alibi mask每个head的系数,支持ND格式,shape为[batch]。
kernel_type： int类型，内核精度类型。0为KERNELTYPE_DEFAULT，1为高精度KERNELTYPE_HIGH_PRECISION。
mask_type：int类型，支持传mask_type=0(mask_type=MASK_TYPE_UNDEFINED)、mask_type=1(mask_type=MASK_TYPE_NORM)、mask_type=2(mask_type=MASK_TYPE_ALIBI)、mask_type=4(mask_type=MASK_TYPE_ALIBI_COMPRESS)、mask_type=5(mask_type=MASK_TYPE_ALIBI_COMPRESS_SQRT)、mask_type=6(mask_type=MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN)。
scale_value：float类型，算子tor值, 在Q*K^T后乘。
num_heads：int类型，query头数量， 需大于0
num_kv_heads：int类型，kv头数量。
out: Device Tensor，经过计算输出的query。数据类型为float16/bf16，shape为[num_tokens, num_head, head_size]。

规格约束:
1.输入数据排布格式为TYPE_BSND时：
    在PA_ENCODER下，Atlas 800I A2 推理产品/Atlas A2 训练系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品上的query，key，value可传二维[num_tokens, head_size]或三维[num_tokens, num_head, head_size]。
2.Atlas 推理系列产品上 ：
    0<batch<=2000。
    mask大小必须使用真实的max_seq_len。
    q_seqLen必须等于kv_seqLen。
3.当在以下场景时，head_size = v_head_size，范围为（0，256]，其他情况head_size可以不等于v_head_size，二者的范围为（0，576]。
    在Atlas 推理系列产品上运行，此时headSize必须为16的倍数。
    在Atlas 训练系列产品上运行，此时headSize必须为16的倍数。
4.关于tensor维度中的nTokens：
    Atlas 800I A2 推理产品/Atlas A2 训练系列产品和Atlas A3 推理系列产品/Atlas A3 训练系列产品上为各batch上seq_len之和。
    Atlas 推理系列产品上：PA_ENCODER下为所有batch的seq_len之和向上对齐到16的整数倍。
    其余情况下为所有batch上的seq_len先向上对齐到16的整数倍，再求和。
5.关于选取何种mask：
    在全量阶段，依据原生模型的配置，选择“maskType”是MASK_TYPE_NORM或MASK_TYPE_ALIBI。
    在增量阶段，如果原模型非alibi，且mask没有做padding，可选择“maskType”为UNDEFINED，即不传mask；其他情况则依据原生模型的配置，选择“maskType”是MASK_TYPE_NORM或MASK_TYPE_ALIBI。
    在长序列场景下，可以考虑压缩mask，压缩mask的构造方法详见压缩mask。
6.在PA_ENCODER下：
    q_scale不生效。
    query、key、value的维度数要相同。
7.若想使用GQA模式，需满足head_num> kv_head_num且headNum%kv_head_num== 0。
8.Atlas 训练系列产品只支持PA_ENCODER，仅支持基础功能。
9.在calc_type为ENCODER，DECODER且非bypass场景下hiddensize必须为16的倍数。
10.query、key、value的headSize为非64的倍数时，会导致算子性能下降，为了确保最佳性能，建议将query、key、value的headSize设置为64的倍数。
11.cpu侧的tensor需要利用hostData设置数据的地址，无需设置deviceData；npu侧的tensor需要利用deviceData设置数据的地址，无需设置hostData。特殊情况：ENCODER/DECODER在非kv-bypass场景下tokenOffset和seq_len这两个tensor如果都同时传了deviceData和hostData，则会采用deviceData中的数据进行kvcache部分的计算。

参考样例：
import torch,torch_npu
import math

kernel_type = 0
mask_type = 4
scale_value = 1.0 / math.sqrt(1.0 * 128)
num_heads = 8
num_kv_heads = 1
num_tokens = 8192
head_size = 128

query = torch.rand((num_tokens, num_heads, head_size), dtype=torch.float16).npu()
key = torch.rand((num_tokens, num_kv_heads, head_size), dtype=torch.float16).npu()
value = torch.rand((num_tokens, num_kv_heads, head_size), dtype=torch.float16).npu()
seq_len = [1024] * batch
mask = torch.rand((256, 256), dtype=torch.float16).npu()
slopes = torch.rand(batch, dtype=torch.float16).npu()
out = torch.rand((num_tokens, num_heads, head_size), dtype=torch.float16).npu()
torch_npu.atb._npu_flash_attention_v2(query, key, value, seq_len, mask=mask, slopes=slopes, kernel_type=kernel_type, mask_type=mask_type, scale_value=scale_value, num_heads=num_heads, num_kv_heads=num_kv_heads, out=out)
    """
    )
    _add_torch_npu_atb_docstr(
        "_npu_flash_attention_prefix_v2",
        """
torch_npu.atb._npu_flash_attention_prefix_v2(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor mask, Tensor seq_len, Tensor context_lens, *, Tensor? slopes=None, int kernel_type=1, int mask_type=3, int num_kv_heads=0, int num_heads=0, float scale_value=1, Tensor(a!) out) -> Tensor(a!)
功能描述：
基于传统flash attention按照layerId管理kvcache的机制计算attention score。支持alibi场景+prefix场景。
FA支持按block存放的kvCache。BlockTables维度为[batch, maxBlockNum]，其中每个batch对应存放当前query对应的key和value的index，maxBlockNum约为key，value中最长kvSeqLen，kvSeqLen/blockSize向上取整。
此接口默认calc_type为PREFIX_ENCODER。
支持mask_type=3(mask_type=MASK_TYPE_NORM_COMPRESS)、mask_type=4(mask_type=MASK_TYPE_ALIBI_COMPRESS)、mask_type=5(mask_type=MASK_TYPE_ALIBI_COMPRESS_SQRT)。

参数说明：
位置参数：
query: Device Tensor, query矩阵。query的shape和key、value的不需要一致。PrefixCacheLen的计算由kvseqlen减seqLen得到，由于实现方案限制，需要让prefixCacheLen 128对齐。headSize需要16对齐。
       支持float16/bf16,支持ND格式, shape为[batch*qSeqLen, qHiddenSize] 或[batch*qSeqLen, headNum, headSize].
key_cache: Device Tensor, cache好的key,支持float16/bf16,支持ND格式。支持shape为[numBlocks, blockSize, kvHiddenSize] 或[numBlocks, blockSize, headNum, headSize]。key，value的HeadSize <= 128。blockSize必须设为128。BlockSize × HeadSize <= 128 × 128。
value_cache：Device Tensor, value矩阵。key，value的shape必须一致。支持float16/bf16,支持ND格式。支持shape为[numBlocks, blockSize, kvHiddenSize] 或[numBlocks, blockSize, headNum, headSize]。
block_table: Device Tensor, 每个query的kvcache的block table,第一维是token索引,第二维表示block索引。 支持int32,shape为[batch, maxBlockNum],支持ND格式。blocktable 中item的value < maxBlockNum、max(kvSeqLen) / blockSize <= maxblockNum
mask: Device Tensor,仅支持压缩mask。支持float16/bf16,支持ND格式。MASK_TYPE_ALIBI_COMPRESS支持维度[256, 256]和[head, seqLen, 128]。MASK_TYPE_ALIBI_COMPRESS_SQRT支持维度[256, 256]和[head, seqLen, 128]。MASK_TYPE_NORM_COMPRESS支持维度[128, 128]。
seq_len: int类型数组, query对应的每个batch的序列长度, 支持ND格式。shape为[batch]。
context_lens: int类型数组, key, value对应的每个batch的序列长度。支持int32,shape为[batch],支持ND格式。

关键字参数：
slopes:可选Device Tensor，alibi coefficient，当使用alibiMask时传入tensor时，为alibiMask的每个head的系数。支持float32，支持ND格式，shape为[head]。
kernel_type: int类型，内核精度类型。当前场景仅支持1，为高精度KERNELTYPE_HIGH_PRECISION。
mask_type:int类型，mask_type=3(mask_type=MASK_TYPE_NORM_COMPRESS)、mask_type=4(mask_type=MASK_TYPE_ALIBI_COMPRESS)、mask_type=5(mask_type=MASK_TYPE_ALIBI_COMPRESS_SQRT)
num_kv_heads: int类型，kv头数量。
num_heads: int类型，query头数量。此默认值不可用，用户需配置此项参数。
scale_value: float类型，算子tor值。
out: Device Tensor，经过计算输出的query。支持ND格式，数据类型为float16/bf16，shape为[batch*qSeqLen, qHiddenSize] 或[batch*qSeqLen, headNum, headSize]。

规格约束:
不支持qScale，需设为1。
isTriuMask需设置为1。
kernelType必须设为KERNELTYPE_HIGH_PRECISION，因为kv计算内部均使用float32计算。
不支持clamp，clampMin、clampMax需设置为0。
inputLayout只支持TYPE_BSND。
不支持mla，mlaVHeadSize需设为0。
PrefixCacheLen的计算由kvseqlen减seqLen得到，由于实现方案限制，需要让prefixCacheLen 128对齐。
headNum有如下约束：
    headNum > 0
    headNum >= kvHeadNum
    headNum是kvHeadNum的整数倍
maskType支持MASK_TYPE_ALIBI_COMPRESS、MASK_TYPE_ALIBI_COMPRESS_SQRT或MASK_TYPE_NORM_COMPRESS。

参考样例：
import torch,torch_npu

batch = 1
kv_head = 1
headNum = 12
headSize = 128
scale_value = 1.0 / math.sqrt(1.0 * headSize)
blockSize = 128
numBlocks = 1024
maxBlockNum = 4

query = torch.randn((batch*q_seqlens, headNum, headSize), dtype=torch.float16).npu()
key_cache = torch.randn((numBlocks, blockSize, kvHiddenSize), dtype=torch.float16).npu()
value_cache = torch.rand((numBlocks, blockSize, kvHiddenSize), dtype=torch.float16).npu()
block_table = torch.randint(0, 10, (batch, maxBlockNum), dtype=torch.int32).npu()
mask = torch.rand((256, 256), dtype=torch.float16).npu()
seq_len = [128]
context_lens = [512]
slopes = torch.rand((headSize), dtype=torch.float16).npu()
out = torch.rand((batch*qSeqLen, headNum, headSize), dtype=torch.float16).npu()

torch_npu.atb._npu_flash_attention_prefix_v2(query, key_cache, value_cache, block_table,
                                             mask, seq_len, context_lens, 
                                             slopes=slopes.to(torch.float32), kernel_type=1, mask_type=5,
                                             num_kv_heads=kv_head, num_heads=headNum, scale_value=scale_value, out=out)
    """
    )
    _add_torch_npu_atb_docstr(
        "npu_fused_add_topk_div",
        """
torch_npu.atb.npu_fused_add_topk_div(x, add_num, *, mapping_num=None, mapping_table=None, activation_type=None, group_num=1, group_topk=1, n=1, k=1, is_norm=True, scale=1, enable_expert_mapping=False, y=None, indices=None) -> (Tensor, Tensor)
功能描述：
AtbFusedAddTopkDiv主要应用在deepseek 的MOE场景中，融合了路由权重分配，专家选择等场景。主要是Sigmoid，Add，GroupTopk，Gather，ReduceSum，RealDiv，Muls算子的融合，相比小算子性能有1倍提升.
算子支持两种模式：常规模式（物理专家模式）、逻辑专家模式。
一开始，每个专家经过sigmoid激活函数、Add偏置后，得出[bs, expert_num]的专家评分。接下来每组经过DeepSeekV3的组级TOPK，专家级TOPK选出k个专家。在常规模式下，这些专家作为物理专家的ID，会直接输出，用于指导后续的通信算子将每个token发往k个专家对应的卡上。
然而在存在冗余专家场景下，每个物理专家会属于不同的卡，此时衍生出逻辑专家的概念，即每个卡上的不同专家为逻辑专家，一个物理专家映射到多个逻辑专家，逻辑专家ID/每个卡的专家数=对应卡的RankID，逻辑专家ID%每个卡的专家数=在卡上的位置。
在逻辑专家模式下，新增mappingNum指示每个物理专家被映射到的逻辑专家数，新增mappingTable指示每个物理专家被映射到的逻辑专家ID，算子内使用伪随机逻辑，随机选择其中一个逻辑专家进行映射，并更正输出indices为逻辑专家ID。

参数说明：
位置参数：
x: Device Tensor, 输入数据，支持ND格式，数据类型为float16/float32/bf16，shape为[a, b]。
add_num: Device Tensor, 用于与x相加。数据类型和格式与x一致。支持ND格式，数据类型为float16/float32/bf16，shape为[b]。

关键字参数：
mapping_num： 可选Device Tensor，enable_expert_mapping为false时不启用，true时输入tensor，每个物理专家被实际映射到的逻辑专家数量。支持ND格式，数据类型为int32，shape为[b]。
mapping_table： 可选Device Tensor，enable_expert_mapping为false时不启用，true时输入tensor，物理专家/逻辑专家映射表。支持ND格式，数据类型为int32，shape为[b,c] c<=128。
activation_type：可选字符串,激活类型。当前只支持传入activation_type=activation_sigmoid。
group_num：int类型，分组数量。
group_topk：int类型，选择k个组。
n: int类型，组内选取n个最大值求和。
k: int类型，topk选取前k个值。
is_norm: bool类型，是否归一化。
scale: float类型，归一化后的乘系数。
enable_expert_mapping: bool类型，是否使能物理专家向逻辑专家的映射。
y: 可选Device Tensor，输出维度为[a, k]，支持ND格式，数据类型为float32。
indices: 可选Device Tensor，输出维度为[a, k]，支持ND格式，数据类型为int32。

规格约束:
1. b为group_num的整数倍。
2. groupTopk <= group_num。
3. k <= b。
4. b >= group_num * n。
5. b <= group_num * 32。
6. 若b >= 32，则group_num = 8。
7. max_redundant_expert_num代表最大可能出现的额冗余专家数，目前最大支持128。
8. mapping_num里面的单值: 0<=元素值<c
9. 含义：a=bs，b=expert_num，c= max_redundant_expert_num。

硬件支持情况:
Atlas 800I A2 推理产品/Atlas A2 训练系列产品:支持
Atlas A3 训练系列产品:支持
Atlas 200I/500 A2 推理产品:不支持
Atlas 推理系列产品:不支持
Atlas 训练系列产品:不支持

参考样例：
import torch,torch_npu
a = 16
b = 256
c = 64
k = 8
x = torch.randn(a, b, dtype=torch.float16).npu()
add_num = torch.randn(b,dtype=torch.float16).npu()
mapping_num = torch.randint(1, c+1, (b,), dtype=torch.int32).npu()
mapping_table = torch.randint(0, 10, (b, c), dtype=torch.int32).npu()
y = torch.empty(a, k, dtype=torch.float32).npu()
indices = torch.empty(a, k, dtype=torch.int32).npu()
torch_npu.atb.npu_fused_add_topk_div(x, add_num, mapping_num=mapping_num, mapping_table=mapping_table, activation_type='activation_sigmoid', group_num=8, group_topk=4, n=2, k=k, is_norm=True, scale=1, enable_expert_mapping=True, y=y, indices=indices)
    """
    )
    _add_torch_npu_atb_docstr(
        "npu_ring_mla",
        """
torch_npu.atb.npu_ring_mla(q_nope, q_rope, k_nope, k_rope, value, mask, seqlen, int head_num, int kv_head_num, *, pre_out=None, prev_lse=None, float qk_scale=1, str? kernel_type=None, str? mask_type=None, str? input_layout=None, str? calc_type=None, out=None) -> (Tensor, Tensor)
功能描述：
在长序列场景中，计算QK会产生O(seq_len)的显存开销，为了支持长序列的计算，ringAttention 提出了新的计算方式，将QKV沿序列长度切分，分块计算，合并分块计算结果时需要使用前一块计算的中间结果（prev_out，prev_lse）。- chunked prefill 在prefill阶段，将长的prefill分为多个小的chunk，顺序计算并合并结果。第一个chunk时，ringMLA支持calcType=CALC_TYPE_FISRT_RING，不接收lse，输出lse。非第一个chunk时，ringMLA支持calcType=CALC_TYPE_DEFAULT，接收并合并上一次计算的中间结果lse。
首卡场景
开启方式：calc_type = calc_type_first_ring
区别：无prevLse，prevOut传入，生成softmaxLse输出
非首末卡场景
开启方式：calc_type = calc_type_default
区别：有prevLse，prevOut传入，生成softmaxLse输出

参数说明：
位置参数：
q_nope: Device Tensor,无位置编码query矩阵，支持float16/bf16,支持ND格式,shape为[qNTokens, headNum, 128]。
q_rope: Device Tensor,query旋转位置编码分量，支持float16/bf16,支持ND格式,shape为[qNTokens, headNum, 64]。
k_nope: Device Tensor,无位置编码key矩阵，支持float16/bf16,支持ND格式,shape为[kvNTokens, kvHeadNum, 128]。
k_rope: Device Tensor,key旋转位置编码，支持float16/bf16,支持ND格式,shape为[kvNTokens, kvHeadNum, 64]。
value: Device Tensor,value矩阵，支持float16/bf16,支持ND格式,shape为[kvNTokens, kvHeadNum, 128]。
mask: Device Tensor,掩码，支持float16/bf16,支持ND格式,shape为[512, 512]。
seqlen: Device Tensor,序列长度,支持ND格式。shape为[batch]/[2, batch]。
    若shape为[batch] (，代表每个batch的序列长度，query，cacheK，cacheV相同。)
    若shape为[2，batch]，seqlen[0]代表query的序列长度，seqlen[1]代表cacheK，cacheV的序列长度。
head_num: int类型, kv头数量, 该值需要用户根据使用的模型实际情况传入。

关键字参数：
pre_out: 可选Device Tensor，模型lse输入场景中的前次输出，支持float16/bf16,支持ND格式,shape为[qNTokens, headNum, 128]。
prev_lse: 可选Device Tensor，模型lse输入场景中的前次QK^T * tor的结果，先取softmax，exp，sum，最后求log，支持float,支持ND格式,shape为[headNum, qNTokens]。
qk_scale: float类型，在Q*K^T后乘。
kernel_type: 可选字符串，算子内核精度类型，当前支持kernel_type_default、kernel_type_high_precision。
mask_type: 可选字符串， mask类型，当前支持no_mask、mask_type_triu。
input_layout: 可选字符串，当前支持type_bsnd。数据排布格式默认为BSND。
calc_type: 可选字符串，当前支持calc_type_default、calc_type_first_ring。
out: 可选Device Tensor，支持float16/bf16,支持ND格式,shape为[qNTokens, headNum, 128]。
softmax_lse：可选Device Tensor，支持float,支持ND格式,shape为[headNum, qNTokens]。

规格约束:
1.maskType = MASK_TYPE_TRIU时才使用mask。
2.inputLayout仅支持TYPE_BSND
3.对于每个batch，kvSeqLen >= qSeqLen或者kvSeqLen为0；qSeqLen不可为0。
qSeqLen为seqLen第一维度。
kvSeqLen为seqLen第二维度。

硬件支持情况:
Atlas 800I A2推理产品 支持
Atlas 800I A3推理产品 支持

参考样例：
import torch,torch_npu
import math

qn_tokens = 200
kvn_tokens = 200
head_num = 64
kv_head_num = 64
max_seq_len = 512
qkScale = 1/math.sqrt(128)

q_nope = torch.randn(qn_tokens, head_num, 128, dtype=torch.bfloat16).npu()
q_rope = torch.randn(qn_tokens, head_num, 64, dtype=torch.bfloat16).npu()
k_nope = torch.randn(kvn_tokens, kv_head_num, 128, dtype=torch.bfloat16).npu()
k_rope = torch.randn(kvn_tokens, kv_head_num, 64, dtype=torch.bfloat16).npu()
value = torch.randn(kvn_tokens, head_num, 128, dtype=torch.bfloat16).npu()
mask = torch.randn(max_seq_len, max_seq_len, dtype=torch.bfloat16).npu()
seqLen = torch.tensor([100, 100], dtype=torch.int32)
prevOut = torch.randn(qn_tokens, head_num, 128, dtype=torch.bfloat16).npu()
prevLse = torch.randn(head_num, qn_tokens, dtype=torch.float32).npu()
output = torch.empty(qn_tokens, head_num, 128, dtype=torch.bfloat16).npu()
softmaxLse = torch.empty(head_num, qn_tokens, dtype=torch.float32).npu()

torch_npu.atb.npu_ring_mla(
    q_nope, q_rope, k_nope, k_rope, value,
    mask, seqLen, head_num=head_num, kv_head_num=kv_head_num, 
    pre_out=prevOut, prev_lse=prevLse, qk_scale=qkScale,
    kernel_type='kernel_type_high_precision', mask_type='mask_type_triu', input_layout='type_bsnd', calc_type='calc_type_default',
    output=output, softmax_lse=softmaxLse)
    """
    )
    _add_torch_npu_atb_docstr(
        "npu_paged_cache_load",
        """
torch_npu.atb.npu_paged_cache_load(key_cache, value_cache, block_table, context_lens, *, seq_starts=None, cumsum=False, key=None, value=None) -> (Tensor, Tensor)
功能描述：
vllm在v1 engine上默认开启chunked prefill，deepseekV3/R1 chunked prefill依赖pagedcacheload算子功能：DeepSeek长序列场景，支撑模型prefix cache特性，优化显存。
根据blockTable中的blockId值、contextLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。
提供SeqStarts以为每个batch提供在blockTable中的初始位置（类似于offset）功能。
场景分支：chunked-prefill 场景与 prefix caching场景，ringMLA算子不支持读取cache，需要先调用pagedcacheload算子读取kvcache，乘以升维矩阵后再传给ringMLA算子。

参数说明
位置参数：
key_cache: Device Tensor, cache好的key, 支持float16/bf16/int8数据类型，shape为[num_blocks, block_size, num_heads, head_size_k]，支持ND格式，其中num_heads*head_size_k 需要是32B对齐。
value_cache: Device Tensor, cache好的value, 支持float16/bf16/int8数据类型，shape为[num_blocks, block_size, num_heads, head_size_v]，支持ND格式，其中num_heads*head_size_v 需要是32B对齐。
block_table: Device Tensor, 记录每个batch的sequence在kvcache中的blockId，支持int32数据类型,shape为[batch, block_indices]，支持ND格式。
context_lens: Device Tensor, 支持shape为[batch]/[batch+1]，每个batch序列长度或每个batch序列长度的累加和，支持int32数据类型,支持ND格式。

关键字参数：
seq_starts: 可选Device Tensor，每个batch在blocktable中对应的起始位置, shape为[batch],支持int32，支持ND格式。
cumsum: bool类型，表示传入的seqLens是否是每个batch序列长度的累加和，默认为False。
key: 可选Device Tensor，计算key输出，shape为[num_tokens, num_heads, head_size_k]，支持float16/bf16/int8数据类型，支持ND格式。
value: 可选Device Tensor，计算value输出，shape为[num_tokens, num_heads, head_size_k]，支持float16/bf16/int8数据类型，支持ND格式。

约束：
key_cache的shape信息中num_heads*head_size_k 需要是32B对齐。
value_cache的shape信息中num_heads*head_size_v 需要是32B对齐。


参考样例：
import torch,torch_npu
batch = 8
num_blocks = 1024
block_size = 16
num_heads = 1
head_size_k = 512
head_size_v = 64
block_indices = 1024
max_seq_len = 512
max_start = max_seq_len // 2
cumsum = False

key_cache = torch.randn(num_blocks, block_size, num_heads, head_size_k, dtype=torch.float16).npu()
value_cache = torch.randn(num_blocks, block_size, num_heads, head_size_v, dtype=torch.float16).npu()
block_table = torch.empty((batch, num_blocks), dtype=torch.int32).npu()
for b in range(batch):
    perm = torch.randperm(num_blocks).npu()
    block_table[b, :] = perm
context_lens = torch.randint(0, max_seq_len + 1, (batch,), dtype=torch.int32).npu()
seq_starts = torch.randint(0, max_start + 1, (batch,), dtype=torch.int32).npu()
output = torch_npu.atb.npu_paged_cache_load(key_cache, value_cache, block_table, context_lens, seq_starts=seq_starts, cumsum=cumsum)
    """
    )
