import types

import torch._C
from torch._C import _add_docstr as add_docstr

import torch_npu


def _add_torch_npu_docstr(method, docstr):
    """Add doc to operator API.
    If implementing the Python side interface with pybind11, _add_docstr is needed to add doc.
    """
    func = getattr(torch_npu, method, None)
    if not func:
        return
    # PT1.11/2.0 requires the use of _add_doc
    if isinstance(func, types.BuiltinMethodType):
        add_docstr(func, docstr)
    else:
        getattr(torch_npu, method).__doc__ = docstr


_add_torch_npu_docstr(
    "_npu_dropout",
    """
torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)

功能描述
不使用种子(seed)进行dropout结果计数。与torch.dropout相似，优化NPU设备实现。

参数说明
self (Tensor) - 输入张量。
p (Float) - 丢弃概率。
示例
>>> input = torch.tensor([1.,2.,3.,4.]).npu()
>>> input
tensor([1., 2., 3., 4.], device='npu:0')
>>> prob = 0.3
>>> output, mask = torch_npu._npu_dropout(input, prob)
>>> output
tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
>>> mask
tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
      253, 255], device='npu:0', dtype=torch.uint8)
"""
)


_add_torch_npu_docstr(
    "copy_memory_",
    """
torch_npu.copy_memory_(dst, src, non_blocking=False) -> Tensor
功能描述
从src拷贝元素到self张量，并返回self。

参数说明
dst (Tensor) - 拷贝源张量。
src (Tensor) - 返回张量所需数据类型。
non_blocking (Bool,默认值为False) - 如果设置为True且此拷贝位于CPU和NPU之间，则拷贝可能相对于主机异步发生。在其他情况下，此参数没有效果。
约束说明
copy_memory_仅支持NPU张量。copy_memory_的输入张量应具有相同的dtype和设备index。

示例
>>> a=torch.IntTensor([0,  0, -1]).npu()
>>> b=torch.IntTensor([1, 1, 1]).npu()
>>> torch_npu.copy_memory_(a, b) 
tensor([1, 1, 1], device='npu:0', dtype=torch.int32)
"""
)


_add_torch_npu_docstr(
    "empty_with_format",
    """
torch_npu.empty_with_format(size, dtype, layout, device, pin_memory, acl_format)
功能描述
返回一个填充未初始化数据的张量。

参数说明
size (ListInt) - 定义输出张量shape的整数序列。可以是参数数量(可变值)，也可以是列表或元组等集合。
dtype (torch.dtype, 可选，默认值为None) - 返回张量所需数据类型。如果值为None，请使用全局默认值(请参见torch.set_default_tensor_type()).
layout (torch.layout, 可选，默认值为torch.strided) - 返回张量所需布局。
device (torch.device, 可选，默认值为None) - 返回张量的所需设备。
pin_memory (Bool, 可选，默认值为False) - 如果设置此参数，返回张量将分配在固定内存中。
acl_format (Int，默认值为2) - 返回张量所需内存格式。
示例
>>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "fast_gelu",
    """
torch_npu.fast_gelu(self) -> Tensor

功能描述
gelu的npu实现。支持FakeTensor模式。

参数说明
self (Tensor) - 数据类型：float16、float32。

示例
示例一：

>>> x = torch.rand(2).npu()
>>> x
tensor([0.5991, 0.4094], device='npu:0')
>>> torch_npu.fast_gelu(x)
tensor([0.4403, 0.2733], device='npu:0')
示例二：

//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2).npu()
...     torch_npu.fast_gelu(x)
>>> FakeTensor(..., device='npu:0', size=(2,))
"""
)

_add_torch_npu_docstr(
    "npu_fast_gelu",
    """
功能描述
算子功能: 快速高斯误差线性单元激活函数(Fast Gaussian Error Linear Units activation function), 对输入的每个元素计算FastGelu的前向结果. 
计算公式
公式1: fast_gelu(x)=$$\frac{x}{1+e^{-1.702\begin{vmatrix}x\end{vmatrix}}}e^{0.851x(x-\begin{vmatrix}x\end{vmatrix})
该公式支持: Atlas 训练系列产品/Atlas 推理系列产品
公式2: $$\frac{x}{1+e^{-1.702x}}
该公式支持: Atlas A2 训练系列产品/Atlas 800I A2 推理产品/Atlas A3 训练系列产品

接口原型
torch_npu.npu_fast_gelu(Tensor input) -> Tensor

参数说明
input: Tensor类型, 即公式中的x. 数据格式支持ND, 支持非连续的Tensor. 输入最大支持8维. 
Atlas 训练系列产品: 数据类型支持float16、float32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float32、bfloat16. 
Atlas A3 训练系列产品: 数据类型支持float16、float32、bfloat16. 
Atlas 推理系列产品: 数据类型仅支持float16、float32. 

输出说明
一个Tensor类型的输出, 代表fast_gelu的计算结果. 

约束说明
该接口支持推理、训练场景下使用. 
该接口支持图模式. 
input输入不能含有None. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1

支持的型号
Atlas 训练系列产品
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列产品

示例
单算子调用
import os
import torch
import torch_npu
import numpy as np
data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
x = torch.from_numpy(data_var).to(torch.float32).npu()
y = torch_npu.npu_fast_gelu(x).cpu().numpy()
图模式调用
import os
import torch
import torch_npu
import numpy as np
import torch.nn as nn
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

os.environ["ENABLE_ACLNN"] = "false"
torch_npu.npu.set_compile_mode(jit_compile=True)
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
    def forward(self, x): 
        y = torch_npu.npu_fast_gelu(x)
        return y
        
npu_mode = Network()
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
x = torch.from_numpy(data_var).to(torch.float32)
y =npu_mode(x).cpu().numpy()
"""
)

_add_torch_npu_docstr(
    "npu_alloc_float_status",
    """
torch_npu.npu_alloc_float_status(self) -> Tensor

功能描述
生成一个包含8个0的一维张量。

参数说明
self (Tensor) - 任何张量。

示例
>>> input    = torch.randn([1,2,3]).npu()
>>> output = torch_npu.npu_alloc_float_status(input)
>>> input
tensor([[[ 2.2324,  0.2478, -0.1056],
        [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
>>> output
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_anchor_response_flags",
    """
torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor
功能描述
在单个特征图中生成锚点的责任标志。

参数说明
self (Tensor) - 真值框，shape为[batch, 4]的2D张量。
featmap_size (ListInt of length 2) - 特征图大小。
strides (ListInt of length 2) - 当前水平的步长。
num_base_anchors (Int) - base anchors的数量。
示例
>>> x = torch.rand(100, 4).npu()
>>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
>>> y.shape
torch.Size([32400])
"""
)


_add_torch_npu_docstr(
    "npu_apply_adam",
    """
torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
功能描述
adam结果计数。

参数说明
beta1_power (Scalar) - beta1的幂。
beta2_power (Scalar) - beta2的幂。
lr (Scalar) - 学习率。
beta1 (Scalar) - 一阶矩估计值的指数衰减率。
beta2 (Scalar) - 二阶矩估计值的指数衰减率。
epsilon (Scalar) - 添加到分母中以提高数值稳定性的项数。
grad (Tensor) - 梯度。
use_locking (Bool，可选) - 设置为True时使用lock进行更新操作。
use_nesterov (Bool，可选) - 设置为True时采用nesterov更新。
var (Tensor) - 待优化变量。
m (Tensor) - 变量平均值。
v (Tensor) - 变量方差。
"""
)


_add_torch_npu_docstr(
    "npu_batch_gather_matmul",
    """
接口原型:
npu_batch_gather_matmul(Tensor input, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None, int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor

功能描述:
npu_batch_gather_matmul: 对于GPU的Batched Gather Matrix-Vector Multiplication (BGMV)。将输入x根据输入索引indices，分别和对应的weight_a，weight_b相乘， 然后将结果累加到输入y并输出。

参数说明:
input：Device侧的tensor，表示待进行累加更新的张量，数据类型Float16，shape支持2维：[batch_size, y_column]。数据格式支持ND。第一维需要和x的第一维一致。支持非连续的Tensor，不支持空Tensor。
x：Device侧的tensor，表示分组前的输入张量，数据类型Float16，shape支持2维：[batch_size, H1]，且H1是16的整数倍。数据格式支持ND。支持非连续的Tensor，不支持空Tensor。
weight_b：Device侧的tensor，表示进行矩阵乘的第二个权重矩阵，数据类型Float16。shape支持4维：[W, L, H2, R]，第三维需要小于y的第二维（H2<y_column），且H2是16的整数倍。当weight_a为空，weight_b 的shape 是[W, L, H2, H1]。支持非连续的Tensor，不支持空Tensor。
indices：Device侧的tensor，标识输入x的分组索引，数据类型Int32。shape支持1维：[batch_size]。数据格式支持ND。第一维需要和x以及y的第一维保持一致。支持非连续的Tensor，不支持空Tensor。
weight_a：Device侧的tensor，表示进行矩阵乘的第一个权重矩阵，数据类型Float16。为空指针时会跳过第一个矩阵乘。shape支持4维：[W, L, R, H1]，前两维需要和weight_b的前两维一致，用W和L表示；第三维需要和weight_b的第四维保持一致，都用R表示，R需要是16的整数倍且取值范围为[16, 128] ；第四维需要和x的第二维保持一致，都用H1表示，需要是16的整数倍。支持非连续的Tensor，不支持空Tensor。
layer_idx：Host侧的整型，表示weight的层数索引，数据类型Int，默认值为0。默认值为0。值需要小于weight_b的第二个维度L。
scale： Host侧的浮点型，表示matmul结果的缩放系数，数据类型Float，默认值为1e-3。
y_offset： Host侧的整型，表示y更新的偏移值，数据类型Int，默认值为0。值需要小于y的第二个维度y_column。
y_slice_size： Host侧的整型，表示y更新时的范围，数据类型Int，默认值为-1。当为-1时，按照y_column的值传入；当非-1 时，以传入的值做更新范围。

输出说明:
out：Device侧的Tensor类型，计算输出，复用y输入地址；数据类型和shape与y一致。

约束说明:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品：仅在推理场景下使用。

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例:
单算子调用
import numpy as np
import torch
import torch_npu

x_data=torch.from_numpy(np.random.uniform(-1, 1, (4096, 16)).astype(np.float16)).npu() 
y_data = torch.from_numpy(np.ones((4096, 6144)).astype(np.float16)).npu()
wa_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 16, 4096)).astype(np.float16)).npu()
wb_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 4096, 16)).astype(np.float16)).npu()
indices_data =torch.from_numpy(np.random.randint(-1, 2, size=4096).reshape(4096).astype(np.int32)).npu()
pred=torch_npu.npu_batch_gather_matmul(y_data,x_data,wb_t_all_data,indices_data,wa_t_all_data,y_slice_size=4096,scale=1e-3,y_offset=0,layer_idx=0)
torch_npu.npu_batch_gather_matmul_(y_data,x_data,wb_t_all_data,indices_data,wa_t_all_data,y_slice_size=4096,scale=1e-3,y_offset=0,layer_idx=0)
print(y_data)

图模式调用
import numpy as np
import torch
import torch_npu
import torchair
config = torchair.CompilerConfig()
npu_backend_plain = torchair.get_npu_backend(compiler_config=config)
x_data=torch.from_numpy(np.random.uniform(-1, 1, (4096, 16)).astype(np.float16)).npu() 
y_data = torch.from_numpy(np.ones((4096, 6144)).astype(np.float16)).npu()
wa_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 16, 4096)).astype(np.float16)).npu()
wb_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 4096, 16)).astype(np.float16)).npu()
indices_data=torch.from_numpy(np.random.randint(-1,2,size=4096).reshape(4096).astype(np.int32)).npu()
def f(y_data, x_data, wb_t_all_data, indices_data, wa_t_all_data=None, y_slice_size=4096, scale=2, y_offset=0):
    with torch.npu.amp.autocast():
        pred = torch_npu.npu_batch_gather_matmul(y_data, x_data, wb_t_all_data, indices_data, wa_t_all_data, y_slice_size=y_slice_size, scale=scale, y_offset=y_offset, layer_idx=0)
    return pred 
opt =torch.compile(f, backend=npu_backend_plain, dynamic=True)
y2 = opt(y_data, x_data, wb_t_all_data, indices_data)
print(y2)
"""
)


_add_torch_npu_docstr(
    "npu_batch_gather_matmul_",
    """
接口原型:
npu_batch_gather_matmul_(Tensor(a!) input, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None, int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor(a!)

功能描述:
npu_batch_gather_matmul_: npu_batch_gather_matmul的inplace版本。将输入x根据输入索引indices，分别和对应的weight_a，weight_b 相乘，然后将结果累加到输入y并输出。

参数说明:
input ：Device侧的tensor，表示待进行累加更新的张量，数据类型Float16，shape支持2维：[batch_size, y_column]。数据格式支持ND。第一维需要和x的第一维一致。支持非连续的Tensor，不支持空Tensor。
x：Device侧的tensor，表示分组前的输入张量，数据类型Float16，shape支持2维：[batch_size, H1]，且H1是16的整数倍。数据格式支持ND。支持非连续的Tensor，不支持空Tensor。
weight_b：Device侧的tensor，表示进行矩阵乘的第二个权重矩阵，数据类型Float16。shape支持4维：[W, L, H2, R]，第三维需要小于y的第二维（H2<y_column），且H2是16的整数倍。当weight_a为空，weight_b 的shape 是[W, L, H2, H1]。支持非连续的Tensor，不支持空Tensor。
indices：Device侧的tensor，标识输入x的分组索引，数据类型Int32。shape支持1维：[batch_size]。数据格式支持ND。第一维需要和x以及y的第一维保持一致。支持非连续的Tensor，不支持空Tensor。
weight_a ：Device侧的tensor，表示进行矩阵乘的第一个权重矩阵，数据类型Float16。为空指针时会跳过第一个矩阵乘。shape支持4维：[W, L, R, H1]，前两维需要和weight_b的前两维一致，用W和L表示；第三维需要和weight_b的第四维保持一致，都用R表示，R需要是16的整数倍且取值范围为[16, 128] ；第四维需要和x的第二维保持一致，都用H1表示，需要是16的整数倍。支持非连续的Tensor，不支持空Tensor。
layer_idx：Host侧的整型，表示weight的层数索引，数据类型Int，默认值为0。默认值为0。值需要小于weight_b的第二个维度L。
scale： Host侧的浮点型，表示matmul结果的缩放系数，数据类型Float，默认值为1e-3。
y_offset： Host侧的整型，表示y更新的偏移值，数据类型Int，默认值为0。值需要小于y的第二个维度y_column。
y_slice_size： Host侧的整型，表示y更新时的范围，数据类型Int，默认值为-1。当为-1时，按照y_column的值传入；当非-1 时，以传入的值做更新范围。

输出说明:
out：Device侧的Tensor类型，计算输出，复用y输入地址；数据类型和shape与y一致。

约束说明:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品：仅在推理场景下使用。

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例:
单算子调用
import numpy as np
import torch
import torch_npu

x_data=torch.from_numpy(np.random.uniform(-1, 1, (4096, 16)).astype(np.float16)).npu() 
y_data = torch.from_numpy(np.ones((4096, 6144)).astype(np.float16)).npu()
wa_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 16, 4096)).astype(np.float16)).npu()
wb_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 4096, 16)).astype(np.float16)).npu()
indices_data =torch.from_numpy(np.random.randint(-1, 2, size=4096).reshape(4096).astype(np.int32)).npu()
pred=torch_npu.npu_batch_gather_matmul(y_data,x_data,wb_t_all_data,indices_data,wa_t_all_data,y_slice_size=4096,scale=1e-3,y_offset=0,layer_idx=0)
torch_npu.npu_batch_gather_matmul_(y_data,x_data,wb_t_all_data,indices_data,wa_t_all_data,y_slice_size=4096,scale=1e-3,y_offset=0,layer_idx=0)
print(y_data)

图模式调用
import numpy as np
import torch
import torch_npu
import torchair
config = torchair.CompilerConfig()
npu_backend_plain = torchair.get_npu_backend(compiler_config=config)
x_data=torch.from_numpy(np.random.uniform(-1, 1, (4096, 16)).astype(np.float16)).npu() 
y_data = torch.from_numpy(np.ones((4096, 6144)).astype(np.float16)).npu()
wa_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 16, 4096)).astype(np.float16)).npu()
wb_t_all_data =torch.from_numpy(np.random.uniform(-1, 1, (2, 1, 4096, 16)).astype(np.float16)).npu()
indices_data=torch.from_numpy(np.random.randint(-1,2,size=4096).reshape(4096).astype(np.int32)).npu()
def f(y_data, x_data, wb_t_all_data, indices_data, wa_t_all_data=None, y_slice_size=4096, scale=2, y_offset=0):
    with torch.npu.amp.autocast():
        pred = torch_npu.npu_batch_gather_matmul(y_data, x_data, wb_t_all_data, indices_data, wa_t_all_data, y_slice_size=y_slice_size, scale=scale, y_offset=y_offset, layer_idx=0)
    return pred 
opt =torch.compile(f, backend=npu_backend_plain, dynamic=True)
y2 = opt(y_data, x_data, wb_t_all_data, indices_data)
print(y2)
"""
)

_add_torch_npu_docstr(
    "npu_batch_nms",
    """
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
功能描述
根据batch分类计算输入框评分，通过评分排序，删除评分高于阈值(iou_threshold)的框，支持多批多类处理。通过NonMaxSuppression(nms)操作可有效删除冗余的输入框，提高检测精度。NonMaxSuppression：抑制不是极大值的元素，搜索局部的极大值，常用于计算机视觉任务中的检测类模型。

参数说明
self (Tensor) - 必填值，输入框的tensor，包含batch大小，数据类型Float16，输入示例：[batch_size, num_anchors, q, 4]，其中q=1或q=num_classes。
scores (Tensor) - 必填值，输入tensor，数据类型Float16，输入示例：[batch_size, num_anchors, num_classes]。
score_threshold (Float32) - 必填值，指定评分过滤器的iou_threshold，用于筛选框，去除得分较低的框，数据类型Float32。
iou_threshold (Float32) - 必填值，指定nms的iou_threshold，用于设定阈值，去除高于阈值的的框，数据类型Float32。
max_size_per_class (Int) - 必填值，指定每个类别的最大可选的框数，数据类型Int。
max_total_size (Int) - 必填值，指定每个batch最大可选的框数，数据类型Int。
change_coordinate_frame (Bool，默认值为False) -可选值， 是否正则化输出框坐标矩阵，数据类型Bool。
transpose_box (Bool，默认值为False) - 可选值，确定是否在此op之前插入转置，数据类型Bool。True表示boxes使用4,N排布。 False表示boxes使用过N,4排布。
输出说明
nmsed_boxes (Tensor) - shape为(batch, max_total_size, 4)的3D张量，指定每批次输出的nms框，数据类型Float16。
nmsed_scores (Tensor) - shape为(batch, max_total_size)的2D张量，指定每批次输出的nms分数，数据类型Float16。
nmsed_classes (Tensor) - shape为(batch, max_total_size)的2D张量，指定每批次输出的nms类，数据类型Float16。
nmsed_num (Tensor) - shape为(batch)的1D张量，指定nmsed_boxes的有效数量，数据类型Int32。
示例
>>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
>>> scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
>>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
>>> nmsed_boxes
>>> nmsed_scores
>>> nmsed_classes
>>> nmsed_num
"""
)


_add_torch_npu_docstr(
    "npu_bert_apply_adam",
    """
torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out=(var,m,v))
功能描述
adam结果计数。

参数说明
参数:
var (Tensor) - float16或float32类型张量。
m (Tensor) - 数据类型和shape与exp_avg相同。
v (Tensor) - 数据类型和shape与exp_avg相同。
lr (Scalar) - 数据类型与exp_avg相同。
beta1 (Scalar) - 数据类型与exp_avg相同。
beta2 (Scalar) - 数据类型与exp_avg相同。
epsilon (Scalar) - 数据类型与exp_avg相同。
grad (Tensor) - 数据类型和shape与exp_avg相同。
max_grad_norm (Scalar) - 数据类型与exp_avg相同。
global_grad_norm (Scalar) - 数据类型与exp_avg相同。
weight_decay (Scalar) - 数据类型与exp_avg相同。
step_size (Tensor，可选，默认值为None) - shape为(1, )，数据类型与exp_avg一致。
adam_mode (Int，默认值为0) - 选择adam模式。0表示“adam”，1表示“mbert_adam”。
关键字参数:
out (Tensor，可选) - 输出张量。
示例
>>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
>>> m_in = torch.zeros(321538).npu()
>>> v_in = torch.zeros(321538).npu()
>>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
>>> max_grad_norm = -1.
>>> beta1 = 0.9
>>> beta2 = 0.99
>>> weight_decay = 0.
>>> lr = 0.
>>> epsilon = 1e-06
>>> global_grad_norm = 0.
>>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
>>> var_out
tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872],
      device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_bmmV2",
    """
torch_npu.npu_bmmV2(self, mat2, output_sizes) -> Tensor
功能描述
将矩阵“a”乘以矩阵“b”，生成“a*b”。支持FakeTensor模式。

参数说明
self (Tensor) - 2D或更高维度矩阵张量。数据类型：float16、float32、int32。格式：[ND, NHWC, FRACTAL_NZ]。
mat2 (Tensor) - 2D或更高维度矩阵张量。数据类型：float16、float32、int32。格式：[ND, NHWC, FRACTAL_NZ]。
output_sizes (ListInt，默认值为[]) - 输出的shape，用于matmul的反向传播。
示例
示例一：

>>> mat1 = torch.randn(10, 3, 4).npu()
>>> mat2 = torch.randn(10, 4, 5).npu()
>>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
>>> res.shape
torch.Size([10, 3, 5])
示例二：

//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     mat1 = torch.randn(10, 3, 4).npu()
...     mat2 = torch.randn(10, 4, 5).npu()
...     result = torch_npu.npu_bmmV2(mat1, mat2, [])
...
>>> result
FakeTensor(..., device='npu:0', size=(10, 3, 5))

"""
)


_add_torch_npu_docstr(
    "npu_bounding_box_decode",
    """
torch_npu.npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor
功能描述
根据rois和deltas生成标注框。自定义FasterRcnn算子。

参数说明
rois (Tensor) - 区域候选网络(RPN)生成的region of interests(ROI)。shape为(N,4)数据类型为float32或float16的2D张量。“N”表示ROI的数量， “4”表示“x0”、“x1”、“y0”和“y1”。
deltas (Tensor) - RPN生成的ROI和真值框之间的绝对变化。shape为(N,4)数据类型为float32或float16的2D张量。“N”表示错误数，“4”表示“dx”、“dy”、“dw”和“dh”。
means0 (Float) - index。
means1 (Float) - index。
means2 (Float) - index。
means3 (Float，默认值为[0,0,0,0]) - index。"deltas" = "deltas" x "stds" + "means"
stds0 (Float) - index。
stds1 (Float) - index。
stds2 (Float) - index。
stds3 (Float, 默认值：[1.0,1.0,1.0,1.0]) - index。"deltas" = "deltas" x "stds" + "means"
max_shape (ListInt of length 2) - shape[h, w]，指定传输到网络的图像大小。用于确保转换后的bbox shape不超过“max_shape”。
wh_ratio_clip (Float) -“dw”和“dh”的值在(-wh_ratio_clip, wh_ratio_clip)范围内。
示例
>>> rois = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> deltas = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
>>> output
tensor([[2.5000, 6.5000, 9.0000, 9.0000],
        [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_bounding_box_encode",
    """
torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor
功能描述
计算标注框和ground truth真值框之间的坐标变化。自定义FasterRcnn算子。

参数说明
anchor_box (Tensor) - 输入张量。锚点框。shape为(N,4)数据类型为float32的2D张量。“N”表示标注框的数量，“4”表示“x0”、“x1”、“y0”和“y1”。
ground_truth_box (Tensor) - 输入张量。真值框。shape为(N,4)数据类型为float32的2D张量。“N”表示标注框的数量，“4”表示“x0”、“x1”、“y0”和“y1”。
means0 (Float) - index。
means1 (Float) - index。
means2 (Float) - index。
means3 (Float, 默认值为[0,0,0,0]) - index。 "deltas" = "deltas" x "stds" + "means"
stds0 (Float) - index。
stds1 (Float) - index。
stds2 (Float) - index。
stds3 (Float, 默认值：[1.0,1.0,1.0,1.0]) -index。 "deltas" = "deltas" x "stds" + "means"
示例
>>> anchor_box = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> ground_truth_box = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
>>> output
tensor([[13.3281, 13.3281,  0.0000,  0.0000],
        [13.3281,  6.6641,  0.0000, -5.4922]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_broadcast",
    """
torch_npu.npu_broadcast(self, size) -> Tensor
功能描述
返回self张量的新视图，其单维度扩展，结果连续。

张量也可以扩展更多维度，新的维度添加在最前面。

参数说明
self (Tensor) - 输入张量。
size (ListInt) - 对应扩展尺寸。
示例
>>> x = torch.tensor([[1], [2], [3]]).npu()
>>> x.shape
torch.Size([3, 1])
>>> torch_npu.npu_broadcast(x, [3, 4])
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_ciou",
    """
torch_npu.npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor
功能描述
应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty item，并propose CIoU。

参数说明
boxes1 (Tensor)：格式为xywh、shape为(4, n)的预测检测框。
boxes2 (Tensor)：相应的gt检测框，shape为(4, n)。
trans (Bool，默认值为False)：是否有偏移。
is_cross (Bool，默认值为True)：box1和box2之间是否有交叉操作。
mode (Int，默认值为0)：选择CIoU的计算方式。0表示IoU，1表示IoF。
atan_sub_flag (Bool，默认值为False)：是否将正向的第二个值传递给反向。
输出说明
torch.Tensor：mask操作的结果。

约束说明
到目前为止，CIoU向后只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。

示例
    >>> box1 = torch.randn(4, 32).npu()
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32).npu()
    >>> box2.requires_grad = True
    >>> ciou = torch_npu.npu_ciou(box1, box2, trans=True, is_cross=False, mode=0)
    >>> l = ciou.sum()
    >>> l.backward()
"""
)


_add_torch_npu_docstr(
    "npu_clear_float_status",
    """
torch_npu.npu_clear_float_status(self) -> Tensor
功能描述
在每个核中设置地址0x40000的值为0。

参数说明
self (Tensor) - 数据类型为float32的张量。

示例
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_clear_float_status(x)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_confusion_transpose",
    """
torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first) -> Tensor
功能描述
混淆reshape和transpose运算。

参数说明
self (Tensor) - 数据类型：float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64。
perm (ListInt) - self张量的维度排列。
shape (ListInt) - 输入shape。
transpose_first (Bool) - 如果值为True，首先执行transpose，否则先执行reshape。
示例
>>> x = torch.rand(2, 3, 4, 6).npu()
>>> x.shape
torch.Size([2, 3, 4, 6])
>>> y = torch_npu.npu_confusion_transpose(x, (0, 2, 1, 3), (2, 4, 18), True)
>>> y.shape
torch.Size([2, 4, 18])
>>> y2 = torch_npu.npu_confusion_transpose(x, (0, 2, 1), (2, 12, 6), False)
>>> y2.shape
torch.Size([2, 6, 12])
"""
)


_add_torch_npu_docstr(
    "npu_conv2d",
    """
torch_npu.npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
功能描述
在由多个输入平面组成的输入图像上应用一个2D卷积。

参数说明
input (Tensor) - shape的输入张量，值为 (minibatch, in_channels, iH, iW)。
weight (Tensor) - shape过滤器，值为 (out_channels, in_channels/groups, kH, kW)。
bias (Tensor, 可选) - shape偏差 (out_channels)。
stride (ListInt) - 卷积核步长。
padding (ListInt) - 输入两侧的隐式填充。
dilation (ListInt) - 内核元素间距。
groups (Int) - 对输入进行分组。In_channels可被组数整除。
"""
)


_add_torch_npu_docstr(
    "npu_conv3d",
    """
torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
功能描述
在由多个输入平面组成的输入图像上应用一个3D卷积。

参数说明
input (Tensor) - shape的输入张量，值为 (minibatch, in_channels, iT, iH, iW)。
weight (Tensor) - shape过滤器，值为 (out_channels, in_channels/groups, kT, kH, kW)。
bias (Tensor, 可选) - shape偏差 (out_channels)。
stride (ListInt) - 卷积核步长。
padding (ListInt) - 输入两侧的隐式填充。
dilation (ListInt) - 内核元素间距。
groups (Int) - 对输入进行分组。In_channels可被组数整除。
"""
)


_add_torch_npu_docstr(
    "npu_conv_transpose2d",
    """
torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
功能描述
在由多个输入平面组成的输入图像上应用一个2D转置卷积算子，有时这个过程也被称为“反卷积”。

参数说明
input (Tensor) - shape的输入张量，值为 (minibatch, in_channels, iH, iW)。
weight (Tensor) - shape过滤器，值为 (in_channels, out_channels/groups, kH, kW)。
bias (Tensor, 可选) - shape偏差 (out_channels)。
padding (ListInt) - (dilation * (kernel_size - 1) - padding) 用零来填充输入每个维度的两侧。
output_padding (ListInt) - 添加到输出shape每个维度一侧的附加尺寸。
stride (ListInt) - 卷积核步长。
dilation (ListInt) - 内核元素间距。
groups (Int) - 对输入进行分组。In_channels可被组数整除。
"""
)


_add_torch_npu_docstr(
    "npu_convolution",
    """
torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor
功能描述
在由多个输入平面组成的输入图像上应用一个2D或3D卷积。

参数说明
input (Tensor) - shape的输入张量，值为 (minibatch, in_channels, iH, iW) 或 (minibatch, in_channels, iT, iH, iW)。
weight (Tensor) - shape过滤器，值为 (out_channels, in_channels/groups, kH, kW) 或 (out_channels, in_channels/groups, kT, kH, kW)。
bias (Tensor, 可选) - shape偏差 (out_channels)。
stride (ListInt) - 卷积核步长。
padding (ListInt) - 输入两侧的隐式填充。
dilation (ListInt) - 内核元素间距。
groups (Int) - 对输入进行分组。In_channels可被组数整除。
"""
)


_add_torch_npu_docstr(
    "npu_convolution_transpose",
    """
torch_npu.npu_convolution_transpose(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
功能描述
在由多个输入平面组成的输入图像上应用一个2D或3D转置卷积算子，有时这个过程也被称为“反卷积”。

参数说明
input (Tensor) - shape的输入张量，值为 (minibatch, in_channels, iH, iW) 或 (minibatch, in_channels, iT, iH, iW)。
weight (Tensor) - shape过滤器，值为 (in_channels, out_channels/groups, kH, kW) 或 (in_channels, out_channels/groups, kT, kH, kW)。
bias (Tensor, 可选) - shape偏差 (out_channels)。
padding (ListInt) - (dilation * (kernel_size - 1) - padding) 用零来填充输入每个维度的两侧。
output_padding (ListInt) - 添加到输出shape每个维度一侧的附加尺寸。
stride (ListInt) - 卷积核步长。
dilation (ListInt) - 内核元素间距。
groups (Int) - 对输入进行分组。In_channels可被组数整除。
"""
)


_add_torch_npu_docstr(
    "npu_deformable_conv2d",
    """
torch_npu.npu_deformable_conv2d(self, weight, offset, bias, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)
功能描述
使用预期输入计算变形卷积输出(deformed convolution output)。

参数说明
self (Tensor) - 输入图像的4D张量。格式为“NHWC”，数据按以下顺序存储：[batch, in_height, in_width, in_channels]。
weight (Tensor) - 可学习过滤器的4D张量。数据类型需与self相同。格式为“HWCN”，数据按以下顺序存储：[filter_height, filter_width, in_channels / groups, out_channels]。
offset (Tensor) - x-y坐标偏移和掩码的4D张量。格式为“NHWC”，数据按以下顺序存储：[batch, out_height, out_width, deformable_groups * filter_height * filter_width * 3]。
bias (Tensor，可选) - 过滤器输出附加偏置(additive bias)的1D张量，数据按[out_channels]的顺序存储。
kernel_size (ListInt of length 2) - 内核大小，2个整数的元组/列表。
stride (ListInt) - 4个整数的列表，表示每个输入维度的滑动窗口步长。维度顺序根据self的数据格式解释。N维和C维必须设置为1。
padding (ListInt) - 4个整数的列表，表示要添加到输入每侧(顶部、底部、左侧、右侧)的像素数。
dilations (ListInt，默认值为[1, 1, 1, 1]) - 4个整数的列表，表示输入每个维度的膨胀系数(dilation factor)。维度顺序根据self的数据格式解释。N维和C维必须设置为1。
groups (Int，默认值为1) - int32类型单整数，表示从输入通道到输出通道的阻塞连接数。In_channels和out_channels需都可被“groups”数整除。
deformable_groups (Int，默认值为1) - int32类型单整数，表示可变形组分区的数量。In_channels需可被“deformable_groups”数整除。
modulated (Bool，可选，默认值为True) - 指定DeformableConv2D版本。True表示v2版本, False表示v1版本，目前仅支持v2。
示例
>>> x = torch.rand(16, 32, 32, 32).npu()
>>> weight = torch.rand(32, 32, 5, 5).npu()
>>> offset = torch.rand(16, 75, 32, 32).npu()
>>> output, _ = torch_npu.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
>>> output.shape
torch.Size([16, 32, 32, 32])

"""
)


_add_torch_npu_docstr(
    "npu_diou",
    """
torch_npu.npu_diou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor
功能描述
应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。

参数说明
boxes1 (Tensor) - 格式为xywh、shape为(4, n)的预测检测框。
boxes2 (Tensor) - 相应的gt检测框，shape为(4, n)。
trans (Bool，默认值为False) - 是否有偏移。
is_cross (Bool，默认值为False) - box1和box2之间是否有交叉操作。
mode (Int，默认值为0) - 选择DIoU的计算方式。0表示IoU，1表示IoF。
输出说明
torch.Tensor (Tensor) - mask操作的结果。

约束说明
到目前为止，DIoU向后只支持当前版本中的trans==True、is_cross==False、mode==0('iou')。如果需要反向传播，确保参数正确。

示例
    >>> box1 = torch.randn(4, 32).npu()
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32).npu()
    >>> box2.requires_grad = True
    >>> diou = torch_npu.contrib.function.npu_diou(box1, box2)
    >>> l = diou.sum()
    >>> l.backward()
"""
)


_add_torch_npu_docstr(
    "npu_dropout_with_add_softmax",
    """
torch_npu.npu_dropout_with_add_softmax(Tensor self, Tensor x1, Scalar alpha, float prob, int dim) -> (Tensor, Tensor, Tensor)
功能描述
实现axpy_v2、softmax_v2、drop_out_domask_v3功能。即：

y=x1+ self *alpha

Softmax(xi)= exp(xi)/∑jexp(xj)

output = 根据mask舍弃x中的元素，留下来的元素乘(1/prob)

参数说明
Tensor self：4维张量，shape为(N, C, H, W)。
Tensor x1：4维张量，shape为(N, C, H, W)。
约束说明
self和x1的shape相同；
H和W是[128, 256, 384, 512]其中之一；
(N * C)%32结果为0；
dim为-1。
示例
self = torch.rand(16, 16, 128, 128).npu()
tensor([[[[7.2556e-02, 3.0909e-01, 7.9734e-01,  ..., 6.1179e-01,
           6.2624e-03, 8.5186e-01],
          [8.9196e-02, 3.3319e-01, 4.0780e-01,  ..., 1.9144e-01,
           2.2701e-01, 6.4018e-01],
          [4.7275e-01, 7.4895e-01, 4.6215e-01,  ..., 9.3753e-01,
           6.6048e-02, 8.1877e-02],
          ...,
          [7.9366e-01, 5.1516e-01, 5.6594e-01,  ..., 1.6457e-01,
           1.0640e-01, 3.4322e-03],
          [1.5743e-02, 1.2893e-01, 5.8990e-01,  ..., 4.1721e-01,
           8.7816e-02, 6.8886e-01],
          [4.2980e-01, 5.5447e-01, 3.1894e-01,  ..., 9.2638e-01,
           9.9324e-01, 4.6225e-01]],

         [[6.2426e-01, 4.5948e-01, 1.0837e-01,  ..., 8.9386e-01,
           3.6932e-01, 1.2406e-01],
          [9.1823e-01, 6.2311e-01, 5.1474e-01,  ..., 2.1042e-01,
           6.5943e-01, 3.1797e-01],
          [5.2891e-01, 2.0183e-01, 2.1452e-01,  ..., 9.1638e-01,
           6.4109e-01, 9.4484e-01],
          ...,
          [3.7783e-02, 1.3218e-01, 3.1192e-01,  ..., 2.4931e-01,
           4.8809e-01, 9.6085e-01],
          [3.3197e-01, 9.1186e-02, 2.4839e-01,  ..., 2.1156e-03,
           6.4952e-01, 8.5996e-01],
          [1.7941e-01, 5.1532e-01, 7.8133e-01,  ..., 3.5526e-01,
           5.3576e-01, 6.0538e-01]],

         [[2.6743e-01, 7.4942e-01, 1.9146e-01,  ..., 4.9179e-01,
           6.3319e-01, 9.9269e-01],
          [1.5163e-01, 3.7388e-01, 8.0604e-02,  ..., 8.1193e-01,
           1.7922e-01, 8.6578e-01],
          [8.2558e-01, 9.5139e-01, 2.1313e-01,  ..., 2.1722e-01,
           2.8402e-01, 8.8888e-01],
          ...,
          [1.8222e-01, 2.7645e-01, 6.7305e-01,  ..., 6.8003e-01,
           4.0917e-01, 7.6655e-01],
          [3.1234e-01, 7.8519e-01, 8.8509e-01,  ..., 7.2574e-01,
           9.6134e-01, 2.2267e-01],
          [4.9233e-01, 8.8407e-01, 7.4390e-01,  ..., 5.2253e-02,
           5.5150e-02, 4.4108e-02]],
         ...,
         [[4.3370e-01, 2.1176e-01, 4.7512e-01,  ..., 5.7611e-01,
           3.2619e-01, 1.1523e-01],
          [6.1469e-01, 7.4528e-01, 7.9559e-02,  ..., 9.7112e-01,
           1.8391e-01, 8.9883e-01],
          [8.6677e-02, 3.5051e-02, 1.6875e-01,  ..., 3.9833e-01,
           6.7967e-01, 4.7062e-01],
          ...,
          [7.1648e-01, 1.8378e-01, 5.3054e-01,  ..., 8.4282e-01,
           9.1972e-01, 7.0031e-01],
          [5.9876e-01, 6.7868e-01, 6.4128e-01,  ..., 4.9516e-02,
           7.2571e-01, 5.8792e-01],
          [7.6723e-01, 6.9527e-01, 9.3573e-01,  ..., 6.3490e-02,
           6.6129e-01, 2.4517e-01]],

         [[5.0158e-01, 8.2565e-01, 7.5532e-01,  ..., 6.9342e-01,
           3.3244e-01, 5.3913e-01],
          [2.3347e-01, 9.7822e-02, 1.5009e-01,  ..., 5.5090e-01,
           9.1813e-01, 7.9857e-01],
          [7.2416e-02, 5.9086e-01, 1.2243e-01,  ..., 7.8511e-01,
           2.4803e-01, 5.3717e-01],
          ...,
          [7.4899e-01, 1.5467e-02, 4.9711e-01,  ..., 2.2938e-02,
           1.6099e-01, 3.1928e-01],
          [3.9111e-01, 1.2422e-01, 6.1795e-02,  ..., 8.4212e-01,
           6.1346e-01, 1.0957e-01],
          [3.6311e-02, 8.9652e-01, 7.7428e-01,  ..., 9.2212e-01,
           4.9290e-01, 4.5609e-01]],

         [[2.2052e-01, 4.4260e-01, 8.8627e-01,  ..., 9.2381e-01,
           7.7046e-01, 9.2057e-01],
          [5.5775e-01, 8.8951e-01, 7.9238e-01,  ..., 3.9209e-01,
           9.6636e-01, 8.1876e-01],
          [3.4709e-01, 7.8678e-01, 1.4396e-01,  ..., 7.9073e-01,
           3.9021e-01, 8.5285e-01],
          ...,
          [1.4238e-01, 9.8432e-01, 2.7802e-01,  ..., 5.1720e-01,
           1.6290e-01, 8.2036e-01],
          [2.0184e-01, 1.0635e-01, 1.9612e-01,  ..., 9.7101e-01,
           9.6679e-01, 7.0811e-01],
          [5.8240e-01, 3.1642e-01, 9.6549e-01,  ..., 5.1130e-02,
           5.6725e-01, 3.5238e-01]]]], device='npu:0')



x1 = torch.rand(16, 16, 128, 128).npu()
tensor([[[[2.4353e-01, 8.5665e-01, 5.3571e-01,  ..., 5.9101e-01,
           4.0872e-01, 6.3873e-01],
          [1.4489e-01, 8.7982e-01, 3.3114e-01,  ..., 2.5155e-01,
           8.4987e-01, 8.7096e-01],
          [6.5837e-02, 2.2677e-02, 7.2063e-01,  ..., 2.3542e-01,
           9.3041e-01, 8.9596e-01],
          ...,
          [5.1450e-01, 7.9412e-01, 8.9288e-01,  ..., 3.3639e-01,
           5.6086e-01, 4.8770e-02],
          [4.7557e-01, 1.4793e-01, 4.9800e-01,  ..., 3.9479e-01,
           5.6052e-01, 9.8271e-01],
          [7.4438e-01, 7.5646e-01, 2.7942e-02,  ..., 3.0381e-01,
           4.3703e-01, 1.4037e-02]],

         [[4.0232e-01, 9.4407e-01, 6.4969e-01,  ..., 3.4524e-01,
           8.2647e-01, 5.4792e-01],
          [1.1801e-01, 1.8281e-01, 6.1723e-01,  ..., 1.9393e-01,
           4.5877e-01, 8.9990e-01],
          [2.6244e-01, 6.9614e-01, 3.6008e-01,  ..., 5.0258e-01,
           8.1919e-01, 4.6943e-01],
          ...,
          [7.4710e-01, 5.8911e-01, 1.5292e-01,  ..., 6.6590e-01,
           4.0754e-01, 3.6944e-01],
          [9.0501e-01, 2.7943e-01, 3.7068e-01,  ..., 1.5053e-01,
           7.3413e-01, 7.9626e-01],
          [9.5200e-01, 7.8327e-01, 3.4033e-01,  ..., 8.0892e-01,
           4.0480e-01, 3.8717e-01]],

         [[7.5938e-01, 2.9089e-01, 5.9916e-01,  ..., 6.2526e-01,
           3.9670e-01, 3.3548e-01],
          [7.0733e-01, 8.1400e-01, 4.9259e-01,  ..., 1.6607e-02,
           6.5331e-01, 7.3150e-02],
          [5.2770e-01, 7.8141e-01, 4.1904e-01,  ..., 3.8917e-01,
           4.1405e-01, 9.9596e-01],
          ...,
          [4.8669e-01, 9.9948e-01, 1.2023e-01,  ..., 7.0420e-01,
           2.8522e-01, 6.6192e-01],
          [4.9718e-01, 7.5792e-01, 6.6748e-01,  ..., 9.7302e-01,
           3.3443e-01, 3.6536e-01],
          [7.7033e-01, 6.0550e-01, 8.2024e-01,  ..., 2.9711e-01,
           1.9410e-01, 6.6304e-01]],
         ...,
         [[1.0284e-01, 6.5712e-01, 6.0831e-01,  ..., 6.2622e-01,
           2.0355e-01, 9.4250e-01],
          [4.9053e-01, 2.0148e-01, 2.4974e-01,  ..., 9.2521e-01,
           1.9919e-01, 4.4700e-01],
          [7.6515e-01, 8.7755e-01, 1.3500e-01,  ..., 8.2136e-01,
           2.0848e-01, 5.6432e-01],
          ...,
          [3.3618e-01, 1.8585e-01, 5.3475e-01,  ..., 4.9333e-01,
           9.1018e-01, 9.5052e-01],
          [2.1400e-01, 1.7407e-01, 5.8925e-01,  ..., 7.5722e-01,
           2.9850e-01, 3.9298e-01],
          [6.3625e-01, 1.7168e-01, 2.9183e-01,  ..., 9.9674e-01,
           2.1718e-01, 5.2626e-01]],

         [[1.8651e-01, 2.5385e-01, 2.0384e-01,  ..., 3.4462e-01,
           8.4150e-01, 4.7431e-01],
          [2.4992e-01, 1.1788e-01, 1.9730e-01,  ..., 4.3722e-02,
           7.8943e-01, 9.9097e-01],
          [1.4493e-02, 6.4856e-01, 8.3344e-01,  ..., 8.6623e-01,
           1.5456e-01, 7.8423e-01],
          ...,
          [6.1458e-01, 4.4260e-01, 7.4133e-01,  ..., 2.5126e-01,
           2.7251e-01, 6.9784e-01],
          [2.2419e-01, 3.4159e-01, 2.3232e-01,  ..., 8.2850e-01,
           8.2644e-02, 4.8390e-01],
          [1.0171e-01, 8.7662e-01, 2.0457e-01,  ..., 7.6868e-01,
           7.6592e-01, 3.1254e-01]],

         [[1.8866e-01, 1.5755e-01, 3.1025e-02,  ..., 6.5044e-01,
           7.8293e-01, 9.8030e-01],
          [3.7703e-01, 5.3198e-01, 1.8633e-01,  ..., 4.7398e-01,
           8.3618e-01, 8.7283e-01],
          [5.7119e-01, 4.3620e-01, 8.2536e-01,  ..., 2.5390e-01,
           5.6144e-01, 4.4044e-01],
          ...,
          [1.3243e-01, 6.2002e-02, 7.5278e-01,  ..., 7.5907e-01,
           4.2472e-01, 1.7624e-01],
          [4.7985e-01, 7.9769e-01, 8.1433e-01,  ..., 7.3780e-01,
           2.2877e-02, 4.8816e-01],
          [4.5100e-01, 9.9698e-02, 7.0776e-01,  ..., 9.8046e-01,
           2.2372e-01, 8.6304e-01]]]], device='npu:0')

_, _, out = torch_npu.npu_dropout_with_add_softmax(self, x1, 2, 0.9, -1)

tensor([[[[0.0000, 0.0639, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0632, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0794,  ..., 0.0000, 0.0000, 0.1571],
          [0.0000, 0.0000, 0.0000,  ..., 0.1270, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.1030, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.2134, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0342, 0.0000, 0.0633,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.1578, 0.1334, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],
         ...,
         [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.2316, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0237, 0.0000,  ..., 0.0000, 0.2128, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.1421, 0.0000, 0.0000,  ..., 0.0499, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0218,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],

         [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1461, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.1130, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.1976,  ..., 0.0000, 0.0000, 0.0000]]]],
       device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_dtype_cast",
    """
torch_npu.npu_dtype_cast(input, dtype) -> Tensor
功能描述
执行张量数据类型(dtype)转换。支持FakeTensor模式。

参数说明
input (Tensor) - 输入张量。
dtype (torch.dtype) - 返回张量的目标数据类型。
示例
示例一：

>>> torch_npu.npu_dtype_cast(torch.tensor([0, 0.5, -1.]).npu(), dtype=torch.int)
tensor([ 0,  0, -1], device='npu:0', dtype=torch.int32)
示例二：

//FakeTensor模式
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     x = torch.rand(2, dtype=torch.float32).npu()
...     res = torch_npu.npu_dtype_cast(x, torch.float16)
...
>>> res
FakeTensor(..., device='npu:0', size=(2,), dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_format_cast",
    """
torch_npu.npu_format_cast(self, acl_format) -> Tensor
功能描述
修改NPU张量的格式。

参数说明
self (Tensor) - 输入张量。
acl_format (Int) - 目标格式。
示例
>>> x = torch.rand(2, 3, 4, 5).npu()
>>> torch_npu.get_npu_format(x)
0
>>> x1 = torch_npu.npu_format_cast(x, 29)
>>> torch_npu.get_npu_format(x1)
29
"""
)


_add_torch_npu_docstr(
    "npu_format_cast_",
    """
torch_npu.npu_format_cast_(self, src) -> Tensor
功能描述
原地修改self张量格式，与src格式保持一致。

参数说明
self (Tensor) - 输入张量。
src (Tensor，int) - 目标格式。
示例
>>> x = torch.rand(2, 3, 4, 5).npu()
>>> torch_npu.get_npu_format(x)
0
>>> torch_npu.get_npu_format(torch_npu.npu_format_cast_(x, 29))
29
"""
)


_add_torch_npu_docstr(
    "npu_fused_attention_score",
    """
torch_npu.npu_fused_attention_score(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> Tensor
功能描述
实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。

参数说明
query_layer：Tensor类型，仅支持float16。
key_layer：Tensor类型，仅支持float16。
value_layer：Tensor类型，仅支持float16 。
attention_mask：Tensor类型，仅支持float16 。
scale：缩放系数，浮点数标量 。
keep_prob：不做dropout的概率，0-1之间，浮点数。
query_transpose：query是否做转置，bool类型，默认为False 。
key_transpose：key是否做转置，bool类型，默认为False 。
bmm_score_transpose_a：bmm计算中左矩阵是否做转置，bool类型，默认为False。
bmm_score_transpose_b：bmm计算中右矩阵是否做转置，bool类型，默认为False。
value_transpose：value是否做转置，bool类型，默认为False。
dx_transpose：反向计算时dx是否做转置，bool类型，默认为False。
约束说明
输入tensor的格式编号必须均为29，数据类型为FP16。

支持的型号:
Atlas 训练系列产品

示例
>>> import torch
>>> import torch_npu
>>> query_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> key_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> value_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 512).npu(), 29).half()
>>> scale = 0.125
>>> keep_prob = 0.5
>>> context_layer = torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
>>> print(context_layer)
tensor([[0.5010, 0.4709, 0.4841,  ..., 0.4321, 0.4448, 0.4834],
        [0.5107, 0.5049, 0.5239,  ..., 0.4436, 0.4375, 0.4651],
        [0.5308, 0.4944, 0.5005,  ..., 0.5010, 0.5103, 0.5303],
        ...,
        [0.5142, 0.5068, 0.5176,  ..., 0.5498, 0.4868, 0.4805],
        [0.4941, 0.4731, 0.4863,  ..., 0.5161, 0.5239, 0.5190],
        [0.5459, 0.5107, 0.5415,  ..., 0.4641, 0.4688, 0.4531]],
       device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_fusion_attention",
    """
功能描述实现
“Transformer Attention Score”的融合计算, 实现的计算公式如下: 
$y=Softmax(Mask(scale*(pse+query*key^{T}),atten_mask),keep_prob)$
$attention=Dropout(y)*value$

接口原型
torch_npu.npu_fusion_attention(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, float scale=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int[]? prefix=None, int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True, bool sync=False, str softmax_layout="") -> (Tensor, Tensor, Tensor, Tensor, int, int, int)

参数说明
query: Tensor类型, 数据类型支持float16、bfloat16、float32, 数据格式支持ND. 综合约束请见约束说明. 
key: Tensor类型, 数据类型支持float16、bfloat16、float32, 数据格式支持ND. 综合约束请见约束说明. 
value: Tensor类型, 数据类型支持float16、bfloat16、float32, 数据格式支持ND. 综合约束请见约束说明. 
head_num: int类型, 代表head个数, 数据类型支持int64. 综合约束请见约束说明. 
input_layout: string类型, 代表输入query、key、value的数据排布格式, 支持BSH、SBH、BSND、BNSD、TND(actual_seq_qlen/actual_seq_kvlen需传值); 后续章节如无特殊说明, S表示query或key、value的sequence length, Sq表示query的sequence length, Skv表示key、value的sequence length, SS表示Sq*Skv. 
pse: Tensor类型, 可选参数, 表示位置编码. 数据类型支持float16、bfloat16、float32, 数据格式支持ND. 非varlen场景支持四维输入, 包含BNSS格式、BN1Skv格式、1NSS格式. 如果非varlen场景Sq大于1024或varlen场景、每个batch的Sq与Skv等长且是sparse_mode为0、2、3的下三角掩码场景, 可使能alibi位置编码压缩, 此时只需要输入原始PSE最后1024行进行内存优化, 即alibi_compress = ori_pse[:, :, -1024:, :], 参数每个batch不相同时, 输入BNHSkv(H=1024), 每个batch相同时, 输入1NHSkv(H=1024). 
padding_mask: Tensor类型, 暂不支持该传参. 
atten_mask: Tensor类型, 可选参数, 取值为1代表该位不参与计算(不生效), 为0代表该位参与计算, 数据类型支持bool、uint8, 数据格式支持ND, 输入shape类型支持BNSS格式、B1SS格式、11SS格式、SS格式. varlen场景只支持SS格式, SS分别是maxSq和maxSkv. 综合约束请见约束说明. 
scale: 浮点型, 可选参数, 代表缩放系数, 作为计算流中Muls的scalar值, 数据类型支持float, 默认值为1. 
keep_prob: 浮点型, 可选参数, 代表Dropout中1的比例, 取值范围为(0, 1]. 数据类型支持float, 默认值为1, 表示全部保留. 
pre_tockens: 整型, 用于稀疏计算的参数, 可选参数, 数据类型支持int64, 默认值为2147483647. 综合约束请见约束说明. 
next_tockens: 整型, 用于稀疏计算的参数, 可选参数, 数据类型支持int64, 默认值为2147483647. next_tockens和pre_tockens取值与atten_mask的关系请参见sparse_mode参数, 参数取值与atten_mask分布不一致会导致精度问题. 综合约束请见约束说明. 
inner_precise: 整型, 用于提升精度, 数据类型支持int64, 默认值为0. 
当前0、1为保留配置值, 2为使能无效行计算, 其功能是避免在计算过程中存在整行mask进而导致精度有损失, 但是该配置会导致性能下降. 
如果算子可判断出存在无效行场景, 会自动使能无效行计算, 例如sparse_mode为3, Sq > Skv场景. 
prefix: int类型数组, 可选参数, 代表prefix稀疏计算场景每个Batch的N值. 数据类型支持int64, 数据格式支持ND. 综合约束请见约束说明. 
actual_seq_qlen: int类型数组, 可选参数, varlen场景时需要传入此参数. 表示query每个S的累加和长度, 数据类型支持int64, 数据格式支持ND. 综合约束请见约束说明. 
比如真正的S长度列表为: 2 2 2 2 2, 则actual_seq_qlen传: 2 4 6 8 10. 
actual_seq_kvlen: int类型数组, 可选参数, varlen场景时需要传入此参数. 表示key/value每个S的累加和长度. 数据类型支持int64, 数据格式支持ND. 综合约束请见约束说明. 
比如真正的S长度列表为: 2 2 2 2 2, 则actual_seq_kvlen传: 2 4 6 8 10. 
sparse_mode: 整型, 表示sparse的模式, 可选参数. 数据类型支持int64, 默认值为0, 支持配置值为0、1、2、3、4、5、6、7、8. 当整网的atten_mask都相同且shape小于2048*2048时, 建议使用defaultMask模式, 来减少内存使用量. 综合约束请见约束说明. 
softmax_layout: string类型，可选参数，用于控制TND场景下softmax的输出（softmax_max和softmax_sum）的数据排布方式。当前仅在input\_layout=“TND”时进行配置，仅支持传入“TND”。默认情况下，softmax的输出排布为NTD排布；传入TND时，softmax的输出排布为TND排布。
表1 sparse_mode不同取值场景说明
sparse_mode
0: defaultMask模式. 
1: allMask模式. 
2: leftUpCausal模式. 
3: rightDownCausal模式. 
4: band模式. 
5: prefix非压缩模式. varlen场景不支持. 
6: prefix压缩模式. 
7: varlen外切场景, rightDownCausal模式. 仅varlen场景支持. 
8: varlen外切场景, leftUpCausal模式. 仅varlen场景支持. 
atten_mask的工作原理为, 在Mask为True的位置遮蔽query(Q)与key(K)的转置矩阵乘积的值. QKT矩阵在atten_mask为True的位置会被遮蔽
说明: 保留该值, atten_mask中, 应该配置为False; 遮蔽该值, atten_mask中应配置为True. sparse_mode为0时, 代表defaultMask模式. 不传mask: 如果atten_mask未传入则不做mask操作, atten_mask取值为None, 忽略pre_tockens和next_tockens取值. 
next_tockens取值为0, pre_tockens大于等于Sq, 表示causal场景sparse, atten_mask应传入下三角矩阵, 此时pre_tockens和next_tockens之间的部分需要计算,atten_mask应传入下三角矩阵
pre_tockens小于Sq, next_tockens小于Skv, 且都大于等于0, 表示band场景, 此时pre_tockens和next_tockens之间的部分需要计算. atten_mask应传入band形状矩阵
next_tockens为负数, 以pre_tockens=9, next_tockens=-3为例, pre_tockens和next_tockens之间的部分需要计算. 说明: next_tockens为负数时, pre_tockens取值必须大于等于next_tockens的绝对值, 且next_tockens的绝对值小于Skv. 
pre_tockens为负数, 以next_tockens=7, pre_tockens=-3为例, pre_tockens和next_tockens之间的部分需要计算. 说明: pre_tockens为负数时, next_tockens取值必须大于等于pre_tockens的绝对值, 且pre_tockens的绝对值小于Sq. 
sparse_mode为1时, 代表allMask, 即传入完整的atten_mask矩阵. 该场景下忽略next_tockens、pre_tockens取值
sparse_mode为2时, 代表leftUpCausal模式的mask, 对应以左上顶点划分的下三角场景(参数起点为左上角). 该场景下忽略pre_tockens、next_tockens取值.传入的atten_mask为优化后的压缩下三角矩阵(2048*2048)
sparse_mode为3时, 代表rightDownCausal模式的mask, 对应以右下顶点划分的下三角场景(参数起点为右下角). 该场景下忽略pre_tockens、next_tockens取值. atten_mask为优化后的压缩下三角矩阵(2048*2048)
sparse_mode为4时, 代表band场景, 即计算pre_tockens和next_tockens之间的部分, 参数起点为右下角, pre_tockens和next_tockens之间需要有交集. atten_mask为优化后的压缩下三角矩阵(2048*2048). 
sparse_mode为5时, 代表prefix非压缩场景, 即在rightDownCasual的基础上, 左侧加上一个长为Sq, 宽为N的矩阵, N的值由可选输入prefix获取, 例如下图中表示batch=2场景下prefix传入数组[4,5], 每个batch轴的N值可以不一样, 参数起点为左上角. 该场景下忽略pre_tockens、next_tockens取值, atten_mask矩阵数据格式须为BNSS或B1SS
sparse_mode为6时, 代表prefix压缩场景, 即prefix场景时, attenMask为优化后的压缩下三角+矩形的矩阵(3072*2048): 其中上半部分[2048, 2048]的下三角矩阵, 下半部分为[1024,2048]的矩形矩阵, 矩形矩阵左半部分全0, 右半部分全1. 该场景下忽略pre_tockens、next_tockens取值. 
sparse_mode为7时, 表示varlen且为长序列外切场景(即长序列在模型脚本中进行多卡切query的sequence length); 用户需要确保外切前为使用sparse_mode 3的场景; 当前mode下用户需要设置pre_tockens和next_tockens(起点为右下顶点), 且需要保证参数正确, 否则会存在精度问题. Masked QKT矩阵示意如下, 在第二个batch对query进行切分, key和value不切分, 4x6的mask矩阵被切分成2x6和2x6的mask, 分别在卡1和卡2上计算: 卡1的最后一块mask为band类型的mask, 配置pre_tockens=6(保证大于等于最后一个Skv), next_tockens=-2, actual_seq_qlen应传入{3,5}, actual_seq_kvlen应传入{3,9}. 卡2的mask类型切分后不变, sparse_mode为3, actual_seq_qlen应传入{2,7,11}, actual_seq_kvlen应传入{6,11,15}. 
如果配置sparse_mode=7, 但实际只存在一个batch, 用户需按照band模式的要求来配置参数; sparse_mode=7时, 用户需要输入2048x2048的下三角mask作为该融合算子的输入. 
基于sparse_mode=3进行外切产生的band模式的sparse的参数应符合以下条件: 
pre_tockens >= last_Skv. 
next_tockens <= 0. 
当前模式下不支持可选输入pse. 
sparse_mode为8时, 表示varlen且为长序列外切场景; 用户需要确保外切前为使用sparse_mode 2的场景; 当前mode下用户需要设置pre_tockens和next_tockens(起点为右下顶点), 且需要保证参数正确, 否则会存在精度问题. Masked QKT矩阵示意如下, 在第二个batch对query进行切分, key和value不切分, 5x4的mask矩阵被切分成2x4和3x4的mask, 分别在卡1和卡2上计算: 卡1的mask类型切分后不变, sparse_mode为2, actual_seq_qlen应传入{3,5}, actual_seq_kvlen应传入{3,7}. 卡2的第一块mask为band类型的mask, 配置pre_tockens=4(保证大于等于第一个Skv), next_tockens=1, actual_seq_qlen应传入{3,8,12}, actual_seq_kvlen应传入{4,9,13}. 
如果配置sparse_mode=8, 但实际只存在一个batch, 用户需按照band模式的要求来配置参数; sparse_mode=8时, 用户需要输入2048x2048的下三角mask作为该融合算子的输入. 
基于sparse_mode=2进行外切产生的band模式的sparse的参数应符合以下条件: 
pre_tockens >= first_Skv. 
next_tockens范围无约束, 根据实际情况进行配置. 
当前模式下不支持可选输入pse. 
gen_mask_parallel: 布尔型, DSA生成dropout随机数向量mask的控制开关. 默认值为True: 同AI Core计算并行; 设为False: 同AI Core计算串行. 
sync: 布尔型, DSA生成dropout随机数向量mask的控制开关. 默认值为False: dropout mask异步生成; 设为True: dropout mask同步生成.

输出说明
共7个输出, 类型依次为Tensor、Tensor、Tensor、Tensor、int、int、int. 
第1个输出为Tensor, 计算公式的最终输出attention_out, 数据类型支持float16、bfloat16、float32. 
第2个输出为Tensor, Softmax计算的Max中间结果, 用于反向计算, 数据类型支持float. 
第3个输出为Tensor, Softmax计算的Sum中间结果, 用于反向计算, 数据类型支持float. 
第4个输出为Tensor, 预留参数, 暂未使用. 
第5个输出为int, DSA生成dropoutmask中, Philox算法的seed. 
第6个输出为int, DSA生成dropoutmask中, Philox算法的offset. 
第7个输出为int, DSA生成dropoutmask的长度. 

约束说明
该接口仅在训练场景下使用. 
该接口暂不支持图模式. 
输入query、key、value、pse的数据类型必须一致. 
输入query、key、value的input_layout必须一致. 
输入query、key、value的shape说明:
1. 输入key和value的shape必须一致. 
2. B: batchsize必须相等; 非varlen场景B取值范围1~2M; varlen场景B取值范围1~2K. 
3. D: Head Dim必须满足Dq=Dk和Dk≥Dv，取值范围1~768. 
4. S: sequence length, 取值范围1~1M. 
varlen场景下:
1. 要求T(B*S)取值范围1~1M. 
2. atten_mask输入不支持补pad，即atten_mask中不能存在某一行全1的场景.
支持输入query的N和key/value的N不相等, 但必须成比例关系, 即Nq/Nkv必须是非0整数, Nq取值范围1~256. 当Nq/Nkv > 1时, 即为GQA\(grouped-query attention); 当Nq/Nkv=1时，即为MHA(multi-head attention). 本文如无特殊说明, N表示的是Nq. 
输入key/value的shape必须一致. 
sparse_mode取值说明:
1. sparse_mode为1、2、3、4、5、6、7、8时, 应传入对应正确的atten_mask, 否则将导致计算结果错误. 当atten_mask输入为None时, sparse_mode, pre_tockens, next_tockens参数不生效, 固定为全计算. 
2. sparse_mode配置为1、2、3、5、6时, 用户配置的pre_tockens、next_tockens不会生效. 
3. sparse_mode配置为0、4时, 需保证atten_mask与pre_tockens、next_tockens的范围一致.
4. sparse_mode配置为7、8时，不支持可选参数pse. 
prefix稀疏计算场景B不大于32, varlen场景不支持非压缩prefix, 即不支持sparse_mode=5; 当Sq>Skv时, prefix的N值取值范围[0, Skv], 当Sq<=Skv时, prefix的N值取值范围[Skv-Sq, Skv]. 
支持actual_seq_qlen中某个Batch上的S长度为0; 如果存在S为0的情况, 不支持pse输入, 假设真实的S长度为[2, 2, 0, 2, 2], 则传入的actual_seq_qlen为[2, 4, 4, 6, 8]. actual_seq_qlen的长度取值范围为1~2K, varlen场景下长度最大支持1K. 
TND格式下, 支持尾部部分Batch不参与计算, 此时actual_seq_qlen和actual_seq_kvlen尾部传入对应个数个0即可. 假设真实的S长度为[2, 3, 4, 5, 6], 此时后两个Batch不参与计算, 则传入的actual_seq_qlen为[2, 5, 9, 0, 0]. 
部分场景下, 如果计算量过大可能会导致算子执行超时(aicore error类型报错, errorStr为: timeout or trap error), 此时建议做轴切分处理, 注: 这里的计算量会受B、S、N、D等参数的影响, 值越大计算量越大.

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11.0

支持的型号
Atlas A2 训练系列产品

调用示例
单算子模式调用: 
import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, atten_mask):
        scale = 0.08838
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        qk = qk + atten_mask * (-10000.0)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1)
        attention_out = torch.matmul(softmax_res, value)
        return attention_out

    def custom_op_exec(self, query, key, value, sparse_params):
        scale = 0.08838
        atten_mask = None
        if sparse_params[0] == 0:
            shape = [1, 8, 256, 256]
            atten_mask_u = np.triu(np.ones(shape), k=sparse_params[1] + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-sparse_params[2] - 1)
            atten_masks = atten_mask_u + atten_mask_l
            atten_mask = torch.tensor(atten_masks).to(torch.float16).bool().npu()
        if sparse_params[0] == 2 or sparse_params[0] == 3 or sparse_params[0] == 4:
            atten_masks = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
            atten_mask = torch.tensor(atten_masks).to(torch.float16).bool().npu()
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=8, input_layout="BNSD", scale=scale, sparse_mode=sparse_params[0],
            atten_mask=atten_mask, pre_tockens=sparse_params[1], next_tockens=sparse_params[2])

    def get_atten_mask(self, sparse_mode=0, pre_tokens=65536, next_tokens=65536):
        atten_masks = []
        shape = [1, 8, 256, 256]
        if sparse_mode == 0:
            atten_mask_u = np.triu(np.ones(shape), k=next_tokens + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-pre_tokens - 1)
            atten_masks = atten_mask_u + atten_mask_l

        elif sparse_mode == 1:
            atten_masks = np.zeros(shape)
            pre_tokens = 65536
            next_tokens = 65536

        elif sparse_mode == 2:
            atten_masks = np.triu(np.ones(shape), k=1)

        elif sparse_mode == 3:
            atten_masks = np.triu(np.ones(shape), k=1)

        elif sparse_mode == 4:
            atten_mask_u = np.triu(np.ones(shape), k=next_tokens + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-pre_tokens - 1)
            atten_masks = atten_mask_u + atten_mask_l

        atten_mask = torch.tensor(atten_masks).to(torch.float16)
        return atten_mask

    # sparse_params = [sparse_mode, pre_tokens, next_tokens]
    # Prec and prec16 indicate the precision comparison standards for float32 and float16 respectively.
    # In this example, 0.01 is used as the standard. You can change the value as required. 
    def check_result(self, query, key, value, sparse_params):
        atten_mask = self.get_atten_mask(sparse_params[0], sparse_params[1], sparse_params[2])
        output = self.supported_op_exec(query.float(), key.float(), value.float(), atten_mask)
        fa_result = self.custom_op_exec(query.npu(), key.npu(), value.npu(), sparse_params)
        self.assertRtolEqual(output.half(), fa_result[0], prec=0.01, prec16=0.01)


    def test_npu_flash_attention(self, device="npu"):
        query = torch.randn(1, 8, 256, 256, dtype=torch.float16)
        key = torch.randn(1, 8, 256, 256, dtype=torch.float16)
        value = torch.randn(1, 8, 256, 256, dtype=torch.float16)

        # sparse_params: [sparse_mode, pre_tokens, next_tokens]
        sparse_params_list = [
            [0, 128, 128],
            [1, 65536, 65536],
            [2, 65536, 0],
            [3, 65536, 0],
            [4, 128, 128]
        ]

        for sparse_params in sparse_params_list:
            self.check_result(query, key, value, sparse_params)

if __name__ == "__main__":
    run_tests()
"""
)


_add_torch_npu_docstr(
    "npu_geglu",
    """
torch_npu. npu_geglu(Tensor self, int dim=-1, int approximate=1) -> (Tensor, Tensor)
功能描述
对输入Tensor完成GeGlu运算。

参数说明
Tensor self：待进行GeGlu计算的入参，npu device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16(Atlas A2 训练系列产品支持)，支持非连续的Tensor，数据格式支持ND。
int dim：可选入参，设定的slice轴，数据类型支持INT64。
int approximate：可选入参，GeGlu计算使用的激活函数索引，0表示使用none，1表示使用tanh，数据类型支持INT64。
out：GeGlu计算的出参，npu device侧的aclTensor，数据类型必须和self一致，支持非连续的Tensor，数据格式支持ND。
outGelu：GeGlu计算的出参，npu device侧的aclTensor，数据类型必须和self一致，支持非连续的Tensor，数据格式支持ND。
约束说明
out、outGelu在dim维的size等于self在dim维size的1/2。

当self.dim()==0时，dim的取值在[-1, 0]范围内；当self.dim()>0时，取值在[-self.dim(), self.dim()-1]范围内。

示例
data_x = np.random.uniform(-2, 2, [24,9216,2560]).astype(np.float16)
x_npu = torch.from_numpy(data_x).npu()

x_npu:
tensor([[[ 0.8750,  0.4766, -0.3535,  ..., -1.4619,  0.3542, -1.8389],
         [ 0.9424, -0.0291,  0.9482,  ...,  0.5640, -1.2959,  1.7666],
         [-0.4958, -0.6787,  0.0179,  ...,  0.4365, -0.8311, -1.7676],
         ...,
         [-1.1611,  1.4766, -1.1934,  ..., -0.5913,  1.1553, -0.4626],
         [ 0.4873, -1.8105,  0.5723,  ...,  1.3193, -0.1558, -1.6191],
         [ 1.6816, -1.2080, -1.6953,  ..., -1.3096,  0.4158, -1.2168]],

        [[ 1.4287, -1.9863,  1.4053,  ..., -1.7676, -1.6709, -1.1582],
         [-1.3281, -1.9043,  0.7725,  ..., -1.5596,  0.1632, -1.0732],
         [ 1.0254, -1.6650,  0.1318,  ..., -0.8159, -0.7134, -0.4536],
         ...,
         [ 0.0327, -0.6206, -0.1492,  ..., -1.2559,  0.3777, -1.2822],
         [-1.1904,  1.1260, -1.3369,  ..., -1.4814,  0.4463,  1.0205],
         [-0.1192,  1.7783,  0.1040,  ...,  1.0010,  1.5342, -0.5728]],

        [[-0.3296,  0.5703,  0.6338,  ...,  0.2131,  1.1113,  0.9854],
         [ 1.4336, -1.7568,  1.8164,  ..., -1.2012, -1.8721,  0.6904],
         [ 0.6934,  0.3743, -0.9448,  ..., -0.9946, -1.6494, -1.3564],
         ...,
         [ 1.1855, -0.9663, -0.8252,  ...,  0.2285, -1.5684, -0.4277],
         [ 1.1260,  1.2871,  1.2754,  ..., -0.5171, -1.1064,  0.9624],
         [-1.4639, -0.0661, -1.7178,  ...,  1.2656, -1.9023, -1.1641]],

        ...,

        [[-1.8350,  1.0625,  1.6172,  ...,  1.4160,  1.2490,  1.9775],
         [-0.5615, -1.9990, -0.5996,  ..., -1.9404,  0.5068, -0.9829],
         [-1.0771, -1.5537, -1.5654,  ...,  0.4678, -1.5215, -1.7920],
         ...,
         [-1.3389, -0.3228, -1.1514,  ...,  0.8882, -1.9971,  1.2432],
         [-1.5439, -1.8154, -1.9238,  ...,  0.2556,  0.2131, -1.7471],
         [-1.1074,  1.0391,  0.1556,  ...,  1.1689,  0.6470,  0.2463]],

        [[ 1.2617, -0.8911,  1.9160,  ..., -0.3027,  1.7764,  0.3381],
         [-1.4160,  1.6201, -0.5396,  ...,  1.8271,  1.3086, -1.8770],
         [ 1.8252,  1.3779, -0.3535,  ..., -1.5215, -1.4727, -1.0420],
         ...,
         [-1.4600, -1.7617, -0.7754,  ...,  0.4697, -0.4734, -0.3838],
         [ 1.8506, -0.3945, -0.0142,  ..., -1.3447, -0.6587,  0.5728],
         [ 1.1523, -1.8027,  0.4731,  ...,  0.5464,  1.4014, -1.8594]],

        [[-0.1467, -0.5752,  0.3298,  ..., -1.9902, -1.8281,  1.8506],
         [ 0.2473,  1.0693, -1.8184,  ...,  1.9277,  1.6543,  1.0088],
         [ 0.0804, -0.7939,  1.3486,  ..., -1.1543, -0.4053, -0.0055],
         ...,
         [ 0.3672,  0.3274, -0.3369,  ...,  1.4951, -1.9580, -0.7847],
         [ 1.3525, -0.4780, -0.5000,  ..., -0.1610, -1.9209,  1.5498],
         [ 0.4905, -1.7832,  0.4243,  ...,  0.9492,  0.3335,  0.9565]]],
       device='npu:0', dtype=torch.float16)

y_npu, y_gelu_npu = torch_npu.npu_geglu(x_npu, dim=-1, approximate=1)

y_npu:
tensor([[[-9.2590e-02, -1.2054e-01,  1.6980e-01,  ..., -6.8542e-02,
          -2.5254e+00, -6.9519e-02],
         [ 1.2405e-02, -1.4902e+00,  8.0750e-02,  ...,  3.4570e-01,
          -1.5029e+00,  2.8442e-01],
         [-9.0271e-02,  4.3335e-01, -1.7402e+00,  ...,  1.3574e-01,
          -5.5762e-01, -1.3123e-01],
         ...,
         [ 1.0004e-01,  1.5312e+00,  1.4189e+00,  ..., -2.6172e-01,
           1.6113e-01, -1.1887e-02],
         [-5.9845e-02,  2.0911e-01, -6.4735e-03,  ...,  5.1422e-02,
           2.6289e+00,  2.5977e-01],
         [ 1.3649e-02, -1.3329e-02, -6.9031e-02,  ...,  3.5977e+00,
          -1.2178e+00, -2.3242e+00]],

        [[-3.1816e+00, -2.6719e+00,  1.4038e-01,  ...,  2.6660e+00,
           7.7820e-02,  2.3999e-01],
         [ 2.9297e+00, -1.7754e+00,  2.6703e-02,  ..., -1.3318e-01,
          -6.2109e-01, -1.9072e+00],
         [ 1.1316e-01,  5.8887e-01,  8.2959e-01,  ...,  1.1273e-01,
           1.1481e-01,  4.2419e-02],
         ...,
         [-2.6831e-01, -1.7288e-02,  2.6343e-01,  ...,  9.3750e-02,
          -2.2324e+00,  1.2894e-02],
         [-2.0630e-01,  5.9619e-01, -1.4210e-03,  ..., -1.2598e-01,
          -6.5552e-02,  1.1115e-01],
         [-1.6143e+00, -1.6150e-01, -4.9774e-02,  ...,  8.6426e-02,
           1.1879e-02, -1.9795e+00]],

        [[ 4.3152e-02,  1.9250e-01, -4.7485e-02,  ..., -5.8632e-03,
           1.4551e-01, -2.1289e+00],
         [ 4.7951e-03,  2.0691e-01,  4.4458e-01,  ...,  4.7485e-02,
          -4.8889e-02,  1.5684e+00],
         [-8.9404e-01, -8.0420e-01, -2.9248e-01,  ...,  1.6205e-02,
           3.5449e+00,  8.2397e-02],
         ...,
         [-1.9385e+00, -1.8838e+00,  6.0010e-01,  ..., -8.5059e-01,
           6.1829e-02,  1.0547e-01],
         [-5.1086e-02, -1.0760e-01, -7.1228e-02,  ..., -9.2468e-02,
           4.7900e-01, -3.5278e-01],
         [ 1.7078e-01,  1.6846e-01,  2.5528e-02,  ...,  1.3708e-01,
           1.4954e-01, -2.8418e-01]],

        ...,

        [[-6.3574e-01, -2.0156e+00,  9.3994e-02,  ...,  2.2402e+00,
          -6.2218e-03,  8.7402e-01],
         [ 1.5010e+00, -1.8518e-01, -3.0930e-02,  ...,  1.1511e-01,
          -3.8300e-02, -1.6150e-01],
         [-2.8442e-01,  4.4373e-02, -1.0022e-01,  ...,  9.2468e-02,
          -1.2524e-01, -1.2115e-01],
         ...,
         [ 3.4760e-02,  1.9812e-01, -9.1431e-02,  ..., -1.1650e+00,
           2.4011e-01, -1.0919e-01],
         [-1.5283e-01,  1.8535e+00,  4.4360e-01,  ...,  6.4844e-01,
          -2.8784e-01, -2.5938e+00],
         [-9.9915e-02,  4.6436e-01,  6.6528e-02,  ..., -1.2817e-01,
          -1.5686e-01, -5.4962e-02]],

        [[-2.3279e-01,  4.5630e-01, -5.4834e-01,  ...,  5.9013e-03,
          -4.7974e-02, -2.7617e+00],
         [-1.0760e-01, -2.0371e+00,  3.7915e-01,  ...,  6.4551e-01,
           2.6953e-01, -1.0910e-03],
         [ 4.9683e-01,  1.2402e+00, -1.0429e-02,  ...,  3.4294e-03,
          -8.2959e-01,  1.2012e-01],
         ...,
         [ 1.6956e-01,  5.3027e-01, -1.6418e-01,  ..., -2.1094e-01,
          -9.8267e-02,  2.3364e-01],
         [ 4.1687e-02, -1.1365e-01,  1.2598e+00,  ..., -5.6299e-01,
           1.5967e+00,  9.3445e-02],
         [ 9.7656e-02, -4.5410e-01, -2.9395e-01,  ..., -1.6565e-01,
          -8.2153e-02, -7.0068e-01]],

        [[ 1.6345e-01,  2.5806e-01, -6.1951e-02,  ..., -6.5857e-02,
          -6.0303e-02, -1.9080e-01],
         [ 1.9666e-01,  1.8262e+00, -1.1951e-01,  ...,  1.0138e-01,
          -2.0911e-01, -6.0638e-02],
         [-6.9141e-01, -2.5234e+00, -1.2734e+00,  ...,  1.0510e-01,
          -1.6504e+00, -9.7070e-01],
         ...,
         [-2.5406e-03, -3.1342e-02, -7.0862e-02,  ...,  9.2041e-02,
           7.7271e-02,  8.0518e-01],
         [-1.5161e-01, -6.8848e-02,  7.0801e-01,  ...,  7.0166e-01,
          -3.3661e-02, -1.4319e-01],
         [-3.0899e-02,  1.4490e-01,  1.9763e-01,  ..., -8.1116e-02,
           7.8955e-01,  1.8347e-01]]], device='npu:0', dtype=torch.float16)

y_gelu_npu:
tensor([[[-1.5771e-01, -1.4331e-01, -1.0846e-01,  ..., -1.1133e-01,
           1.3818e+00, -1.5076e-01],
         [-1.8600e-02,  1.6904e+00, -6.9336e-02,  ...,  3.6890e-01,
           1.6768e+00,  2.5146e-01],
         [ 7.5342e-01,  6.0742e-01,  1.0820e+00,  ...,  1.5063e-01,
           1.1572e+00, -9.4482e-02],
         ...,
         [-1.5796e-01,  8.4082e-01,  9.2627e-01,  ..., -1.6064e-01,
          -1.1096e-01, -1.6370e-01],
         [ 3.4814e-01, -1.6418e-01, -3.1982e-02,  ..., -1.5186e-01,
           1.3330e+00, -1.4111e-01],
         [-8.4778e-02, -1.1023e-01, -1.0669e-01,  ...,  1.9521e+00,
           9.5654e-01,  1.5635e+00]],

        [[ 1.7881e+00,  1.8359e+00, -1.6663e-01,  ...,  1.4609e+00,
          -1.6760e-01, -1.6528e-01],
         [ 1.9434e+00,  1.7168e+00, -1.1615e-01,  ..., -9.8816e-02,
           9.4043e-01,  1.2344e+00],
         [-1.6064e-01,  5.7031e-01,  1.6475e+00,  ..., -1.0809e-01,
          -1.6785e-01, -1.6345e-01],
         ...,
         [-1.6797e-01, -4.6326e-02,  2.6904e-01,  ...,  6.9458e-02,
           1.3174e+00,  1.3486e+00],
         [-1.0645e-01,  3.0249e-01, -9.9411e-03,  ..., -1.3928e-01,
          -1.0974e-01, -7.1533e-02],
         [ 1.7012e+00, -1.0254e-01, -8.2825e-02,  ..., -4.8492e-02,
          -1.1926e-01,  1.7490e+00]],

        [[-6.6650e-02, -1.0370e-01, -2.3788e-02,  ..., -1.0706e-01,
          -1.6980e-01,  1.4209e+00],
         [-5.2986e-03, -1.1133e-01,  2.5439e-01,  ..., -3.9459e-02,
          -6.8909e-02,  1.2119e+00],
         [ 6.1035e-01,  6.8506e-01, -1.5039e-01,  ...,  5.8136e-02,
           1.8232e+00, -6.7383e-02],
         ...,
         [ 1.4434e+00,  1.6787e+00,  1.2422e+00,  ...,  7.5488e-01,
          -5.0720e-02, -6.8787e-02],
         [-1.4600e-01, -1.2213e-01, -1.6711e-01,  ...,  3.7280e-01,
           1.3125e+00,  2.2375e-01],
         [ 3.4985e-01, -1.2659e-01, -4.6722e-02,  ..., -1.4685e-01,
           1.4856e-01, -1.6406e-01]],

        ...,

        [[ 4.8730e-01,  1.6680e+00, -5.7098e-02,  ...,  1.4189e+00,
           7.1983e-03,  7.8857e-01],
         [ 1.1328e+00, -1.6931e-01, -1.1163e-01,  ..., -1.6467e-01,
           3.5309e-02, -1.5173e-01],
         [-1.6858e-01, -8.9111e-02, -1.4709e-01,  ..., -8.1970e-02,
           5.4248e-01,  5.0830e-01],
         ...,
         [ 2.1936e-01,  7.7197e-01,  4.8737e-02,  ...,  8.7842e-01,
          -1.6406e-01, -7.1716e-02],
         [-1.2720e-01,  1.9404e+00,  1.0391e+00,  ...,  7.3877e-01,
          -1.6199e-01,  1.5781e+00],
         [-1.6968e-01,  1.0664e+00, -1.6431e-01,  ..., -7.5439e-02,
          -1.5332e-01,  2.1790e-01]],

        [[ 3.0981e-01,  6.0010e-01,  7.9346e-01,  ...,  4.0169e-03,
           5.8447e-01,  1.7109e+00],
         [-1.6699e-01,  1.7646e+00,  5.9326e-01,  ...,  3.3813e-01,
          -1.5845e-01, -4.7699e-02],
         [ 3.7573e-01,  9.4580e-01, -9.5276e-02,  ...,  2.4805e-01,
           8.3350e-01,  1.2573e-01],
         ...,
         [-1.5369e-01,  1.2021e+00, -1.6626e-01,  ..., -1.1108e-01,
           1.6084e+00, -1.4807e-01],
         [-4.6234e-02, -6.4331e-02,  8.9844e-01,  ...,  9.2871e-01,
           7.9834e-01, -1.6992e-01],
         [-6.4941e-02,  1.1465e+00, -1.5161e-01,  ..., -1.5076e-01,
          -8.6487e-02,  1.0137e+00]],

        [[-1.1731e-01, -1.4404e-01, -8.9050e-02,  ..., -1.2128e-01,
          -1.0919e-01, -1.6943e-01],
         [ 1.5186e-01,  1.1396e+00, -6.5735e-02,  ..., -7.4829e-02,
          -1.6455e-01, -8.9355e-02],
         [ 6.4404e-01,  1.5625e+00,  1.7725e+00,  ..., -5.5176e-02,
           1.7920e+00,  6.6504e-01],
         ...,
         [ 1.9083e-03,  3.8452e-01, -4.9011e-02,  ..., -1.5405e-01,
          -1.6003e-01,  1.3975e+00],
         [ 1.0437e-01, -8.6182e-02,  5.5713e-01,  ...,  1.0645e+00,
          -1.3818e-01,  5.1562e-01],
         [-1.0229e-01, -1.0529e-01,  2.6562e-01,  ..., -5.6702e-02,
           1.0830e+00, -1.6833e-01]]], device='npu:0', dtype=torch.float16)

"""
)


_add_torch_npu_docstr(
    "npu_get_float_status",
    """
torch_npu.npu_get_float_status(self) -> Tensor
功能描述
计算npu_get_float_status算子函数。

参数说明
self (Tensor) - 数据内存地址张量，数据类型为float32。

示例
>>> x = torch.rand(2).npu()
>>> torch_npu.npu_get_float_status(x)
tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_giou",
    """
torch_npu.npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
功能描述
首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。

参数说明
self (Tensor) - 标注框，shape为(N, 4) 数据类型为float16或float32的2D张量。“N”表示标注框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
gtboxes (Tensor) - 真值框，shape为(M, 4) 数据类型为float16或float32的2D张量。“M”表示真值框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
trans (Bool，默认值为False) - 值为True代表“xywh”，值为False代表“xyxy”。
is_cross (Bool，默认值为False) - 控制输出shape是[M, N]还是[1,N]。如果值为True，则输出shape为[M,N]。如果为False，则输出shape为[1,N]。
mode (Int，默认值为0) - 计算模式，取值为0或1。0表示IoU，1表示IoF。
示例
>>> a=np.random.uniform(0,1,(4,10)).astype(np.float16)
>>> b=np.random.uniform(0,1,(4,10)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
>>> output
tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_grid_assign_positive",
    """
torch_npu.npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor
功能描述
执行position-sensitive的候选区域池化梯度计算。

参数说明
self (Tensor) - float16或float32类型的张量, shape为(n, )。
overlaps (Tensor) - 数据类型与assigned_gt_inds相同，表示gt_bboxes和bboxes之间的IoU，shape为(k,n)。
box_responsible_flags (Tensor) - 支持uint8数据类型。表示框是否responsible的标志。
max_overlaps (Tensor) - 数据类型与assigned_gt_inds. overlaps.max(axis=0)相同。
argmax_overlaps (Tensor) - 支持uint32数据类型，overlaps.argmax(axis=0)。
gt_max_overlaps (Tensor) - 数据类型与assigned_gt_inds. overlaps.max(axis=1)相同。
gt_argmax_overlaps (Tensor) - 支持uint32数据类型， overlaps.argmax(axis=1)。
num_gts (Tensor) - 支持uint32数据类型，real k ，shape为 (1, )。
pos_iou_thr (Float) - 正检测框的IoU阈值。
min_pos_iou (Float) - 检测框被视为正检测框的最小IoU
gt_max_assign_all (Bool) - 是否将与某个gt有相同最高重叠的所有检测框分配给该gt。
示例
>>> assigned_gt_inds = torch.rand(4).npu()
>>> overlaps = torch.rand(2,4).npu()
>>> box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).npu()
>>> max_overlap = torch.rand(4).npu()
>>> argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
>>> gt_max_overlaps = torch.rand(2).npu()
>>> gt_argmax_overlaps = torch.tensor([1, 0],dtype=torch.int32).npu()
>>> output = torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap, argmax_overlap, gt_max_overlaps, gt_argmax_overlaps, 128, 0.5, 0., True)
>>> output.shape
torch.Size([4])
"""
)


_add_torch_npu_docstr(
    "npu_gru",
    """
torch_npu.npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
功能描述
计算DynamicGRUV2。

参数说明
input (Tensor) - 数据类型：float16；格式：FRACTAL_NZ。
hx (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
weight_input (Tensor) - 数据类型：float16；格式：FRACTAL_Z。
weight_hidden (Tensor) - 数据类型：float16；格式：FRACTAL_Z。
bias_input (Tensor) - 数据类型：float16, float32；格式：ND。
bias_hidden (Tensor) - 数据类型：float16, float32；格式：ND。
seq_length (Tensor) - 数据类型：int32；格式：ND。
has_biases (Bool，默认值为True)
num_layers (Int)
dropout (Float)
train (Bool，默认值为True) - 标识训练是否在op进行的bool参数。
bidirectional (Bool，默认值为True)
batch_first (Bool，默认值为True)
输出说明
y (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
output_h (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
update (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
reset (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
new (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
hidden_new (Tensor) - 数据类型：float16, float32；格式：FRACTAL_NZ。
"""
)


_add_torch_npu_docstr(
    "npu_hans_encode",
    """
torch_npu.npu_hans_encode(input, statistic, reshuff, out=(pdf, mantissa, fixed, var))
功能描述
对输入张量基于概率密度分布（PDF）进行无损压缩

参数说明
input: Device侧的Tensor类型，表示输入的待压缩张量；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；输入Shape无限制，数据元素大小仅支持64的倍数且大于等于32768。
statistic: bool类型，控制是否重新统计pdf（概率密度分布）；设置为True时会重新统计输入input指数位字节的概率密度分布并覆盖pdf，设置为False时会使用输入的pdf进行压缩；默认值为False；
reshuff: bool类型，控制是否将fixed中多核压缩的结果连续化；限制为fixed大小大于等于压缩上界时候才能使用，详细见约束。设置为True则将多核压缩的结果连续化，设置为False时则不做处理；设置为True时var参数失效；该参数需同步传入解码；默认值为False；

输出说明
pdf：Device侧的Tensor类型，表示input指数位字节的概率密度分布，数据类型为INT32，shape为[1, 256]，其中每一个元素的值表示其对应索引，在input中出现的次数；当statistic设置为True时会统计输入input指数位的pdf并覆盖原有pdf，设置为False时会使用当前输入的pdf进行压缩；
mantissa：可为Device侧的Tensor类型、或Host侧内存通过虚拟内存映射至Deive，表示input输入的尾数部分；数据类型与input保持一致；输入Shape无限制，输入大小见约束。
fixed：Device侧的Tensor类型，表示input指数位字节压缩的定长部分，一般由上层应用设定固定容量的空间来存储压缩结果；数据类型与input保持一致；输入Shape无限制，输入大小见约束。
var：可为Device侧的Tensor类型、或Host侧内存通过虚拟内存映射至Deive，表示input指数位字节压缩的变长部分，一般由上层应用设定容量大小；数据类型与input保持一致；输入Shape无限制，输入大小见约束。

约束说明
输入input的元素数量为64的倍数且大于等于32768。
pdf的shape为[1, 256]，数据类型为INT32。
mantissa.numel() * mantissa.element_size() = input.numel() * (input.element_size() – 1)，尾数的大小可根据input输入的类型和大小严格计算。
fixed.numel() * fixed.element_size() >= 512，即fixed的大小必须大于512Byte来存储压缩的元信息。
fixed.numel() * fixed.element_size() + var.numel() * var.element_size() >= input.numel() + input.numel() / 64 + 8448 * 当前硬件Vector核数，即fixed与var的空间大小总和必须大于压缩上界。
如果reshuff为True，则fixed.numel() * fixed.element_size() 需要大于input.numel() + input.numel() / 64 + 8448 * 硬件vector核数，即保证压缩结果同时存在于fixed上，fixed的大小需大于等于压缩上界。

支持的型号
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例
import torch
import torch_npu
data_shape = (4096, 512)
statistic = True
reshuff = False
input_tensor = torch.randn(data_shape, dtype=dtype).npu()
pdf = torch.zeros(256, dtype=torch.int32).npu()
mantissa_numel = input_tensor.numel() * (input_tensor.element_size() - 1)
mantissa =  torch.zeros(mantissa_numel // input_tensor.element_size(), dtype=input_tensor.dtype).npu()
fixed = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()
var = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()
pdf, mantissa, fixed, var = torch_npu.npu_hans_encode(input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))
"""
)


_add_torch_npu_docstr(
    "npu_hans_decode",
    """
torch_npu.npu_hans_decode( mantissa, fixed, var, pdf, reshuff, out=out)
功能描述
基于概率密度分布（PDF）对压缩后的结果进行无损解压缩

参数说明（包括 类型、默认值、含义、参数使用限制）
mantissa：可为Device侧的Tensor类型、或Host侧内存通过虚拟内存映射至Deive，表示压缩前张量的尾数部分。数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；输入Shape无限制，为npu_hans_encode的输出。
fixed：Device侧的Tensor类型，表示压缩前张量的指数位字节压缩的定长部分；数据类型与input保持一致；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；输入Shape无限制，为npu_hans_encode的输出。
var：可为Device侧的Tensor类型、或Host侧内存通过虚拟内存映射至Deive，表示压缩前张量的指数位字节压缩的变长部分。数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；输入Shape无限制，为npu_hans_encode的输出。
pdf：Device侧的Tensor类型，表示压缩时采用的概率密度分布，数据类型为INT32，shape为[1, 256]。
reshuff: bool类型，表示在压缩时是否将fixed中多核压缩的结果进行了连续化，设置为True则表示已将多核压缩的结果连续化，设置为False时则表示没有将fixed压缩的结果连续化；默认值为False。

输出说明
out：Device侧的Tensor类型，表示解压缩后的张量，数据类型与mantissa等输入一致，Shape无限制，大小详见约束；

约束说明
输出out的元素数量为64的倍数且大于等于32768。
pdf的shape为[1, 256]，数据类型为INT32。
mantissa.numel() * mantissa.element_size() = out.numel() * (out.element_size() – 1)。

支持的型号
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例
import torch
import torch_npu
data_shape = (4096, 512)
statistic = True
reshuff = False
input_tensor = torch.randn(data_shape, dtype=dtype).npu()
recover = torch.zeros(data_shape, dtype=dtype).npu()
pdf = torch.zeros(256, dtype=torch.int32).npu()
mantissa_numel = input_tensor.numel() * (input_tensor.element_size() - 1)
mantissa =  torch.zeros(mantissa_numel // input_tensor.element_size(), dtype=input_tensor.dtype).npu()
fixed = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()
var = torch.zeros(input_tensor.numel(), dtype=input_tensor.dtype).npu()
pdf, mantissa, fixed, var = torch_npu.npu_hans_encode(input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))
recover = torch_npu.npu_hans_decode(mantissa, fixed, var, pdf, reshuff, out=recover)
"""
)


_add_torch_npu_docstr(
    "npu_ifmr",
    """
torch_npu.npu_ifmr(Tensor data, Tensor data_min, Tensor data_max, Tensor cumsum, float min_percentile, float max_percentile, float search_start, float search_end, float search_step, bool with_offset) -> (Tensor, Tensor)
功能描述
使用“begin,end,strides”数组对ifmr结果进行计数。

参数说明
data (Tensor) - 特征图张量。
data_min (Tensor) - 特征图最小值的张量。
data_max (Tensor) - 特征图最大值的张量。
cumsum (Tensor) - cumsum bin数据张量。
min_percentile (Float) - 最小初始化百分位数。
max_percentile (Float) - 最大初始化百分位数。
search_start (Float) - 搜索起点。
search_end (Float) - 搜索终点。
search_step (Float) - 搜索步长。
with_offset (Bool) - 是否使用offset。
输出说明
scale (Tensor) - 最优尺度。
offset (Tensor) - 最优offset。
示例
>>> import torch
>>> import torch_npu
>>> torch.npu.set_compile_mode(jit_compile=True)
>>> input = torch.rand((2,2,3,4),dtype=torch.float32).npu()
>>> input
tensor([[[[0.4508, 0.6513, 0.4734, 0.1924],
          [0.0402, 0.5502, 0.0694, 0.9032],
          [0.4844, 0.5361, 0.9369, 0.7874]],
        [[0.5157, 0.1863, 0.4574, 0.8033],
          [0.5986, 0.8090, 0.7605, 0.8252],
          [0.4264, 0.8952, 0.2279, 0.9746]]],
        [[[0.0803, 0.7114, 0.8773, 0.2341],
         [0.6497, 0.0423, 0.8407, 0.9515],
         [0.1821, 0.5931, 0.7160, 0.4968]],
          [[0.7977, 0.0899, 0.9572, 0.0146],
          [0.2804, 0.8569, 0.2292, 0.1118],
          [0.5747, 0.4064, 0.8370, 0.1611]]]], device='npu:0')
>>> min_value = torch.min(input)
>>> min_value
tensor(0.0146, device='npu:0')
>>> max_value = torch.max(input)
>>> max_value
tensor(0.9746, device='npu:0')
>>> hist = torch.histc(input.to('cpu'),
                         bins=128,
                         min=min_value.to('cpu'),
                         max=max_value.to('cpu'))
>>> hist
tensor([1., 0., 0., 2., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
          0., 1., 0., 0., 2., 1., 0., 0., 0., 0., 2., 1., 0., 0., 0., 0., 0., 1.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
          1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1.,
          0., 0., 1., 0., 0., 2., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.,
         0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 2., 0., 0.,
          1., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1.,
          0., 1.])  >>> cdf = torch.cumsum(hist,dim=0).int().npu()
>>> cdf
tensor([ 1,  1,  1,  3,  3,  3,  3,  4,  5,  5,  6,  6,  7,  7,  7,  7,  7,  7,
          7,  8,  8,  8, 10, 11, 11, 11, 11, 11, 13, 14, 14, 14, 14, 14, 14, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16,
          17, 17, 17, 17, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 23, 24, 24, 25,
         25, 25, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30, 30, 30,
         30, 30, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 34, 35, 37, 37, 37,
          38, 39, 40, 40, 41, 41, 41, 42, 42, 43, 44, 44, 44, 44, 45, 45, 46, 47,
          47, 48], device='npu:0', dtype=torch.int32)
>>> scale, offset = torch_npu.npu_ifmr(input,
                                     min_value,
                                    max_value,
                                   cdf,
                                    min_percentile=0.999999,
                                    max_percentile=0.999999,
                                     search_start=0.7,
                                    search_end=1.3,
                                    search_step=0.01,
                                     with_offset=False)
>>> scale  tensor(0.0080, device='npu:0')
>>> offset  tensor(0., device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_indexing",
    """
torch_npu.npu_indexing(self, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0) -> Tensor
功能描述
使用“begin,end,strides”数组对index结果进行计数。

参数说明
self (Tensor) - 输入张量。
begin (ListInt) - 待选择的第一个值的index。
end (ListInt) - 待选择的最后一个值的index。
strides (ListInt) - index增量。
begin_mask (Int，默认值为0) - 位掩码(bitmask)，其中位“i”为“1”意味着忽略开始值，尽可能使用最大间隔。
end_mask (Int，默认值为0) - 类似于“begin_mask”。
ellipsis_mask (Int，默认值为0) - 位掩码，其中位“i”为“1”意味着第“i”个位置实际上是省略号。
new_axis_mask (Int，默认值为0) - 位掩码，其中位“i”为“1”意味着在第“i”位创建新的1D shape。
shrink_axis_mask (Int，默认值为0) - 位掩码，其中位“i”意味着第“i”位应缩小维数。
示例
>>> input = torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8]], dtype=torch.int32).to("npu")
>>> input
tensor([[1, 2, 3, 4],
      [5, 6, 7, 8]], device='npu:0', dtype=torch.int32)
>>> output = torch_npu.npu_indexing(input, [0, 0], [2, 2], [1, 1])
>>> output
tensor([[1, 2],
      [5, 6]], device='npu:0', dtype=torch.int32)
"""
)


_add_torch_npu_docstr(
    "npu_iou",
    """
torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor
torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor
功能描述
根据ground-truth和预测区域计算交并比(IoU)或前景交叉比(IoF)。

参数说明
bboxes (Tensor) - 输入张量。
gtboxes (Tensor) - 输入张量。
mode (Int，默认值为0) - 0为IoU模式，1为IoF模式。
示例
>>> bboxes = torch.tensor([[0, 0, 10, 10],
                           [10, 10, 20, 20],
                           [32, 32, 38, 42]], dtype=torch.float16).to("npu")
>>> gtboxes = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float16).to("npu")
>>> output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
>>> output_iou
tensor([[0.4985, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
       [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_layer_norm_eval",
    """
torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor
功能描述
对层归一化结果进行计数。与torch.nn.functional.layer_norm相同, 优化NPU设备实现。

参数说明
input (Tensor) - 输入张量。
normalized_shape (ListInt) - size为预期输入的输入shape。
weight (Tensor, 可选，默认值为None) - gamma张量。
bias (Tensor, 可选默认值为None) - beta张量。
eps (Float，默认值为1e-5) - 为保证数值稳定性添加到分母中的ε值。
示例
>>> input = torch.rand((6, 4), dtype=torch.float32).npu()
>>> input
tensor([[0.1863, 0.3755, 0.1115, 0.7308],
        [0.6004, 0.6832, 0.8951, 0.2087],
        [0.8548, 0.0176, 0.8498, 0.3703],
        [0.5609, 0.0114, 0.5021, 0.1242],
        [0.3966, 0.3022, 0.2323, 0.3914],
        [0.1554, 0.0149, 0.1718, 0.4972]], device='npu:0')
>>> normalized_shape = input.size()[1:]
>>> normalized_shape
torch.Size([4])
>>> weight = torch.Tensor(*normalized_shape).npu()
>>> weight
tensor([        nan,  6.1223e-41, -8.3159e-20,  9.1834e-41], device='npu:0')
>>> bias = torch.Tensor(*normalized_shape).npu()
>>> bias
tensor([5.6033e-39, 6.1224e-41, 6.1757e-39, 6.1224e-41], device='npu:0')
>>> output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
>>> output
tensor([[        nan,  6.7474e-41,  8.3182e-20,  2.0687e-40],
        [        nan,  8.2494e-41, -9.9784e-20, -8.2186e-41],
        [        nan, -2.6695e-41, -7.7173e-20,  2.1353e-41],
        [        nan, -1.3497e-41, -7.1281e-20, -6.9827e-42],
        [        nan,  3.5663e-41,  1.2002e-19,  1.4314e-40],
        [        nan, -6.2792e-42,  1.7902e-20,  2.1050e-40]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_attention_update",
    """
接口原型：
npu_attention_update(Tensor[] lse, Tensor[] local_out, int update_type, Tensor? lse_out=None) -> Tensor

功能描述
将各SP域PA算子的输出的中间结果lse, attention out 两个局部结果更新成全局结果。

计算公式：
lsemax = max(lsei)
lse = Σexp(lsei - lsemax)
lsem = lsemax + log(lse)
O = ΣOi * exp(lsei - lsem)

参数说明
lse：Device侧的TensorList，每个Tensor形状为 (batch * seqLen * headNum)，各sp域计算的lse。数据类型支持 FLOAT，格式支持 ND。
localOut：Device侧的TensorList，TensorList长度为sp，每个Tensor形状为 (batch * seqLen * headNum, head_dim)，各sp域计算的output。数据类型支持 FLOAT，格式支持 ND。
updateType：int64_t 类型，指定执行的操作类型。目前仅支持 0（DECODE_UPDATE）。

输出说明
out：Device侧的Tensor，形状为 (batch * seqLen * headNum, head_dim)，数据类型为 FLOAT，格式为 ND。
lse_out: Tensor类型，预留参数，暂未使用

支持的型号
----------------
Atlas A3训练系列产品/Atlas A3推理系列产品
Atlas A2训练系列产品/Atlas 800I A2推理产品/A200I A2 Box异构组件

调用示例
----------------
import torch
import torch_npu

dtype = torch.float32
N = 4
head_dim = 32  

lse = [
    torch.randn(N, dtype=dtype, device='npu'),  # 1D: [N]
    torch.randn(N, dtype=dtype, device='npu'),  # 1D: [N]
]

local_out = [
    torch.randn(N, head_dim, dtype=dtype, device='npu'),  # 2D: [N, head_dim]
    torch.randn(N, head_dim, dtype=dtype, device='npu'),  # 2D: [N, head_dim]
]

update_type = 0  
output, lse_out = torch_npu.npu_attention_update(lse, local_out, update_type)
"""
)


_add_torch_npu_docstr(
    "npu_linear",
    """
torch_npu.npu_linear(input, weight, bias=None) -> Tensor
功能描述
将矩阵“a”乘以矩阵“b”，生成“a*b”。

参数说明
input (Tensor) - 2D矩阵张量。数据类型：float32、float16、int32、int8。格式：[ND, NHWC, FRACTAL_NZ]。
weight (Tensor) - 2D矩阵张量。数据类型：float32、float16、int32、int8。格式：[ND, NHWC, FRACTAL_NZ]。
bias (Tensor，可选，默认值为None) - 1D张量。数据类型：float32、float16、int32。格式：[ND, NHWC]。
示例
>>> x=torch.rand(2,16).npu()
>>> w=torch.rand(4,16).npu()
>>> b=torch.rand(4).npu()
>>> output = torch_npu.npu_linear(x, w, b)
>>> output
tensor([[3.6335, 4.3713, 2.4440, 2.0081],
        [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_lstm",
    """
torch_npu.npu_lstm(x, weight, bias, seqMask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction)
功能描述
计算DynamicRNN。

参数说明
x (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
weight (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_ZN_LSTM。
bias (Tensor) - 1D张量。数据类型：float16, float32；格式：ND。
seqMask (Tensor) - 张量。仅支持为FRACTAL_NZ格式的float16和ND格式的int32类型。
h (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
c (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
has_biases (Bool) - 如果值为True,则存在偏差。
num_layers (Int) - 循环层数，目前只支持单层。
dropout (Float) - 如果值为非零，则在除最后一层外的每个LSTM层的输出上引入一个dropout层，丢弃概率等于dropout参数值。目前不支持。
train (Bool，默认值为True) - 标识训练是否在op进行的bool参数。
bidirectional (Bool) - 如果值为True，LSTM为双向。当前不支持。
batch_first (Bool) - 如果值为True，则输入和输出张量将表示为(batch, seq, feature)。当前不支持。
flag_seq (Bool) - 如果值为True，输入为PackSequnce。当前不支持。
direction (Bool) - 如果值为True，则方向为“REDIRECTIONAL”，否则为“UNIDIRECTIONAL”。
输出说明
y (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
output_h (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
output_c (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
i (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
j (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
f (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
o (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
tanhct (Tensor) - 4D张量。数据类型：float16, float32；格式：FRACTAL_NZ。
"""
)


_add_torch_npu_docstr(
    "npu_masked_fill_range",
    """
torch_npu.npu_masked_fill_range(self, start, end, value, axis=-1) -> Tensor
功能描述
同轴上被range.boxes屏蔽(masked)的填充张量。自定义屏蔽填充范围算子。

参数说明
self (Tensor) - shape为1D (D,)、2D (N,D)或3D (N,D)的float32/float16/int32/int8 ND张量。
start (Tensor) - 屏蔽填充开始位置。shape为(num,N)的int32 3D张量。
end (Tensor) - 屏蔽填充结束位置。shape为(num,N)的int32 3D张量。
value (Tensor) - 屏蔽填充值。shape为(num,)的float32/float16/int32/int8 2D张量。
axis (Int，默认值为-1) - 带有int32屏蔽填充的轴。
示例
>>> a=torch.rand(4,4).npu()
>>> a
tensor([[0.9419, 0.4919, 0.2874, 0.6560],
        [0.6691, 0.6668, 0.0330, 0.1006],
        [0.3888, 0.7011, 0.7141, 0.7878],
        [0.0366, 0.9738, 0.4689, 0.0979]], device='npu:0')
>>> start = torch.tensor([[0,1,2]], dtype=torch.int32).npu()
>>> end = torch.tensor([[1,2,3]], dtype=torch.int32).npu()
>>> value = torch.tensor([1], dtype=torch.float).npu()
>>> out = torch_npu.npu_masked_fill_range(a, start, end, value, 1)
>>> out
tensor([[1.0000, 0.4919, 0.2874, 0.6560],
        [0.6691, 1.0000, 0.0330, 0.1006],
        [0.3888, 0.7011, 1.0000, 0.7878],
        [0.0366, 0.9738, 0.4689, 0.0979]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_max",
    """
torch_npu.npu_max(self, dim, keepdim=False) -> (Tensor, Tensor)
功能描述
使用dim对最大结果进行计数。类似于torch.max, 优化NPU设备实现。

参数说明
self (Tensor) - 输入张量。
dim (Int) - 待降低维度。
keepdim (Bool，默认值为False) - 输出张量是否保留dim。
输出说明
values (Tensor) - 输入张量中的最大值。
indices (Tensor) - 输入张量中最大值的index。
示例
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> input
tensor([[[[-1.8135,  0.2078],
          [-0.6678,  0.7846]],

        [[ 0.6458, -0.0923],
          [-0.2124, -1.9112]]],

        [[[-0.5800, -0.4979],
         [ 0.2580,  1.1335]],

          [[ 0.6669,  0.1876],
          [ 0.1160, -0.1061]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_max(input, 2)
>>> outputs
tensor([[[-0.6678,  0.7846],
        [ 0.6458, -0.0923]],

        [[ 0.2580,  1.1335],
        [ 0.6669,  0.1876]]], device='npu:0')
>>> indices
tensor([[[1, 1],
        [0, 0]],

        [[1, 1],
        [0, 0]]], device='npu:0', dtype=torch.int32)
"""
)


_add_torch_npu_docstr(
    "npu_min",
    """
torch_npu.npu_min(self, dim, keepdim=False) -> (Tensor, Tensor)
功能描述
使用dim对最小结果进行计数。类似于torch.min, 优化NPU设备实现。

参数说明
self (Tensor) - 输入张量。
dim (Int) - 待降低维度。
keepdim (Bool) - 输出张量是否保留dim。
输出说明
values (Tensor) - 输入张量中的最小值。
indices (Tensor) - 输入张量中最小值的index。
示例
>>> input = torch.randn(2, 2, 2, 2, dtype = torch.float32).npu()
>>> input
tensor([[[[-0.9909, -0.2369],
          [-0.9569, -0.6223]],

        [[ 0.1157, -0.3147],
          [-0.7761,  0.1344]]],

        [[[ 1.6292,  0.5953],
          [ 0.6940, -0.6367]],

        [[-1.2335,  0.2131],
          [ 1.0748, -0.7046]]]], device='npu:0')
>>> outputs, indices = torch_npu.npu_min(input, 2)
>>> outputs
tensor([[[-0.9909, -0.6223],
        [-0.7761, -0.3147]],

        [[ 0.6940, -0.6367],
        [-1.2335, -0.7046]]], device='npu:0')
>>> indices
tensor([[[0, 1],
        [1, 0]],

        [[1, 1],
        [0, 1]]], device='npu:0', dtype=torch.int32)
"""
)


_add_torch_npu_docstr(
    "npu_mish",
    """
按元素计算self的双曲正切。

参数解释：
self (Tensor) - 数据类型：float16、float32。
约束条件：
无

示例：
>>> x = torch.rand(10, 30, 10).npu()
>>> y = torch_npu.npu_mish(x)
>>> y.shape
torch.Size([10, 30, 10])
"""
)


_add_torch_npu_docstr(
    "npu_multi_head_attention",
    """
torch_npu.npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor query_bias, Tensor key_bia, Tensor value_bias, Tensor out_proj_bias, Tensor dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
功能描述
实现Transformer模块中的MultiHeadAttention计算逻辑。

参数说明
query: Tensor类型，仅支持float16
key: Tensor类型，仅支持float16
value: Tensor类型，仅支持float16
query_weight: Tensor类型，仅支持float16
key_weight: Tensor类型，仅支持float16
value_weight: Tensor类型，仅支持float16
attn_mask: Tensor类型，仅支持float16
out_proj_weight: Tensor类型，仅支持float16
query_bias: Tensor类型，仅支持float16
key_bias: Tensor类型，仅支持float16
value_bias: Tensor类型，仅支持float16
out_proj _bias: Tensor类型，仅支持float16
dropout_mask_input: Tensor类型，仅支持float16
attn_head_num： Attention Head numbers, Int型
attn_dim_per_head：Attention dim of a Head , Int型
src_len：source length, Int型
tgt_len：target length, Int型
keep_prob：dropout keep probability, Float型
softmax_use_float：SoftMax Use Float32 to keep precision, Bool型
输出说明
y: Tensor类型，仅支持float16
dropout_mask: Tensor类型，仅支持float16
query_res: Tensor类型，仅支持float16
key_res: Tensor类型，仅支持float16
value_res: Tensor类型，仅支持float16
attn_scores: Tensor类型，仅支持float16
attn_res: Tensor类型，仅支持float16
context: Tensor类型，仅支持float16
约束说明
Attr attn_head_num：需16对齐

Attr attn_dim_per_head：需16对齐

Attr src_len：需16对齐

tgt_len：需16对齐

示例
import torch
import torch_npu
import numpy as np

batch = 8
attn_head_num = 16
attn_dim_per_head = 64
src_len = 64
tgt_len = 64
dropout_prob = 0.0
softmax_use_float = True

weight_col = attn_head_num * attn_dim_per_head
query = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
key = torch.from_numpy(np.random.uniform(-1, 1, (batch * src_len, weight_col)).astype("float16")).npu()
value = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
query_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
key_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
value_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
out_proj_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
attn_mask = torch.from_numpy(np.random.uniform(-1, 1, (batch, attn_head_num, tgt_len, src_len)).astype("float16")).npu()
query_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
key_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
value_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
out_proj_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
dropout_mask_input = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()

npu_result, npu_dropout_mask, npu_query_res, npu_key_res, npu_value_res, npu_attn_scores, npu_attn_res, npu_context = torch_npu.npu_multi_head_attention (query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias,  dropout_mask_input, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)

print(npu_result)



tensor([[ 623.5000,   75.5000,  307.0000,  ...,   25.3125, -418.7500,
           35.9688],
        [-254.2500, -165.6250,  176.2500,  ...,   87.3750,   78.0000,
           65.2500],
        [ 233.2500,  207.3750,  324.7500,  ...,   38.6250, -264.2500,
          153.7500],
        ...,
        [-110.2500,  -92.5000,  -74.0625,  ...,  -68.0625,  195.6250,
         -157.6250],
        [ 300.0000, -184.6250,   -6.0039,  ...,  -15.7969, -299.0000,
          -93.1875],
        [  -2.5996,   36.8750,  100.0625,  ...,  112.7500,  202.0000,
         -166.3750]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_nms_rotated",
    """
torch_npu.npu_nms_rotated(dets, scores, iou_threshold, scores_threshold=0, max_output_size=-1, mode=0) -> (Tensor, Tensor)
功能描述
按分数降序选择旋转标注框的子集。

参数说明
dets (Tensor) - shape为[num_boxes, 5]的2D浮点张量
scores (Tensor) - shape为[num_boxes]的1D浮点张量，表示每个框(每行框)对应的一个分数。
iou_threshold (Float) - 表示框与IoU重叠上限阈值的标量。
scores_threshold (Float，默认值为0) - 表示决定何时删除框的分数阈值的标量。
max_output_size (Int，默认值为-1) - 标量整数张量，表示非最大抑制下要选择的最大框数。为-1时即不施加任何约束。
mode (Int，默认值为0) - 指定dets布局类型。如果mode设置为0，则dets的输入值为x、y、w、h和角度。如果mode设置为1，则dets的输入值为x1、y1、x2、y2和角度。
输出说明
selected_index (Tensor) - shape为[M]的1D整数张量，表示从dets张量中选定的index，其中M <= max_output_size。
selected_num (Tensor) - 0D整数张量，表示selected_indices中有效元素的数量。
约束说明
目前不支持mode=1的场景。

示例
>>> dets=torch.randn(100,5).npu()
>>> scores=torch.randn(100).npu()
>>> dets.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> output1, output2 = torch_npu.npu_nms_rotated(dets, scores, 0.2, 0, -1, 1)
>>> output1
tensor([76, 48, 15, 65, 91, 82, 21, 96, 62, 90, 13, 59,  0, 18, 47, 23,  8, 56,
        55, 63, 72, 39, 97, 81, 16, 38, 17, 25, 74, 33, 79, 44, 36, 88, 83, 37,
        64, 45, 54, 41, 22, 28, 98, 40, 30, 20,  1, 86, 69, 57, 43,  9, 42, 27,
        71, 46, 19, 26, 78, 66,  3, 52], device='npu:0', dtype=torch.int32)
>>> output2
tensor([62], device='npu:0', dtype=torch.int32)
"""
)


_add_torch_npu_docstr(
    "npu_nms_v4",
    """
torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)
功能描述
按分数降序选择标注框的子集。

参数说明
boxes (Tensor) - shape为[num_boxes, 4]的2D浮点张量。
scores (Tensor) - shape为[num_boxes]的1D浮点张量，表示每个框(每行框)对应的一个分数。
max_output_size (Scalar) - 表示non-max suppression下要选择的最大框数的标量。
iou_threshold (Tensor) - 0D浮点张量，表示框与IoU重叠上限的阈值。
scores_threshold (Tensor) - 0D浮点张量，表示决定何时删除框的分数阈值。
pad_to_max_output_size (Bool，默认值为False) - 如果为True，则输出的selected_indices将填充为max_output_size长度。
输出说明
selected_indices (Tensor) - shape为[M]的1D整数张量，表示从boxes张量中选定的index，其中M <= max_output_size。
valid_outputs (Tensor) - 0D整数张量，表示selected_indices中有效元素的数量，有效元素首先呈现。
示例
>>> boxes=torch.randn(100,4).npu()
>>> scores=torch.randn(100).npu()
>>> boxes.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> max_output_size = 20
>>> iou_threshold = torch.tensor(0.5).npu()
>>> scores_threshold = torch.tensor(0.3).npu()
>>> npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
>>> npu_output
(tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
        54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
"""
)


_add_torch_npu_docstr(
    "npu_nms_with_mask",
    """
torch_npu.npu_nms_with_mask(input, iou_threshold) -> (Tensor, Tensor, Tensor)
功能描述
生成值0或1，用于nms算子确定有效位。

参数说明
input (Tensor) - 输入张量
iou_threshold (Scalar) - 阈值。如果超过此阈值，则值为1，否则值为0。
输出说明
selected_boxes (Tensor) - shape为[N,5]的2D张量，表示filtered box，包括proposal box和相应的置信度分数。
selected_idx (Tensor) - shape为[N]的1D张量，表示输入建议框的index。
selected_mask (Tensor) - shape为[N]的1D张量，判断输出建议框是否有效。
约束说明
输入box_scores的2nd-dim必须等于8。

示例
>>> input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6], [6.0, 7.0, 8.0, 9.0, 0.4]], dtype=torch.float16).to("npu")
>>> iou_threshold = 0.5
>>> output1, output2, output3, = torch_npu.npu_nms_with_mask(input, iou_threshold)
>>> output1
tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
        [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]], device='npu:0',      dtype=torch.float16)
>>> output2
tensor([0, 1], device='npu:0', dtype=torch.int32)
>>> output3
tensor([1, 1], device='npu:0', dtype=torch.uint8)
"""
)


_add_torch_npu_docstr(
    "npu_normalize_batch",
    """
torch_npu.npu_normalize_batch(self, seq_len, normalize_type=0) -> Tensor
功能描述
执行批量归一化。

参数说明
self (Tensor) - 支持float32数据类型，shape为(n, c, d)。
seq_len (Tensor) - 支持Int32数据类型，shape为(n, )， 表示每批次标准化数据量 。
normalize_type (Int，默认值为0) - 支持 "per_feature"或"all_features"。值为0表示 "per_feature"，值为1表示"all_features"。
示例
>>> a=np.random.uniform(1,10,(2,3,6)).astype(np.float32)
>>> b=np.random.uniform(3,6,(2)).astype(np.int32)
>>> x=torch.from_numpy(a).to("npu")
>>> seqlen=torch.from_numpy(b).to("npu")
>>> out = torch_npu.npu_normalize_batch(x, seqlen, 0)
>>> out
tensor([[[ 1.1496, -0.6685, -0.4812,  1.7611, -0.5187,  0.7571],
        [ 1.1445, -0.4393, -0.7051,  1.0474, -0.2646, -0.1582],
        [ 0.1477,  0.9179, -1.0656, -6.8692, -6.7437,  2.8621]],

        [[-0.6880,  0.1337,  1.3623, -0.8081, -1.2291, -0.9410],
        [ 0.3070,  0.5489, -1.4858,  0.6300,  0.6428,  0.0433],
        [-0.5387,  0.8204, -1.1401,  0.8584, -0.3686,  0.8444]]],
      device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_one_hot",
    """
torch_npu.npu_one_hot(input, num_classes=-1, depth=1, on_value=1, off_value=0) -> Tensor
功能描述
返回一个one-hot张量。input中index表示的位置采用on_value值，而其他所有位置采用off_value的值。

参数说明
input (Tensor) - 任何shape的class值。
num_classes (Int，默认值为-1) - 待填充的轴。
depth (Int，默认值为1) - one_hot维度的深度。
on_value (Scalar，默认值为1) - 当indices[j] == i时输出中的填充值。
off_value (Scalar，默认值为0) - 当indices[j] != i时输出中的填充值。
示例
>>> a=torch.IntTensor([5, 3, 2, 1]).npu()
>>> b=torch_npu.npu_one_hot(a, depth=5)
>>> b
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_pad",
    """
torch_npu.npu_pad(input, paddings) -> Tensor
功能描述
填充张量。

参数说明
input (Tensor) - 输入张量。
paddings (ListInt) - 数据类型：int32、int64。
示例
>>> input = torch.tensor([[20, 20, 10, 10]], dtype=torch.float16).to("npu")
>>> paddings = [1, 1, 1, 1]
>>> output = torch_npu.npu_pad(input, paddings)
>>> output
tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 20., 20., 10., 10.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_ps_roi_pooling",
    """
torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor
功能描述
执行Position Sensitive ROI Pooling。

参数说明
x (Tensor) - 描述特征图的NC1HWC0张量。维度C1必须等于(int(output_dim+15)/C0)) group_size。
rois (Tensor) - shape为[batch, 5, rois_num]的张量，用于描述ROI。每个ROI由五个元素组成：“batch_id”、“x1”、“y1”、“x2”和“y2”，其中“batch_id”表示输入特征图的index，“x1”、“y1”、“x2”，和“y2”必须大于或等于“0.0”。
spatial_scale (Float32) - 将输入坐标映射到ROI坐标的缩放系数。
group_size (Int32) - 指定用于编码position-sensitive评分图的组数。该值必须在(0,128)范围内。
output_dim (Int32) - 指定输出通道数。必须大于0。
示例
>>> roi = torch.tensor([[[1], [2], [3], [4], [5]],
                        [[6], [7], [8], [9], [10]]], dtype = torch.float16).npu()
>>> x = torch.tensor([[[[ 1]], [[ 2]], [[ 3]], [[ 4]],
                      [[ 5]], [[ 6]], [[ 7]], [[ 8]]],
                      [[[ 9]], [[10]], [[11]], [[12]],
                      [[13]], [[14]], [[15]], [[16]]]], dtype = torch.float16).npu()
>>> out = torch_npu.npu_ps_roi_pooling(x, roi, 0.5, 2, 2)
>>> out
tensor([[[[0., 0.],
          [0., 0.]],
        [[0., 0.],
          [0., 0.]]],
        [[[0., 0.],
          [0., 0.]],
        [[0., 0.],
          [0., 0.]]]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_ptiou",
    """
torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor
功能描述
根据ground-truth和预测区域计算交并比(IoU)或前景交叉比(IoF)。

参数说明
bboxes (Tensor) - 输入张量。
gtboxes (Tensor) - 输入张量。
mode (Int，默认值为0) - 0为IoU模式，1为IoF模式。
示例
>>> bboxes = torch.tensor([[0, 0, 10, 10],
                           [10, 10, 20, 20],
                           [32, 32, 38, 42]], dtype=torch.float16).to("npu")
>>> gtboxes = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float16).to("npu")
>>> output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
>>> output_iou
tensor([[0.4985, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
       [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_random_choice_with_mask",
    """
torch_npu.npu_random_choice_with_mask(x, count=256, seed=0, seed2=0) -> (Tensor, Tensor)
功能描述
混洗非零元素的index。

参数说明
x (Tensor) - 输入张量。
count (Int，默认值为256) - 输出计数。如果值为0，则输出所有非零元素。
seed (Int，默认值为0) - 数据类型：int32，int64。
seed2 (Int，默认值为2) - 数据类型：int32，int64。
输出说明
y (Tensor) - 2D张量, 非零元素的index。
mask (Tensor) - 1D张量, 确定对应index是否有效。
示例
>>> x = torch.tensor([1, 0, 1, 0], dtype=torch.bool).to("npu")
>>> result, mask = torch_npu.npu_random_choice_with_mask(x, 2, 1, 0)
>>> result
tensor([[0], [2]], device='npu:0', dtype=torch.int32)
>>> mask
tensor([True, True], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_reshape",
    """
torch_npu.npu_reshape(self, shape, bool can_refresh=False) -> Tensor
功能描述
reshape张量。仅更改张量shape，其数据不变。

参数说明
self (Tensor) - 输入张量。
shape (ListInt) - 定义输出张量的shape。
can_refresh (Bool，默认值为False) - 是否就地刷新reshape。
约束说明
该运算符不能被aclopExecute API直接调用。

示例
>>> a=torch.rand(2,8).npu()
>>> out=torch_npu.npu_reshape(a,(4,4))
>>> out
tensor([[0.6657, 0.9857, 0.7614, 0.4368],
        [0.3761, 0.4397, 0.8609, 0.5544],
        [0.7002, 0.3063, 0.9279, 0.5085],
        [0.1009, 0.7133, 0.8118, 0.6193]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_rms_norm",
    """
torch_npu.npu_rms_norm(Tensor self, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)
功能描述
RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。


参数说明
self：Tensor类型，支持float16、bfloat16、float32，输入shape支持2-8维。
gamma：Tensor类型，数据类型需要和self保持一致，输入shape支持2-8维，通常为self的最后一维。
epsilon：float数据类型，用于防止除0错误。
输出说明
共两个输出，格式为： (Tensor, Tensor)

第1个输出为Tensor，计算公式的最终输出y；

第2个输出为Tensor，rms_norm的reverse rms中间结果，用于反向计算。

约束说明
输入数据类型仅支持float16、bfloat16和float32。

示例
import torch
import torch_npu
x = torch.randn(24, 1, 128).npu()
w = torch.randn(128).npu()
​
out1 = torch_npu.npu_rms_norm(x, w, epsilon=1e-5)[0]
print(out1)
tensor([[[-0.1123,  0.3398,  0.0986,  ..., -2.1250, -0.8477, -0.3418]],
​
        [[-0.0591,  0.3184, -0.5000,  ...,  1.0312, -1.1719, -0.1621]],
​
        [[-0.1445,  0.3828, -0.3438,  ..., -0.9102, -0.5703,  0.0073]],
​
        ...,
​
        [[-0.1631, -0.3477,  0.4297,  ...,  0.9219,  0.1621,  0.3125]],
​
        [[-0.1387,  0.0815,  0.0967,  ...,  1.7109,  0.1455, -0.1406]],
​
        [[ 0.0698,  1.3438, -0.0127,  ..., -2.2656, -0.4473,  0.3281]]],
       device='npu:0')

"""
)


_add_torch_npu_docstr(
    "npu_roi_align",
    """
torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor
功能描述
从特征图中获取ROI特征矩阵。自定义FasterRcnn算子。

参数说明
features (Tensor) - 5HD张量
rois (Tensor) - ROI位置，shape为(N, 5)的2D张量。“N”表示ROI的数量，“5”表示ROI所在图像的index，分别为“x0”、“y0”、“x1”和“y1”。
spatial_scale (Float32) - 指定“features”与原始图像的缩放比率。
pooled_height (Int32) - 指定H维度。
pooled_width (Int32) - 指定W维度。
sample_num (Int32，默认值为2) - 指定每次输出的水平和垂直采样频率。若此属性设置为0，则采样频率等于“rois”的向上取整值(一个浮点数)。
roi_end_mode (Int32，默认值为1)
示例
>>> x = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                            [7, 8, 9, 10, 11, 12],
                            [13, 14, 15, 16, 17, 18],
                            [19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35, 36]]]]).npu()
>>> rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
>>> out = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
>>> out
tensor([[[[ 4.5000,  6.5000,  8.5000],
          [16.5000, 18.5000, 20.5000],
          [28.5000, 30.5000, 32.5000]]]], device='npu:0')

"""
)


_add_torch_npu_docstr(
    "npu_rotary_mul",
    """
torch_npu.npu_rotary_mul(Tensor input, Tensor r1, Tensor r2, str rotary_mode='half') -> Tensor
功能描述
实现RotaryEmbedding旋转位置编码。支持FakeTensor模式。
    half模式：
    x1, x2 = torch.chunk(input, 2, -1)
    x_new = torch.cat((-x2, x1), dim=-1)
    output = r1 * input + r2 * x_new
    interleave模式：
    x1 = input[..., ::2]
    x2 = input[..., 1::2]
    x_new = rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ...(d two)", two=2)
    output = r1 * input + r2 * x_new
参数说明
input：必选输入，4维Tensor，数据类型float16, bfloat16, float32
cos: 必选输入，4维Tensor，数据类型float16, bfloat16, float32
sin: 必选输入，4维Tensor，数据类型float16, bfloat16, float32
rotary_mode: 可选属性，数据类型string，用于选择计算模式，支持'half'、'interleave'两种模式。缺省为half。
约束说明
jit_compile=False场景：
    half模式：
    input: layout支持: BNSD、BSND、SBND; D < 896，且为2的倍数; B, N < 1000; 当需要计算cos/sin的反向梯度时，B*N <= 1024
    r1: 数据范围：[-1, 1]; 对应input layout的支持情况：
                            input为BNSD: 11SD、B1SD、BNSD;
                            input为BSND: 1S1D、BS1D、BSND;
                            input为SBND: S11D、SB1D、SBND.
    r2: 同r1
    half模式下，当输入layout是BNSD，且D为非32Bytes对齐时，建议不使用该融合算子（模型启动脚本中不开启--use-fused-rotary-pos-emb选项），否则可能出现性能下降。
    interleave模式：
    input: layout支持: BNSD、BSND、SBND; B * N < 1000; D < 896, 且D为2的倍数;
    r1: 数据范围：[-1, 1]; 对应input layout的支持情况：
                            input为BNSD: 11SD;
                            input为BSND: 1S1D;
                            input为SBND: S11D.
    r2: 同r1
    支持Atlas A2训练系列产品，Atlas A3训练系列产品。
jit_compile=True场景：
    仅支持rotary_mode为half模式，且r1/r2 layout一般为11SD、1S1D、S11D。
    shape要求输入为4维，其中B维度和N维度数值需小于等于1000，D维度数值为128。
    广播场景下，广播轴的总数据量不能超过1024
    支持Atlas训练系列产品，Atlas A2训练系列产品, Atlas 推理系列产品。

示例
    >>>x = torch.rand(2, 2, 5, 128).npu()
    >>>r1 = torch.rand(1, 2, 1, 128).npu()
    >>>r2 = torch.rand(1, 2, 1, 128).npu()
    >>>out = torch_npu.npu_rotary_mul(x, r1, r2)
"""
)


_add_torch_npu_docstr(
    "npu_mrope",
    """
torch_npu.npu_mrope(Tensor positions, Tensor query, Tensor key, Tensor cos_sin_cache, int head_size, *, int[]? mrope_section, str? rotary_mode='half') -> (Tensor, Tensor)
功能描述
实现旋转位置编码。通过传入cos和sin的cache执行旋转位置编码计算。

参数说明
positions (Tensor): 输入索引，用于选取位置编码张量。要求是一个维度为1D或2D的Tensor，shape为 (numTokens)或(3, numTokens)，1D维度输入是rope模式，2D维度输入是mrope模式。numTokens表示一个序列中的token数量。支持非连续的Tensor，支持空Tensor。数据类型支持INT32、INT64，数据格式支持ND。
queryIn (Tensor): 要执行旋转位置编码的第一个张量，维度为2D的Tensor，shape为 (numTokens, numQHeads*headSize)。numQHeads表示query的注意力头数量。headSize表示每个注意力头维度大小。支持非连续的Tensor，支持空Tensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式支持ND。
keyIn (Tensor): 要执行旋转位置编码的第二个张量，维度为2D的Tensor，shape为 (numTokens, numKHeads*headSize)。numKHeads表示key的注意力头数量。headSize表示每个注意力头维度大小。支持非连续的Tensor，支持空Tensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式支持ND。
cosSinCache (Tensor): 参与计算的位置编码张量，要求shape为一个2D的(maxSeqLen, rotaryDim)。maxSeqLen表示模型处理的序列的最大长度。rotaryDim表示旋转位置嵌入的维度大小。支持非连续的Tensor，支持空Tensor。数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式支持ND。
headSize(int): 表示每个注意力头维度大小。数据类型int64。
mropeSection(int[]): 可选参数，multimodal section配置，用于整合输入的位置编码张量信息，输入mropeSection属性表示使能mrope模式。默认值为不使能mrope模式(即rope模式)输入为[0, 0, 0]。
rotary_mode(str): 可选参数，旋转模式，'half'表示rotate_half(GPT-NeoX style)计算模式，'interleave'表示rotate_interleaved(GPT-J style)计算模式。默认值为'half'。

约束说明
queryIn、keyIn、cosSinCache只支持2维shape输入。
当输入是BFLOAT16或FLOAT16时，rotary_dim要求是32的倍数，当输入是FLOAT32时，rotary_dim要求是16的倍数。
当输入tensor positions中值域超过cosSinCache的0维maxSeqLen，会有越界报错。
mrope模式下，mropeSection输入mropeSection[0]+mropeSection[1]+mropeSection[2]==rotary_dim/2

示例
>>> num_tokens = 8
>>> num_q_heads = 32
>>> num_kv_heads = num_q_heads
>>> head_size = 128
>>> max_seq_len = num_tokens
>>> rotary_dim = head_size
>>> positions = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1).npu()
>>> query = torch.rand(num_tokens, num_q_heads*head_size, dtype=torch.float32).npu()
>>> key = torch.rand(num_tokens, num_kv_heads*head_size, dtype=torch.float32).npu()
>>> cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()
>>> rotary_mode = 'half'
>>> mrope_section = [16, 24, 24]
>>> query_out, key_out = torch_npu.npu_mrope(positions, query, key, cos_sin_cache, head_size, mrope_section=mrope_section, rotary_mode=rotary_mode)
"""
)


_add_torch_npu_docstr(
    "npu_rotated_box_decode",
    """
torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight) -> Tensor
功能描述
旋转标注框编码。

参数说明
anchor_box (Tensor) - shape为(B,5,N)的3D输入张量，表示锚点框。“B”表示批处理大小数量，“N”表示标注框数量，值“5”表示“x0”、“x1”、“y0”、“y1”和“angle”。
deltas (Tensor) - shape为(B,5,N)数据类型为float32 (float16)的3D张量。
weight (Tensor，默认值为[1.0, 1.0, 1.0, 1.0, 1.0]) - “x0”、“x1”、“y0”、“y1”和“angle”的浮点列表。
示例
>>> anchor_boxes = torch.tensor([[[4.137],[33.72],[29.4], [54.06], [41.28]]], dtype=torch.float16).to("npu")
    >>> deltas = torch.tensor([[[0.0244], [-1.992], [0.2109], [0.315], [-37.25]]], dtype=torch.float16).to("npu")
    >>> weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
    >>> out = torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight)
    >>> out
    tensor([[[  1.7861],
            [-10.5781],
            [ 33.0000],
            [ 17.2969],
            [-88.4375]]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_rotated_box_encode",
    """
torch_npu.npu_rotated_box_encode(anchor_box, gt_bboxes, weight) -> Tensor
功能描述
旋转标注框编码。

参数说明
anchor_box (Tensor) - shape为(B,5,N)的3D输入张量，表示锚点框。“B”表示批处理大小数量，“N”表示标注框数量，值“5”表示“x0”、“x1”、“y0”、“y1”和“angle”。
gt_bboxes (Tensor) - shape为(B,5,N)数据类型为float32 (float16)的3D张量。
weight (Tensor，默认值为[1.0, 1.0, 1.0, 1.0, 1.0]) - “x0”、“x1”、“y0”、“y1”和“angle”的浮点列表。
示例
>>> anchor_boxes = torch.tensor([[[30.69], [32.6], [45.94], [59.88], [-44.53]]], dtype=torch.float16).to("npu")
    >>> gt_bboxes = torch.tensor([[[30.44], [18.72], [33.22], [45.56], [8.5]]], dtype=torch.float16).to("npu")
    >>> weight = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float16).npu()
    >>> out = torch_npu.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)
    >>> out
    tensor([[[-0.4253],
            [-0.5166],
            [-1.7021],
            [-0.0162],
            [ 1.1328]]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_rotated_iou",
    """
torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,v_threshold=0.0, e_threshold=0.0) -> Tensor
功能描述
计算旋转框的IoU。

参数说明
self (Tensor) - 梯度增量数据，shape为(B, 5, N)数据类型为float32的3D张量。
query_boxes (Tensor) - 标注框，shape为(B, 5, K) 数据类型为float32的3D张量。
trans (Bool，默认值为False) - 值为True表示“xyxyt”，值为False表示“xywht”。
is_cross (Bool，默认值为True) - 值为True时表示交叉计算，为False时表示一对一计算。
mode (Int，默认值为0) - 计算模式，取值为0或1。0表示IoU，1表示IoF。
v_threshold (Float，可选，默认值为0.0) - provide condition relaxation for intersection calculation.
e_threshold (Float，可选，默认值为0.0) - provide condition relaxation for intersection calculation.
示例
>>> a=np.random.uniform(0,1,(2,2,5)).astype(np.float16)
>>> b=np.random.uniform(0,1,(2,3,5)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
>>> output
tensor([[[3.3325e-01, 1.0162e-01],
        [1.0162e-01, 1.0000e+00]],

        [[0.0000e+00, 0.0000e+00],
        [0.0000e+00, 5.9605e-08]]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_rotated_overlaps",
    """
torch_npu.npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor
功能描述
计算旋转框的重叠面积。

参数说明
self (Tensor) -梯度增量数据，shape为(B, 5, N)数据类型为float32的3D张量。
query_boxes (Tensor) - 标注框，shape为(B, 5, K) 数据类型为float32的3D张量。
trans (Bool，默认值为False) - 值为True表示“xyxyt”，值为False表示“xywht”。
示例
>>> a=np.random.uniform(0,1,(1,3,5)).astype(np.float16)
>>> b=np.random.uniform(0,1,(1,2,5)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_rotated_overlaps(box1, box2, trans=False)
>>> output
tensor([[[0.0000, 0.1562, 0.0000],
        [0.1562, 0.3713, 0.0611],
        [0.0000, 0.0611, 0.0000]]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_scaled_masked_softmax",
    """
torch_npu.npu_scaled_masked_softmax(x, mask, scale=1.0, fixed_triu_mask=False) -> Tensor
功能描述
计算输入张量x缩放并按照mask遮蔽后的Softmax结果。

参数说明
x(Tensor)- 输入的logits。支持数据类型：float16、float32、bfloat16。支持格式：[ND，FRACTAL_NZ]。
mask(Tensor)- 输入的掩码。支持数据类型：bool。支持格式：[ND，FRACTAL_NZ]。
scale(float，默认值为1.0)- x的缩放系数。
fixed_triu_mask(bool，默认值为False)- 是否使用自动生成的上三角bool掩码。
约束说明
当前输入x的shape，只支持转为[NCHW]格式后，H和W轴长度大于等于32、小于等于4096、且能被32整除的场景。
输入mask的shape，必须能被broadcast成x的shape。
示例
>>> import torch
>>> import torch_npu
>>>
>>> shape = [4, 4, 2048, 2048]
>>> x = torch.rand(shape).npu()
>>> mask = torch.zeros_like(x).bool()
>>> scale = 1.0
>>> fixed_triu_mask = False
>>>
>>> output = torch_npu.npu_scaled_masked_softmax(x, mask, scale, fixed_triu_mask)
>>> output.shape
torch.size([4, 4, 2048, 2048])
"""
)


_add_torch_npu_docstr(
    "npu_scatter",
    """
torch_npu.npu_scatter(self, indices, updates, dim) -> Tensor
功能描述
使用dim对scatter结果进行计数。类似于torch.scatter，优化NPU设备实现。

参数说明
self (Tensor) - 输入张量。
indices (Tensor) - 待scatter的元素index，可以为空，也可以与src有相同的维数。当为空时，操作返回“self unchanged”。
updates (Tensor) - 待scatter的源元素。
dim (Int) - 要进行index的轴。

支持的型号:
Atlas 训练系列产品

示例
>>> input    = torch.tensor([[1.6279, 0.1226], [0.9041, 1.0980]]).npu()
>>> input
tensor([[1.6279, 0.1226],
        [0.9041, 1.0980]], device='npu:0')
>>> indices  = torch.tensor([0, 1],dtype=torch.int32).npu()
>>> indices
tensor([0, 1], device='npu:0', dtype=torch.int32)
>>> updates  = torch.tensor([-1.1993, -1.5247]).npu()
>>> updates
tensor([-1.1993, -1.5247], device='npu:0')
>>> dim = 0
>>> output = torch_npu.npu_scatter(input, indices, updates, dim)
>>> output
tensor([[-1.1993,  0.1226],
        [ 0.9041, -1.5247]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_sign_bits_pack",
    """
torch_npu.npu_sign_bits_pack(Tensor self, int size) -> Tensor
功能描述
将float类型1位Adam打包为uint8。

参数说明
x(Tensor) - 1D float张量。
size(Int) - reshape时输出张量的第一个维度。
约束说明
Size可被float打包的输出整除。如果x的size可被8整除，则输出的size为(size of x)/8；否则，输出的size为(size of x // 8) + 1。将在小端位置添加-1浮点值以填充可整除性。Atlas 训练系列产品支持float32和float16类型输入。Atlas 推理系列产品(Ascend 310P处理器)支持float32和float16类型输入。Atlas 200/300/500 推理产品仅支持float16类型输入。

示例
    >>>a = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2],dtype=torch.float32).npu()
    >>>b = torch_npu.npu_sign_bits_pack(a, 2)
    >>>b
    >>>tensor([[159],[15]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_sign_bits_unpack",
    """
torch_npu.npu_sign_bits_unpack(x, size, dtype) -> Tensor
功能描述
将uint8类型1位Adam拆包为float。

参数说明
x(Tensor) - 1D uint8张量。
size(Int) - reshape时输出张量的第一个维度。
dtype(torch.dtype) - 值为1设置输出类型为float16，值为0设置输出类型为float32。
约束说明
Size可被uint8s拆包的输出整除。输出大小为(size of x) * 8。

示例
    >>>a = torch.tensor([159, 15], dtype=torch.uint8).npu()
    >>>b = torch_npu.npu_sign_bits_unpack(a, 2, torch.float32)
    >>>b
    >>>tensor([[1., 1., 1., 1., 1., -1., -1., 1.],
    >>>[1., 1., 1., 1., -1., -1., -1., -1.]], device='npu:0')
(binary form of 159 is ob00001111)
"""
)


_add_torch_npu_docstr(
    "npu_silu",
    """
torch_npu.npu_silu(self) -> Tensor
功能描述
计算self的Swish。

参数说明
self (Tensor) - 数据类型：float16、float32

示例
>>> a=torch.rand(2,8).npu()
>>> output = torch_npu.npu_silu(a)
>>> output
tensor([[0.4397, 0.7178, 0.5190, 0.2654, 0.2230, 0.2674, 0.6051, 0.3522],
        [0.4679, 0.1764, 0.6650, 0.3175, 0.0530, 0.4787, 0.5621, 0.4026]],
       device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_slice",
    """
torch_npu.npu_slice(self, offsets, size) -> Tensor
功能描述
从张量中提取切片。

参数说明
self (Tensor) - 输入张量。
offsets (ListInt) - 数据类型：int32，int64。
size (ListInt) - 数据类型：int32，int64。
示例
>>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16).to("npu")
>>> offsets = [0, 0]
>>> size = [2, 2]
>>> output = torch_npu.npu_slice(input, offsets, size)
>>> output
tensor([[1., 2.],
        [6., 7.]], device='npu:0', dtype=torch.float16)
"""
)


_add_torch_npu_docstr(
    "npu_softmax_cross_entropy_with_logits",
    """
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
功能描述
计算softmax的交叉熵cost。

参数说明
features (Tensor) - 张量，一个“batch_size * num_classes”矩阵。
labels (Tensor) - 与“features”同类型的张量。一个“batch_size * num_classes”矩阵。
"""
)


_add_torch_npu_docstr(
    "npu_sort_v2",
    """
torch_npu.npu_sort_v2(self, dim=-1, descending=False, out=None) -> Tensor
功能描述
沿给定维度，按无index值对输入张量元素进行升序排序。若dim未设置，则选择输入的最后一个维度。如果descending为True，则元素将按值降序排序。

参数说明
self (Tensor) - 输入张量。
dim (Int, 可选,默认值为-1) - 进行排序的维度。
descending (Bool, 可选，默认值为None) - 排序顺序控制(升序或降序)。
约束说明
目前仅支持输入的最后一个维度(dim=-1)。

示例
>>> x = torch.randn(3, 4).npu()
>>> x
tensor([[-0.0067,  1.7790,  0.5031, -1.7217],
        [ 1.1685, -1.0486, -0.2938,  1.3241],
        [ 0.1880, -2.7447,  1.3976,  0.7380]], device='npu:0')
>>> sorted_x = torch_npu.npu_sort_v2(x)
>>> sorted_x
tensor([[-1.7217, -0.0067,  0.5029,  1.7793],
        [-1.0488, -0.2937,  1.1689,  1.3242],
        [-2.7441,  0.1880,  0.7378,  1.3975]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_stride_add",
    """
torch_npu.npu_stride_add(x1, x2, offset1, offset2, c1_len) -> Tensor
功能描述
添加两个张量的partial values，格式为NC1HWC0。

参数说明
x1 (Tensor) - 5HD张量。
x2 (Tensor) - 与“x1”类型相同shape相同(C1值除外)的张量。
offset1 (Scalar) - “x1”中C1的offset value。
offset2 (Scalar) - “x2”中C1的offset value。
c1_len (Scalar) - “y”的C1 len。该值必须小于“x1”和“x2”中C1与offset的差值。
示例
>>> a=torch.tensor([[[[[1.]]]]]).npu()
>>> b=torch_npu.npu_stride_add(a, a, 0, 0, 1)
>>> b
tensor([[[[[2.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]],
        [[[0.]]]]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "npu_transpose",
    """
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
功能描述
返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

参数说明
self (Tensor) - 输入张量。
perm (ListInt) - 对应维度排列。
require_contiguous(Bool，默认值为True) - 用户是否需要对输入Tensor做转连续。设置为False时，表示不对输入Tensor做转连续。用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。
示例
>>> x = torch.randn(2, 3, 5).npu()
>>> x.shape
torch.Size([2, 3, 5])
>>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> x1.shape
torch.Size([5, 2, 3])
"""
)


_add_torch_npu_docstr(
    "npu_yolo_boxes_encode",
    """
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
功能描述
返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

参数说明
self (Tensor) - 输入张量。
perm (ListInt) - 对应维度排列。
require_contiguous(Bool，默认值为True) - 用户是否显式指定npu_contiguous算子适配需要对输入Tensor做转连续。默认为False，低性能模式。用户明确知道输入Tensor为连续Tensor或转置Tensor时，才能设置为True使用高性能模式。
示例
>>> x = torch.randn(2, 3, 5).npu()
>>> x.shape
torch.Size([2, 3, 5])
>>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> x1.shape
torch.Size([5, 2, 3])
>>> x2 = x.npu_transpose(2, 0, 1)
>>> x2.shape
torch.Size([5, 2, 3])
"""
)


_add_torch_npu_docstr(
    "one_",
    """
torch_npu.one_(self) -> Tensor

用1填充self张量。

参数解释：
self (Tensor) - 输入张量。
约束条件：
无

示例：
>>> x = torch.rand(2, 3).npu()
>>> x
tensor([[0.6072, 0.9726, 0.3475],
        [0.3717, 0.6135, 0.6788]], device='npu:0')
>>> torch_npu.one_(x)
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='npu:0')
"""
)

_add_torch_npu_docstr(
    "npu_swiglu",
    """
接口原型：
torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)

功能描述：
提供swiglu的激活函数。
公式如下：
outputs = swiglu\(x, dim = -1) = swish(A) * B = A * sigmoid(A) * B
“x”是输入Tensor。
“dim”是切分维度，默认为-1。
“A”和“B”是x沿dim维度切分的Tensor。

参数说明：
“x”：Tensor类型，shape支持1-8维，dtype支持FP32、FP16或BF16类型。
“dim”：Int类型，默认为-1。

输出说明：
输出为Tensor，计算公式的最终输出outputs。

支持的型号:
Atlas A2 训练系列产品

调用示例：
import torch
import torch_npu
input_tensor = torch.randn(2, 32, 6, 6)
output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
"""
)

_add_torch_npu_docstr(
    "npu_trans_quant_param",
    """
功能描述:
完成量化计算参数scale数据类型的转换. 

接口原型:
torch_npu.npu_trans_quant_param(Tensor scale, Tensor? offset=None, int? round_mode=0) -> Tensor

参数说明:
scale: Tensor类型, 数据类型支持float32, 数据格式支持ND, shape是1维(t,)或者2维(1, n). 其中t=1或n, 其中n与matmul计算中的右矩阵中的n一致. 
offset: Tensor类型, 可选参数. 数据类型支持float32, 数据格式支持ND, shape是1维(t,)或者2维(1, n). t=1或n, 其中n与matmul计算中的右矩阵中的n一致. 
round_mode: torch.int8类型，用于指定FP32填充到FP19的模式，可选参数。支持的枚举值为0和1。0表示截断填充，1表示R_INT模式。默认为0。

输出说明:
一个Tensor类型的输出, 代表trans_quant_param的计算结果. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
传入的scale或out不能为空. 
scale、offset或out的数据类型和数据格式需要在支持的范围之内. 
scale、offset的shape需要为1维(t,)或者2维(1, n). 其中t=1或n, 其中n与matmul计算中的右矩阵中的n一致. 
当scale的shape为两维(1, n)时, scale和offset的shape需要保持一致, 且输出shape也为(1, n). 

支持的PyTorch版本
PyTorch 2.5
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import logging
import os
scale = torch.randn(16, dtype=torch.float32)
offset = torch.randn(16, dtype=torch.float32)
npu_out = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu(), round_mode=0)

图模式调用
图模式下, npu_trans_quant_param计算出的结果tensor为uint64数据类型. 由于torch不支持该数据类型, 需要搭配其他接口使用, 如示例代码中的npu_quant_matmul.
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
import numpy as np
os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale, offset, bias):
        scale_1 = torch_npu.npu_trans_quant_param(scale, offset, round_mode=0)
        return torch_npu.npu_quant_matmul(x1, x2, scale_1, offset=offset, bias=bias)
cpu_model = MyModel()
model = cpu_model.npu()
cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
scale = torch.randn(1, dtype=torch.float32)
offset = torch.randn(1, dtype=torch.float32)
bias = torch.randint(-1,1, (15, 1, 128), dtype=torch.int32)
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset.npu(), bias.npu())
"""
)

_add_torch_npu_docstr(
    "npu_dynamic_quant",
    """
功能描述:
算子功能: 对输入的张量进行per-token对称动态量化. 
如果是MoE(Mixture of Experts, 混合专家模型)场景, 会引入group_index, smooth_scales中包含多组smooth向量, 按group_index中的数值作用到x的不同行上. 具体的, 假如x包含m个token, smooth_scales有n行, smooth_scales[0]会作用到x[0:group_index[0]]上, smooth_scales[i]会作用到x[group_index[i-1]: group_index[i]]上, i=1, 2, ..., n-1. 
计算公式: 
如果smooth_scales不存在: 
scale=rowMax(abs(x))/DTYPE_MAX
y=round(x/scale)
如果smooth_scales存在: 
scale=rowMax(abs(x×smooth_scales))/DTYPE_MAX
y=round(x×smooth_scales/scale)
owMax表示求一行的最大值, DTYPE_MAX表示常量, 是y输出对应的数据类型的最大值. 

接口原型:
torch_npu.npu_dynamic_quant(Tensor x, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None) ->(Tensor, Tensor)

参数说明:
x: Tensor类型, 需要进行量化的源数据张量, 必选输入, 数据类型支持torch.float16、torch.bfloat16, 数据格式支持ND, 支持非连续的Tensor. 输入x的维度必须大于1. 进行int4量化时, 要求x形状的最后一维是8的整数倍. 
smooth_scales: Tensor类型, 对x进行scales的张量, 可选输入, 数据类型支持torch.float16、torch.bfloat16, 数据格式支持ND, 支持非连续的Tensor. shape必须是1维, 和x的最后一维相等. 
单算子模式: smooth_scales的dtype必须和x保持一致. 
group_index: Tensor类型, 对smooth_scales进行分组的下标, 可选输入, 仅在MoE场景下生效. 数据类型支持int32, 数据格式支持ND, 支持非连续的Tensor. 
dst_type: ScalarType类型, 指定量化输出的类型, 可选输入, 传None时当做torch.int8处理. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持取值torch.int8、torch.quint4x2. 
Atlas A3 训练系列产品: 支持取值torch.int8、torch.quint4x2. 

输出说明：
y: 量化后的输出Tensor, 数据类型由dst_type指定. 当dst_type是torch.quint4x2时, y的数据类型为int32, 形状最后一维为x最后一维除以8, 其余维度与x一致, 每个int32元素包含8个int4结果. 其他场景下y形状与输入x一致, 数据类型由dst_type指定. 
scale: Tensor类型, 非对称动态量化过程中计算出的缩放系数, 数据类型为float32, 形状为x的形状剔除最后一维. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
该接口仅在如下产品支持MoE场景. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
使用smooth_scales时: 
若不使用group_index, smooth_scales必须是一维Tensor, 元素数量与x的最后一维大小一致. 
若使用group_index, smooth_scales必须是二维Tensor, 第二维元素数量与x的最后一维大小一致, group_index必须是一维数组, 元素数量与smooth_scales第一维一致. group_index中的元素必须是单调递增的, 其最后一个元素的值, 应等于x的元素数量除以x的最后一个维度. 

支持的PyTorch版本
PyTorch 2.5
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
只有一个输入x
import torch
import torch_npu
 
x = torch.rand((3, 3), dtype = torch.float16).to("npu")
output, scale = torch_npu.npu_dynamic_quant(x)
print(output)
print(scale)
使用smooth_scales输入
import torch
import torch_npu
 
x = torch.rand((3, 3), dtype = torch.float16).to("npu")
smooth_scales = torch.rand((3,), dtype = torch.float16).to("npu")
output, scale = torch_npu.npu_dynamic_quant(x, smooth_scales=smooth_scales)
print(output)
print(scale)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
torch_npu.npu.set_compile_mode(jit_compile=True)
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

device=torch.device(f'npu:0')

torch_npu.npu.set_device(device)

class DynamicQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
        out, scale = torch_npu.npu_dynamic_quant(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
        return out, scale

x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
dynamic_quant_model = DynamicQuantModel().npu()
dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
out, scale = dynamic_quant_model(x, smooth_scales=smooth_scales)
print(out)
print(scale)
"""
)

_add_torch_npu_docstr(
    "npu_dynamic_quant_asymmetric",
    """
功能描述:
算子功能: 对输入的张量进行per-token非对称动态量化. 其中输入的最后一个维度对应一个token, 每个token作为一组进行量化. 
计算公式: 假设待量化张量为x, 
scale=(rowMax(x)-rowMin(x))/(DST_MAX-DST_MIN)
offset=DST_MAX-rowMax(x)/scale
y=round(x/scale+offset)
owMax、rowMin代表按行取最大值、按行取最小值, 此处的“行”对应x最后一个维度的数据, 即一个token. 
DST_MAX、DST_MIN分别对应量化后的最大值和最小值, 在进行int8量化时, 二者分别对应+127、-128, 进行int4量化时, 分别对应+7、-8
若使用smooth quant, 会引入smooth_scales输入, 其形状与x最后一个维度大小一致, 在进行量化前, 会先令x乘以smooth_scales, 再按上述公式进行量化
若使用smooth quant, MoE(Mixture of Experts, 混合专家模型)场景下会引入smooth_scales和group_index, 此时smooth_scales中包含多组smooth向量, 按group_index中的数值作用到x的不同行上. 具体的, 假如x包含m个token, smooth_scales有n行, smooth_scales[0]会作用到x[0:group_index[0]]上, smooth_scales[i]会作用到x[group_index[i-1]: group_index[i]]上, i=[1, 2, ..., n-1]. 

接口原型:
torch_npu.npu_dynamic_quant_asymmetric(Tensor x, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None) -> (Tensor, Tensor, Tensor)

参数说明:
x: Tensor类型, 需要进行量化的源数据张量, 必选输入, 数据类型支持float16、bfloat16, 数据格式支持ND, 支持非连续的Tensor. 输入x的维度必须大于1. 进行int4量化时, 要求x形状的最后一维是8的整数倍. 
smooth_scales: Tensor类型, 对x进行平滑缩放的张量, 可选输入, 数据类型需要与x保持一致, 数据格式支持ND, 支持非连续的Tensor. 
group_index: Tensor类型, 在MoE场景下, 对smooth_scales进行分组的下标, 可选输入, 数据类型支持int32, 数据格式支持ND, 支持非连续的Tensor. 
dst_type: ScalarType类型, 用于选择进行int8/int4量化, 可选输入, 输入值只能是torch.int8和torch.quint4x2, 默认为int8量化. 

输出说明：
y: 量化后的输出Tensor, 在进行int8量化时, y的数据类型为int8, 形状与x一致; 在进行int4量化时, y的数据类型为int32, 形状最后一维为x最后一维除以8, 其余维度与x一致, 每个int32元素包含8个int4结果. 
scale: Tensor类型, 非对称动态量化过程中计算出的缩放系数, 数据类型为float32, 形状为x的形状剔除最后一维. 
offset: Tensor类型, 非对称动态量化过程中计算出的偏移系数, 数据类型为float32, 形状为x的形状剔除最后一维. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
使用可选输入smooth_scales、group_index、dst_type时, 必须使用关键字传参. 
使用smooth_scales时: 
若不使用group_index, smooth_scales必须是一维Tensor, 元素数量与x的最后一维大小一致. 
若使用group_index, smooth_scales必须是二维Tensor, 第二维元素数量与x的最后一维大小一致, group_index必须是一维数组, 元素数量与smooth_scales第一维一致. group_index中的元素必须是单调递增的, 其最后一个元素的值, 应等于x的元素数量除以x的最后一个维度. 

支持的PyTorch版本
PyTorch2.5
PyTorch2.4
PyTorch2.3
PyTorch2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
只有一个输入x, 进行int8量化
import torch
import torch_npu
x = torch.rand((3, 8), dtype=torch.half).npu()
y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
print(y, scale, offset)
只有一个输入x, 进行int4量化
import torch
import torch_npu
x = torch.rand((3, 8), dtype=torch.half).npu()
y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
print(y, scale, offset)
使用smooth_scales输入, 非MoE场景(不使用group_index), 进行int8量化
import torch
import torch_npu
x = torch.rand((3, 8), dtype=torch.half).npu()
smooth_scales = torch.rand((8,), dtype=torch.half).npu()
y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
print(y, scale, offset)
使用smooth_scales输入, MoE场景(使用group_index), 进行int8量化
import torch
import torch_npu
x = torch.rand((3, 8), dtype=torch.half).npu()
smooth_scales = torch.rand((2, 8), dtype=torch.half).npu()
group_index = torch.Tensor([1, 3]).to(torch.int32).npu()
y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales, group_index=group_index)
print(y, scale, offset)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
torch_npu.npu.set_compile_mode(jit_compile=True)
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

device=torch.device(f'npu:4')

torch_npu.npu.set_device(device)

class DynamicQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
        out, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
        return out, scale, offset

x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
dynamic_quant_model = DynamicQuantModel().npu()
dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
out, scale, offset = dynamic_quant_model(x, smooth_scales=smooth_scales)
print(out)
print(scale)
print(offset)
"""
)

_add_torch_npu_docstr(
    "npu_quant_matmul_reduce_sum",
    """
功能描述：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出

计算公式：
  out = torch.zeros(m, n)
  for i in range(batch):
    out += (x1[i, ...] @ x2[i, ...]) * x1Scale[i, :, None] * x2Scale[None, :]

函数原型:
  npu_quant_matmul_reduce_sum(x1, x2, *, x1_scale=None, x2_scale=None) -> Tensor

参数说明:
- x1: Tensor类型，必选参数，对应公式中的x1。数据类型支持`int8`，数据格式支持ND，shape支持3维，形状为（batch, m, k）。
- x2: Tensor类型，必选参数，对应公式中的x2。数据类型支持`int8`，数据格式支持NZ，shape支持3维，形状为（batch, k, n）。
- x1_scale: Tensor类型，必选关键字参数。对应公式中的x1Scale。数据类型支持`float32`，数据格式支持ND，shape支持2维，形状为（batch, m）。
- x2_scale: Tensor类型，必选关键字参数。数据类型支持`bfloat16`，数据格式支持ND，shape支持1维，形状为（n,）。

输出说明:
out: Tensor类型，算子的计算结果。输出的数据类型为`bfloat16`，数据格式为ND，shape为2维，形状为(m, n)。

约束说明:
- 该接口支持推理场景下使用。
- 该接口支持静态图模式。
- 传入的x1、x2、x1_scale、x2_scale不能是空。
- 输入和输出支持以下数据类型组合：
  | x1   | w2   | x1Scale | x2Scale  | out      |
  |------|------|---------|----------|----------|
  | int8 | int8 | float32 | bfloat16 | bfloat16 |

支持的PyTorch版本:
PyTorch2.1及以上

支持的型号:
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

调用示例:
- 单算子调用
    import torch
    import torch_npu

    b,m,k,n = (2,3,4,5)
    x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
    x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
    x1_scale = torch.ones((b, m), dtype=torch.float32).npu()
    x2_scale = torch.ones((n,), dtype=torch.bfloat16).npu()
    y = torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=x1_scale, x2_scale=x2_scale)
- 图模式调用
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger

    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    # "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
    os.environ["ENABLE_ACLNN"] = "false"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
 
        def forward(self, x1, x2, scale, pertoken_scale):
            return torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=pertoken_scale, x2_scale=scale)

    cpu_model = MyModel()
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)

    b,m,k,n = (2,3,4,5)
    x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
    x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
    pertoken_scale = torch.ones((b, m), dtype=torch.float32).npu()
    scale = torch.ones((n,), dtype=torch.bfloat16).npu()
    npu_out = model(x1, x2, scale, pertoken_scale)
    print(npu_out)
    """
)

_add_torch_npu_docstr(
    "npu_quant_matmul",
    """
功能描述:
完成量化的矩阵乘计算, 最小支持输入维度为2维, 最大支持输入维度为6维. 

接口原型:
torch_npu.npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? bias=None, ScalarType? output_dtype=None) -> Tensor

参数说明:
x1: Tensor类型, 数据格式支持ND, shape需要在2-6维范围. 
Atlas 推理系列加速卡产品: 数据类型支持int8. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8和int32. 其中int32表示int4类型矩阵乘计算, 每个int32数据存放8个int4数据. 
Atlas A3 训练系列产品: 数据类型支持int8和int32. 其中int32表示int4类型矩阵乘计算, 每个int32数据存放8个int4数据. 
x2: Tensor类型(weight), 数据格式支持ND, shape需要在2-6维范围. 
Atlas 推理系列加速卡产品: 数据类型与x1的数据类型须保持一致. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型与x1的数据类型保持一致. 
Atlas A3 训练系列产品: 数据类型与x1的数据类型保持一致. 
scale: Tensor类型, 数据格式支持ND, 如需传入int64数据类型的scale, 需要提前调用torch_npu.npu_trans_quant_param来获取int64数据类型的scale. 
Atlas 推理系列加速卡产品: 数据类型支持float32、int64. shape需要是1维(t, ), t=1或n, 其中n与x2的n一致. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、int64、bfloat16. shape需要是1维(t, ), t=1或n, 其中n与x2的n一致. 
Atlas A3 训练系列产品: 数据类型支持float32、int64、bfloat16. shape需要是1维(t, ), t=1或n, 其中n与x2的n一致. 
offset: Tensor类型, 可选参数. 数据类型支持float32, 数据格式支持ND, shape需要是1维(t,), t=1或n, 其中n与x2的n一致. 
当x1数据类型为int8时, 才支持该参数. 
pertoken_scale: Tensor类型, 可选参数. 数据类型支持float32, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 不支持pertoken_scale. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32. shape需要是1维(m,), 其中m与x1的m一致. 
Atlas A3 训练系列产品: 数据类型支持float32. shape需要是1维(m,), 其中m与x1的m一致. 
bias: Tensor类型, 可选参数, 数据格式支持ND, shape支持1维(n,)、2维(1, n)或3维(batch, 1, n), n与x2的n一致, 同时batch值需要等于x1和x2 boardcast后推导出的batch值. 当输出是2、4、5、6维情况下, bias的shape必须为1维. 当输出是2维情况下, bias的shape可以为1维或2维. 当输出是3维情况下, bias的shape可以为1维或3维. 
Atlas 推理系列加速卡产品: 数据类型支持int32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int32、bfloat16、float16、float32. 
Atlas A3 训练系列产品: 数据类型支持int32、bfloat16、float16、float32. 
output_dtype: ScalarType类型int类型, 可选参数. 表示输出Tensor的数据类型. 默认值为None, 代表输出Tensor数据类型为int8. 
Atlas 推理系列加速卡产品: 支持输入torch.int8、torch.float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持输入torch.int8、torch.float16、torch.bfloat16、torch.int32. 
Atlas A3 训练系列产品: 支持输入torch.int8、torch.float16、torch.bfloat16、torch.int32. 

输出说明:
result: Tensor类型, 代表量化matmul的计算结果. 
如果output_dtype为torch.float16, 输出的数据类型为float16. 
如果output_dtype为torch.int8或者None, 输出的数据类型为int8. 
如果output_dtype为torch.bfloat16, 输出的数据类型为bfloat16. 
如果output_dtype为torch.int32, 输出的数据类型为int32. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
传入的x1、x2、scale不能是空. 
x1、x2、bias、scale、offset、pertoken_scale、output_dtype的数据类型和数据格式需要在支持的范围之内. 
x1与x2最后一维的shape大小不能超过65535. 
目前输出int8或float16且无pertoken_scale情况下, 图模式不支持scale直接传入float32数据类型. 
如果在PyTorch图模式中使用本接口, 且环境变量ENABLE_ACLNN=false, 则在调用接口前需要对shape为(n, k//8)的x2数据进行转置, 转置过程应写在图中. 
支持将x2转为昇腾亲和的数据排布以提高搬运效率. 需要调用torch_npu.npu_format_cast完成输入x2(weight)为昇腾亲和的数据排布功能. 
Atlas 推理系列加速卡产品: 必须先将x2转置后再转亲和format. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 推荐x2不转置直接转亲和format. 
Atlas A3 训练系列产品: 推荐x2不转置直接转亲和format. 
int4类型计算的额外约束: 
当x1、x2的数据类型均为int32, 每个int32类型的数据存放8个int4数据. 输入的int32 shape需要将数据原本int4类型时shape的最后一维缩小8倍. int4数据的shape最后一维应为8的倍数, 例如: 进行(m, k)乘(k, n)的int4类型矩阵乘计算时, 需要输入int32类型、shape为(m, k//8)、(k, n//8)的数据, 其中k与n都应是8的倍数. x1只能接受shape为(m, k//8)且数据排布连续的数据, x2可以接受(k, n[g1] //8)且数据排布连续的数据或shape为(k//8, n)且是由数据连续排布的(n, k//8)转置而来的数据. 
数据排布连续是指数组中所有相邻的数, 包括换行时内存地址连续, 使用Tensor.is_contiguous返回值为true则表明tensor数据排布连续. 
输入参数间支持的数据类型组合情况如下: 
表1 Atlas 推理系列产品
x1:int8, int8
x2:int8, int8
scale:int64/float32, int64/float32
offset:None, float32/None
bias:int32/None, int32/None
pertoken_scale:None, None
output_dtype:float16, int8
表1 (Atlas A2 训练系列产品/Atlas 800I A2 推理产品)(Atlas A3 训练系列产品)
x1:int8, int8, int8, int8, int32, int8
x2:int8, int8, int8, int8, int32, int8
scale:int64/float32, int64/float32, float32/bfloat16, float32, int64/float32, float32/bfloat16
offset:None, float32/None, None, None, None, None
bias:int32/None, int32/None, int32/bfloat16/float32/None, int32/float16/float32/None, int32/None, int32/None
pertoken_scale:None, None, float32/None, float32, None, None
output_dtype:float16, int8, bfloat16, float16, float16, int32

支持的PyTorch版本
PyTorch 2.5
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列加速卡产品
Atlas A3 训练系列产品

调用示例:
单算子调用
int8类型输入场景: 
import torch
import torch_npu
import logging
import os

cpu_x1 = torch.randint(-5, 5, (1, 256, 768), dtype=torch.int8)
cpu_x2 = torch.randint(-5, 5, (31, 768, 16), dtype=torch.int8)
scale = torch.randn(16, dtype=torch.float32)
offset = torch.randn(16, dtype=torch.float32)
bias = torch.randint(-5, 5, (31, 1, 16), dtype=torch.int32)
# Method 1: You can directly call npu_quant_matmul
npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset=offset.npu(), bias=bias.npu())

# Method 2: You can first call npu_trans_quant_param to convert scale and offset from float32 to int64 when output dtype is not torch.bfloat16 and pertoken_scale is none
scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu())
npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_1,  bias=bias.npu())
图模式调用(ND数据格式)
输出float16
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
import numpy as np
# "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale, offset, bias):
        return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, output_dtype=torch.float16)
cpu_model = MyModel()
model = cpu_model.npu()
cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
scale = torch.randn(1, dtype=torch.float32)  
# pertoken_scale为空时, 输出fp16必须先调用npu_trans_quant_param, 将scale(offset)从float转为int64.
scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), None)
bias = torch.randint(-1,1, (15, 1, 128), dtype=torch.int32)
# dynamic=True: 动态图模式,  dynamic=False: 静态图模式
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale_1, None, bias.npu())
输出bfloat16, 示例代码如下, 仅支持如下产品: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
import numpy as np
os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
        return torch_npu.npu_quant_matmul(x1, x2.t(), scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
cpu_model = MyModel()
model = cpu_model.npu()
m = 15
k = 11264
n = 6912
bias_flag = True
cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
if bias_flag:
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, bias.npu(), pertoken_scale.npu())
else:
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, None, pertoken_scale.npu())
图模式调用(高性能数据排布方式)
将x2转置(batch, n, k)后转format, 示例代码如下, 仅支持Atlas 推理系列加速卡产品. 
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
import numpy as np
os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale, offset, bias):
        return torch_npu.npu_quant_matmul(x1, x2.transpose(2,1), scale, offset=offset, bias=bias)
cpu_model = MyModel()
model = cpu_model.npu()
cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8).npu()
cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8).npu()
# Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,n,k) layout
cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2.transpose(2,1).contiguous(), 29)
scale = torch.randn(1, dtype=torch.float32).npu()
offset = torch.randn(1, dtype=torch.float32).npu()
bias = torch.randint(-1,1, (128,), dtype=torch.int32).npu()
# Process scale from float32 to int64 offline to improve performance
scale_1 = torch_npu.npu_trans_quant_param(scale, offset)
model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
npu_out = model(cpu_x1, cpu_x2_t_29, scale_1, offset, bias)
将x2非转置(batch, k, n)后转format, 示例代码如下, 仅支持如下产品: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
import numpy as np
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
        return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
cpu_model = MyModel()
model = cpu_model.npu()
m = 15
k = 11264
n = 6912
bias_flag = True
cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
# Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,k,n) layout
x2_notranspose_29 = torch_npu.npu_format_cast(cpu_x2.npu().transpose(1,0).contiguous(), 29)
scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
if bias_flag:
    npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, bias.npu(), pertoken_scale.npu())
else:
    npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, None, pertoken_scale.npu())
"""
)

_add_torch_npu_docstr(
    "npu_weight_quant_batchmatmul",
    """
功能描述:
该接口用于实现矩阵乘计算中weight输入和输出的量化操作, 支持per-tensor、per-channel、per-group多场景量化. 
不同产品支持的量化算法不同, 如表 支持的量化场景所示. 
表1 支持的量化场景产品型号
量化方式
Atlas 推理系列加速卡产品: per-tensor、per-channel
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: per-tensor、per-channel、per-group
Atlas A3 训练系列产品: per-tensor、per-channel、per-group

接口原型:
torch_npu.npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale, Tensor? antiquant_offset=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? bias=None, int antiquant_group_size=0, int inner_precise=0) -> Tensor

参数说明:
x : Tensor类型, 即矩阵乘中的x. 数据格式支持ND, 支持带transpose的非连续的Tensor, 支持输入维度为两维(M, K) . 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16. 
weight: Tensor类型, 即矩阵乘中的weight. 支持带transpose的非连续的Tensor, 支持输入维度为两维(K, N), 维度需与x保持一致. 当数据格式为ND时, per-channel场景下为提高性能推荐使用transpose后的weight输入. 
Atlas 推理系列加速卡产品: 数据类型支持int8. 数据格式支持ND、FRACTAL_NZ, 其中FRACTAL_NZ格式只在“图模式”有效, 需依赖接口torch_npu.npu_format_cast完成ND到FRACTAL_NZ的转换, 可参考调用示例. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、int32(通过int32承载int4的输入, 可参考7.2.1.74-torch_npu.npu_convert_weight_to_int4pack调用示例). 数据格式支持ND、FRACTAL_NZ. 
Atlas A3 训练系列产品: 数据类型支持int8、int32(通过int32承载int4的输入, 可参考7.2.1.74-torch_npu.npu_convert_weight_to_int4pack调用示例). 数据格式支持ND、FRACTAL_NZ. 
antiquant_scale: Tensor类型, 反量化的scale, 用于weight矩阵反量化, 数据格式支持ND. 支持带transpose的非连续的Tensor. antiquant_scale支持的shape与量化方式相关: 
per_tensor模式: 输入shape为(1,)或(1, 1). 
per_channel模式: 输入shape为(1, N)或(N,). 
per_group模式: 输入shape为(ceil(K, antiquant_group_size),  N). 
antiquant_scale支持的dtype如下: Atlas 推理系列加速卡产品: 数据类型支持float16, 其数据类型需与x保持一致. Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int64. 若输入为float16、bfloat16,  其数据类型需与x保持一致. 若输入为int64, x数据类型必须为float16且不带transpose输入, 同时weight数据类型必须为int8、数据格式为ND、带transpose输入, 可参考调用示例. 此时只支持per-channel场景, M范围为[1, 96], 且K和N要求64对齐. Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int64. 若输入为float16、bfloat16,  其数据类型需与x保持一致. 若输入为int64, x数据类型必须为float16且不带transpose输入, 同时weight数据类型必须为int8、数据格式为ND、带transpose输入, 可参考调用示例. 此时只支持per-channel场景, M范围为[1, 96], 且K和N要求64对齐. 
antiquant_offset: Tensor类型, 反量化的offset, 用于weight矩阵反量化. 可选参数, 默认值为None, 数据格式支持ND, 支持带transpose的非连续的Tensor, 支持输入维度为两维(1, N)或一维(N, )、(1, ). 
Atlas 推理系列加速卡产品: 数据类型支持float16, 其数据类型需与antiquant_scale保持一致. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int32. per-group场景shape要求为(ceil_div(K, antiquant_group_size), N). 
若输入为float16、bfloat16, 其数据类型需与antiquant_scale保持一致. 
若输入为int32, antiquant_scale的数据类型必须为int64. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int32. per-group场景shape要求为(ceil_div(K, antiquant_group_size), N). 
若输入为float16、bfloat16, 其数据类型需与antiquant_scale保持一致. 
若输入为int32, antiquant_scale的数据类型必须为int64. 
quant_scale: Tensor类型, 量化的scale, 用于输出矩阵的量化, 可选参数, 默认值为None, 仅在weight格式为ND时支持. 数据类型支持float32、int64, 数据格式支持ND, 支持输入维度为两维(1, N)或一维(N, )、(1, ). 当antiquant_scale的数据类型为int64时, 此参数必须为空. 
Atlas 推理系列加速卡产品: 暂不支持此参数. 
quant_offset: Tensor类型, 量化的offset, 用于输出矩阵的量化, 可选参数, 默认值为None, 仅在weight格式为ND时支持. 数据类型支持float32, 数据格式支持ND, 支持输入维度为两维(1, N)或一维(N, )、(1, ). 当antiquant_scale的数据类型为int64时, 此参数必须为空. 
Atlas 推理系列加速卡产品: 暂不支持此参数. 
bias: Tensor类型,  即矩阵乘中的bias, 可选参数, 默认值为None, 数据格式支持ND,  不支持非连续的Tensor, 支持输入维度为两维(1, N)或一维(N, )、(1, ). 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float32. 当x数据类型为bfloat16, bias需为float32; 当x数据类型为float16, bias需为float16. 
Atlas A3 训练系列产品: 数据类型支持float16、float32. 当x数据类型为bfloat16, bias需为float32; 当x数据类型为float16, bias需为float16. 
antiquant_group_size: int类型,  用于控制per-group场景下group大小, 其他量化场景不生效. 可选参数. 默认值为0, per-group场景下要求传入值的范围为[32, K-1]且必须是32的倍数. 
Atlas 推理系列加速卡产品: 暂不支持此参数. 
inner_precise:  int类型, 计算模式选择,  默认为0. 0表示高精度模式, 1表示高性能模式, 可能会影响精度. 当weight以int32类型且以FRACTAL_NZ格式输入, M不大于16的per-group场景下可以设置为1, 提升性能. 其他场景不建议使用高性能模式. 

输出说明:
输出为Tensor类型, 代表计算结果. 当输入存在quant_scale时输出数据类型为int8, 当输入不存在quant_scale时输出数据类型和输入x一致. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 当输入weight为FRACTAL_NZ格式时暂不支持单算子调用, 只支持图模式调用. 
x和weight后两维必须为(M, K)和(K, N)格式, K、N的范围为[1, 65535]; 在x为非转置时, M的范围为[1, 2^31-1], 在x为转置时, M的范围为[1, 65535]. 
不支持空Tensor输入. 
antiquant_scale和antiquant_offset的输入shape要保持一致. 
quant_scale和quant_offset的输入shape要保持一致, 且quant_offset不能独立于quant_scale存在. 
如需传入int64数据类型的quant_scale, 需要提前调用torch_npu.npu_trans_quant_param接口将数据类型为float32的quant_scale和quant_offset转换为数据类型为int64的quant_scale输入, 可参考调用示例. 
当输入weight为FRACTAL_NZ格式且类型为int32时, per-channel场景需满足weight为转置输入; per-group场景需满足x为转置输入, weight为非转置输入, antiquant_group_size为64或128, K为antiquant_group_size对齐, N为64对齐. 
不支持输入weight shape为(1, 8)且类型为int4, 同时weight带有transpose的场景, 否则会报错x矩阵和weight矩阵K轴不匹配, 该场景建议走非量化算子获取更高精度和性能. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列加速卡产品

调用示例:
单算子模式调用
weight非transpose+quant_scale场景, 仅支持如下产品: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
import torch
import torch_npu
# 输入int8+ND 
cpu_x = torch.randn((8192, 320),dtype=torch.float16)
cpu_weight = torch.randint(low=-8, high=8, size=(320, 256),dtype=torch.int8)
cpu_antiquantscale = torch.randn((1, 256),dtype=torch.float16)
cpu_antiquantoffset = torch.randn((1, 256),dtype=torch.float16)
cpu_quantscale = torch.randn((1, 256),dtype=torch.float32)
cpu_quantoffset = torch.randn((1, 256),dtype=torch.float32)
quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())
npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())
weight transpose+antiquant_scale场景
import torch
import torch_npu
# 输入int8+ND 
cpu_x = torch.randn((96, 320),dtype=torch.float16)
cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
cpu_antiquantscale = torch.randn((256,1),dtype=torch.float16)
cpu_antiquantoffset = torch.randint(-128, 127, (256,1), dtype=torch.float16)
npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu().transpose(-1, -2), cpu_antiquantscale.npu().transpose(-1, -2), cpu_antiquantoffset.npu().transpose(-1, -2))
weight transpose+antiquant_scale场景 , 仅支持如下产品: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列加速卡产品
import torch
import torch_npu
cpu_x = torch.randn((96, 320),dtype=torch.float16)
cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
cpu_antiquantscale = torch.randn((256),dtype=torch.float16)
# 构建int64类型的scale参数
antiquant_scale = torch_npu.npu_trans_quant_param(cpu_antiquantscale.to(torch.float32).npu()).reshape(256, 1)
cpu_antiquantoffset = torch.randint(-128, 127, (256, 1), dtype=torch.int32)
npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.transpose(-1,-2).npu(), antiquant_scale.transpose(-1,-2).npu(), cpu_antiquantoffset.transpose(-1,-2).npu())
图模式调用
weight输入为ND格式
# 图模式
import torch
import torch_npu
import  torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)

cpu_x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
cpu_weight = torch.randn((320, 256),device='npu',dtype=torch.int8)
cpu_antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
cpu_antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
        return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)

cpu_model = MyModel()
model = cpu_model.npu()
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
npu_out = model(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
Atlas 推理系列加速卡产品: weight输入为FRACTAL_NZ格式
import torch_npu
import torch
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
class NPUQuantizedLinearA16W8(torch.nn.Module):
    def __init__(self,
                 weight,
                 antiquant_scale,
                 antiquant_offset,
                 quant_offset=None,
                 quant_scale=None,
                 bias=None,
                 transpose_x=False,
                 transpose_weight=True,
                 w4=False):
        super().__init__()

        self.dtype = torch.float16
        self.weight = weight.to(torch.int8).npu()
        self.transpose_weight = transpose_weight

        if self.transpose_weight:
            self.weight = torch_npu.npu_format_cast(self.weight.contiguous(), 29)
        else:
            self.weight = torch_npu.npu_format_cast(self.weight.transpose(0, 1).contiguous(), 29) # n,k ->nz

        self.bias = None
        self.antiquant_scale = antiquant_scale
        self.antiquant_offset = antiquant_offset
        self.quant_offset = quant_offset
        self.quant_scale = quant_scale
        self.transpose_x = transpose_x

    def forward(self, x):
        x = torch_npu.npu_weight_quant_batchmatmul(x.transpose(0, 1) if self.transpose_x else x,
                                                   self.weight.transpose(0, 1),
                                                   self.antiquant_scale.transpose(0, 1),
                                                   self.antiquant_offset.transpose(0, 1),
                                                   self.quant_scale,
                                                   self.quant_offset,
                                                   self.bias)
        return x


m, k, n = 4, 1024, 4096
cpu_x = torch.randn((m, k),dtype=torch.float16)
cpu_weight = torch.randint(1, 10, (k, n),dtype=torch.int8)
cpu_weight = cpu_weight.transpose(-1, -2)

cpu_antiquantscale = torch.randn((1, n),dtype=torch.float16)
cpu_antiquantoffset = torch.randn((1, n),dtype=torch.float16)
cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)
model = NPUQuantizedLinearA16W8(cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
model = torch.compile(model, backend=npu_backend, dynamic=True)
out = model(cpu_x.npu())
"""
)

_add_torch_npu_docstr(
    "npu_transpose_batchmatmul",
    """
功能描述: 
该接口用于实现矩阵乘计算输入和输出的transpose操作。

接口原型: 
torch_npu.npu_transpose_batchmatmul(Tensor input, Tensor weight, *, Tensor? bias=None, Tensor? scale=None, int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None, int? batch_split_factor=1) -> Tensor

参数说明: 
- input(Tensor, 计算输入): 必选参数, 一个3D的Device侧Tensor输入，矩阵计算的左矩阵。数据类型支持float32、float16、bfloat16，数据格式支持ND。
- weight(Tensor, 计算输入): 必选参数, 一个3D的Device侧Tensor输入，矩阵计算的右矩阵。数据类型支持float32、float16、bfloat16，数据格式支持ND。
- bias(Tensor, 计算输入): 可选参数, 一个1D的Device侧Tensor输入，矩阵计算的bias参数。数据类型支持float32、float16、bfloat16，数据格式支持ND。
- scale(Tensor, 计算输入): 可选参数, 一个1D的Device侧Tensor输入，矩阵计算量化参数。数据类型支持int64、uint64，数据格式支持ND。
- perm_x1(ListInt, 计算输入): 可选参数, int类型列表，将input在矩阵乘之前排列成[B, M, K]。
- perm_x2(ListInt, 计算输入): 可选参数, int类型列表，将weight在矩阵乘之前排列成[B, K, N]。
- perm_y(ListInt, 计算输入): 可选参数, int类型列表，将y在矩阵乘后重新排列。
- batch_split_factor(Int, 计算输入): 可选参数，声明output_batch的系数，默认是1。
- y(Tensor, 计算输出): 一个3D的Tensor，输出数据类型支持float32、float16、int8、bfloat16。

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 推理系列产品

调用示例:
# 单算子调用
import torch
import torch_npu

M, K, N, Batch = 32, 512, 128, 32
x1 = torch.randn((M, Batch, K), dtype=torch.float16)
x2 = torch.randn((Batch, K, N), dtype=torch.float16)
scale = torch.rand((Batch * N, ), dtype=torch.float32)
scale = torch_npu.npu_trans_quant_param(scale.npu(), round_mode=1)
y = torch_npu.npu_transpose_batchmatmul(x1.npu(), x2.npu(), scale=scale.npu(),
                                        perm_x1=[1, 0, 2], perm_y=[1, 0, 2])

# 图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, scale):
        scale = torch_npu.npu_trans_quant_param(scale, round_mode=1)
        output = torch_npu.npu_transpose_batchmatmul(x1, x2, scale=scale,
                                                     perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                                                     perm_y=(1, 0, 2))
        return output

M, K, N, Batch = 32, 512, 128, 32
x1 = torch.randn((M, Batch, K), dtype=torch.float16)
x2 = torch.randn((Batch, K, N), dtype=torch.float16)
scale = torch.rand((Batch * N, ), dtype=torch.float32)

model = Model().npu()
model = torch.compile(model, backend=npu_backend, dynamic=False)
y = model(x1.npu(), x2.npu(), scale.npu())
"""
)

_add_torch_npu_docstr(
    "npu_convert_weight_to_int4pack",
    """
功能描述:
将int32类型的输入tensor打包为int4存放, 每8个int4数据通过一个int32数据承载, 并进行交叠排放. 

接口原型:
torch_npu.npu_convert_weight_to_int4pack(Tensor weight, int inner_k_tiles=0) -> Tensor

参数说明:
weight : Tensor类型, 输入的weight, 数据格式支持ND、FRACTAL_NZ, 数据类型支持int32,  不支持非连续的Tensor; 维度支持2维, shape支持(k, n)、 (n, k), 最后一维度需要8个元素对齐, 元素的值需要在int4的表示范围内, 即[-8, 7]. 
inner_k_tiles: int类型, 用于制定内部打包格式中, 多少个K-tiles被打包在一起, 默认值为0. 预留参数, 暂未使用. 

输出说明:
输出为Tensor类型, 代表int4打包后的输出, 数据类型为int32, shape为(k, n/8), (n, k/8), 数据格式支持ND. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3.1
PyTorch 2.0
PyTorch 2.1
PyTorch 2.2
PyTorch 1.11

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
import torch
import torch_npu

m = 128
k = 64
n = 32
trans_weight = False

cpu_x = torch.randn((m, k), dtype=torch.float16)
if trans_weight:
    cpu_weight = torch.randint(low=-8, high=8, size=(n, k), dtype=torch.int32)
    cpu_antiquantscale = torch.randn((n, 1), dtype=torch.float16)
    cpu_antiquantoffset = torch.randn((n, 1), dtype=torch.float16)
else:
    cpu_weight = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int32)
    cpu_antiquantscale = torch.randn((1, n), dtype=torch.float16)
    cpu_antiquantoffset = torch.randn((1, n), dtype=torch.float16)

weight_int4 = torch_npu.npu_convert_weight_to_int4pack(cpu_weight.npu())

if trans_weight:
    cpu_weight = cpu_weight.transpose(-1, -2)
    weight_int4 = weight_int4.transpose(-1, -2)
    cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
    cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)

npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), weight_int4.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
图模式调用
import torch
import torch_npu
import  torchair
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

m = 16
k = 17
n = 72

trans_weight = False
is_weight_nz = False

cpu_x = torch.randn((m, k),dtype=torch.float16)
if trans_weight:
    cpu_weight = torch.randint(low=-8, high=8, size=(n, k) ,dtype=torch.int32)
    cpu_antiquantscale = torch.ones((n, 1),dtype=torch.float16)
    cpu_antiquantoffset = torch.zeros((n, 1),dtype=torch.float16)
else:
    cpu_weight = torch.randint(low=-8, high=8, size=(k, n) ,dtype=torch.int32)
    cpu_antiquantscale = torch.ones((1, n),dtype=torch.float16)
    cpu_antiquantoffset = torch.zeros((1, n),dtype=torch.float16)

npu_weight = cpu_weight.npu()
if is_weight_nz:
   # nd to fractal_nz
   npu_weight = torch_npu.npu_format_cast(npu_weight.npu(), 29)
# int32 to int4pack
weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(npu_weight)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
        if trans_weight:
            weight  = weight.transpose(-1, -2)
            antiquant_scale = antiquant_scale.transpose(-1, -2)
            antiquant_offset = antiquant_offset.transpose(-1, -2)
        return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)

cpu_model = MyModel()
model = cpu_model.npu()
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True, fullgraph=True)

npu_out = model(cpu_x.npu(), weight_int4pack, cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
"""
)

_add_torch_npu_docstr(
    "npu_grouped_matmul",
    """
功能描述:
算子功能: npu_grouped_matmul是一种对多个矩阵乘法(matmul)操作进行分组计算的高效方法. 该API实现了对多个矩阵乘法操作的批量处理, 通过将具有相同形状或相似形状的矩阵乘法操作组合在一起, 减少内存访问开销和计算资源的浪费, 从而提高计算效率. 
计算公式: 
非量化场景(公式1): 
y_{i}=x_{i}×weight_{i}+bias_{i}
per-channel量化场景 (公式2): 
y_{i}=(x_{i}×weight_{i}+bias_{i})×scale_{i}+offset_{i}
per-token量化场景 (公式3): 
y_{i}=(x_{i}×weight_{i}+bias_{i})×scale_{i}+pertokenscale_{i}
伪量化场景 (公式4): 
y_{i}=x_{i}×(weight_{i}+antiquant_offset_{i})×antiquantscale_{i}+bias_{i}

接口原型:
npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0, output_dtype=None, int[]? tuning_config) -> List[torch.Tensor]

参数说明:
x (List[torch.Tensor]): 输入矩阵列表, 表示矩阵乘法中的左矩阵. 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: torch.float16、torch.float32、torch.bfloat16和torch.int8. 
Atlas 推理系列产品: torch.float16. . 
列表最大长度为128. 
当split_item=0时, 张量支持2至6维输入; 其他情况下, 张量仅支持2维输入. 
weight (List[torch.Tensor]): 权重矩阵列表, 表示矩阵乘法中的右矩阵. 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 
当group_list输入类型为List[int]时, 支持torch.float16、torch.float32、torch.bfloat16和torch.int8. 
当group_list输入类型为torch.Tensor时, 支持torch.float16、torch.float32、torch.bfloat16、int4和torch.int8. 
Atlas 推理系列产品: torch.float16. 
列表最大长度为128. 
每个张量支持2维或3维输入. 
bias (List[torch.Tensor]): 每个分组的矩阵乘法输出的独立偏置项. 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: torch.float16、torch.float32和torch.int32. 
Atlas 推理系列产品: torch.float16. 
列表长度与weight列表长度相同. 
每个张量仅支持1维输入. 
scale (List[torch.Tensor]): 用于缩放原数值以匹配量化后的范围值, 代表量化参数中的缩放因子, 对应公式(2)、公式(3)和公式(5). 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 
当group_list输入类型为List[int]时, 支持torch.int64. 
当group_list输入类型为torch.Tensor时, 支持torch.float32、torch.bfloat16和torch.int64. 
Atlas 推理系列产品: 仅支持传入None. . 
列表长度与weight列表长度相同. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品每个张量仅支持1维输入. 
offset (List[torch.Tensor]): 用于调整量化后的数值偏移量, 从而更准确地表示原始浮点数值, 对应公式(2). 当前仅支持传入None. 
antiquant_scale (List[torch.Tensor]): 用于缩放原数值以匹配伪量化后的范围值, 代表伪量化参数中的缩放因子, 对应公式(4). 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: torch.float16、torch.bfloat16. 
Atlas 推理系列产品: 仅支持传入None. 
列表长度与weight列表长度相同. 
每个张量支持输入维度如下(其中g为matmul组数, G为per-group数, Gi为第i个tensor的per-group数): 
伪量化per-channel场景, weight为单tensor时, shape限制为[g, n]; weight为多tensor时, shape限制为[ni]. 
伪量化per-group场景, weight为单tensor时, shape限制为[g, G, n]; weight为多tensor时, shape限制为[Gi, ni]. 
antiquant_offset (List[torch.Tensor]): 用于调整伪量化后的数值偏移量, 从而更准确地表示原始浮点数值, 对应公式(4). 
支持的数据类型如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: torch.float16、torch.bfloat16. 
Atlas 推理系列产品: 仅支持传入None. 
列表长度与weight列表长度相同. 
每个张量输入维度和antiquant_scale输入维度一致. 
per_token_scale (List[torch.Tensor]): 用于缩放原数值以匹配量化后的范围值, 代表per-token量化参数中由x量化引入的缩放因子, 对应公式(3)和公式(5). 
group_list输入类型为List[int]时, 当前只支持传入None. 
group_list输入类型为torch.Tensor时: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持torch.float32. 
列表长度与x列表长度相同. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 每个张量仅支持1维输入. 
group_list (List[int]/torch.Tensor): 用于指定分组的索引, 表示x的第0维矩阵乘法的索引情况. 数据类型支持torch.int64. 
Atlas 推理系列产品: 仅支持torch.Tensor类型. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持List[int]或torch.Tensor类型. 
Atlas 推理系列产品: 每个张量仅支持1维输入, 长度与weight列表长度相同. 
和Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 每个张量仅支持1维输入, 长度与weight列表长度相同. 
配置值要求如下: 
group_list输入类型为List[int]时, 配置值必须为非负递增数列, 且长度不能为1. 
group_list输入类型为torch.Tensor时: 
当group_list_type为0时, group_list必须为非负单调非递减数列. 
当group_list_type为1时, group_list必须为非负数列, 且长度不能为1. 
activation_input (List[torch.Tensor]): 代表激活函数的反向输入, 当前仅支持传入None. 
activation_quant_scale (List[torch.Tensor]): 预留参数, 当前只支持传入None. 
activation_quant_offset (List[torch.Tensor]): 预留参数, 当前只支持传入None. 
split_item (int): 用于指定切分模式. 数据类型支持torch.int32. 
0/1: 输出为多个张量, 数量与weight相同. 
2/3: 输出为单个张量. 
group_type (int): 代表需要分组的轴. 数据类型支持torch.int32. 
group_list输入类型为List[int]时仅支持传入None. 
group_list输入类型为torch.Tensor时, 若矩阵乘为C[m,n]=A[m,k]xB[k,n], group_type支持的枚举值为: -1代表不分组; 0代表m轴分组; 1代表n轴分组.
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当前支持取-1、0. 
Atlas 推理系列产品: 当前只支持取0. 
group_list_type (int): 代表group_list的表达形式. 数据类型支持torch.int32. 
group_list输入类型为List[int]时仅支持传入None. 
group_list输入类型为torch.Tensor时: 
可取值0或1, 0代表group_list_type中数值为分组轴大小的cumsum结果(累积和), 1代表group_list_type中数值为分组轴上每组大小. 
act_type (int): 代表激活函数类型. 数据类型支持torch.int32. 
group_list输入类型为List[int]时仅支持传入None. 
group_list输入类型为torch.Tensor时, 支持的枚举值包括: 0代表不激活; 1代表RELU激活; 2代表GELU_TANH激活; 3代表暂不支持; 4代表FAST_GELU激活; 5代表SILU激活. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 取值范围为0-5. 
Atlas 推理系列产品: 当前只支持传入0. 
output_dtype (torch.dtype): 输出数据类型. 支持的配置包括: 
None: 默认值, 表示输出数据类型与输入x的数据类型相同. 
与输出y数据类型一致的类型, 具体参考约束说明. 

输出说明:
List[torch.Tensor]: 当split_item为0或1时, 返回的张量数量与weight相同. 当split_item为2或3时, 返回的张量数量为1. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品的内轴限制InnerLimit为65536. x和weight中每一组tensor的最后一维大小都应小于InnerLimit. xi的最后一维指当x不转置时xi的K轴或当x转置时xi的M轴. weighti的最后一维指当weight不转置时weighti的N轴或当weight转置时weighti的K轴. 
各场景输入与输出数据类型使用约束: 
group_list输入类型为List[int]时, Atlas A2 训练系列产品/Atlas 800I A2 推理产品数据类型使用约束:
表1 数据类型约束场景
非量化
x: torch.float16, torch.bfloat16, torch.float32
weight: torch.float16, torch.bfloat16, torch.float32
bias: torch.float16, torch.float32, torch.float32
scale: 无需赋值, 无需赋值, 无需赋值
antiquant_scale: 无需赋值, 无需赋值, 无需赋值
antiquant_offset:  无需赋值, 无需赋值, 无需赋值
output_dtype: torch.float16, torch.bfloat16, torch.float32
y: torch.float16, torch.bfloat16, torch.float32
per-channel量化
x: torch.int8
weight: torch.int8
bias: torch.int32
scale: torch.int64
antiquant_scale: 无需赋值 
antiquant_offset:  无需赋值 
output_dtype: torch.int8
y: torch.int8
伪量化
x: torch.float16, torch.bfloat16
weight: torch.int8, torch.int8
bias: torch.float16, torch.float32
scale: 无需赋值, 无需赋值
antiquant_scale: torch.float16, torch.bfloat16
antiquant_offset: torch.float16, torch.bfloat16
output_dtype: torch.float16, torch.bfloat16
y: torch.float16, torch.bfloat16
group_list输入类型为torch.Tensor时, 数据类型使用约束:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 
表1 数据类型约束场景
非量化
x: torch.float16, torch.bfloat16, torch.float32
weight: torch.float16, torch.bfloat16, torch.float32
bias: torch.float16, torch.float32, torch.float32
scale: 无需赋值, 无需赋值, 无需赋值
antiquant_scale: 无需赋值, 无需赋值, 无需赋值
antiquant_offset: 无需赋值, 无需赋值, 无需赋值
per_token_scale: 无需赋值, 无需赋值, 无需赋值
output_dtype: None/torch.float16, None/torch.bfloat16, None/torch.float32(仅x/weight/y均为单张量)
y: torch.float16, torch.bfloat16,torch.float32 
per-channel量化
x: torch.int8, torch.int8, torch.int8
weight: torch.int8, torch.int8, torch.int8
bias: torch.int32, torch.int32, torch.int32
scale: torch.int64, torch.bfloat16, torch.float32
antiquant_scale: 无需赋值, 无需赋值, 无需赋值
antiquant_offset: 无需赋值, 无需赋值, 无需赋值
per_token_scale: 无需赋值, 无需赋值, 无需赋值
output_dtype: None/torch.int8, torch.bfloat16, torch.float16
y: torch.int8, torch.bfloat16, torch.float16
per-token量化
x: torch.int8, torch.int8
weight: torch.int8, torch.int8
bias: torch.int32, torch.int32
scale: torch.bfloat16, torch.float32
antiquant_scale: 无需赋值, 无需赋值
antiquant_offset: 无需赋值, 无需赋值
per_token_scale: torch.float32, torch.float32
output_dtype: torch.bfloat16, torch.float16
y: torch.bfloat16, torch.float16
伪量化
x: torch.float16, torch.bfloat16
weight: torch.int8/int4, torch.int8/int4
bias: torch.float16, torch.float32
scale: 无需赋值, 无需赋值
antiquant_scale: torch.float16, torch.bfloat16
antiquant_offset: torch.float16, torch.bfloat16
per_token_scale: 无需赋值, 无需赋值
output_dtype: None/torch.float16, None/torch.bfloat16
y: torch.float16, torch.bfloat16
伪量化场景, 若weight的类型为torch.int8, 仅支持per-channel模式; 若weight的类型为int4, 支持per-channel和per-group两种模式. 若为per-group, per-group数G或Gi必须要能整除对应的ki. 若weight为多tensor, 定义per-group长度si = ki / Gi, 要求所有si(i=1,2,...g)都相等. 
伪量化场景, 若weight的类型为int4, 则weight中每一组tensor的最后一维大小都应是偶数. weighti的最后一维指weight不转置时weighti的N轴或当weight转置时weighti的K轴. 并且在per-group场景下, 当weight转置时, 要求per-group长度si是偶数. tensor转置: 指若tensor shape为[M,K]时, 则stride为[1,M],数据排布为[K,M]的场景, 即非连续tensor. 
当前PyTorch不支持int4类型数据, 需要使用时可以通过torch_npu.npu_quantize接口使用torch.int32数据表示int4. 
Atlas 推理系列产品: 
表1 数据类型约束
x: torch.float16
weight: torch.float16
bias: torch.float16
scale: 无需赋值
antiquant_scale: 无需赋值
antiquant_offset: 无需赋值
per_token_scale: torch.float32
output_dtype: torch.float16
y: torch.float16
根据输入x、输入weight与输出y的Tensor数量不同, 支持以下几种场景. 场景中的“单”表示单个张量, “多”表示多个张量. 场景顺序为x、weight、y, 例如“单多单”表示x为单张量, weight为多张量, y为单张量. 
group_list输入类型为List[int]时, Atlas A2 训练系列产品/Atlas 800I A2 推理产品各场景的限制. 
场景说明
多多多: x和weight为多张量, y为多张量. 每组数据的张量是独立的. 
单多单: x为单张量, weight为多张量, y为单张量. 
单多多: x为单张量, weight为多张量, y为多张量. 
多多单: x和weight为多张量, y为单张量. 每组矩阵乘法的结果连续存放在同一个张量中. 
场景限制
多多多: 仅支持split_item为0或1. x中tensor要求维度一致, 支持2-6维, weight中tensor需为2维, y中tensor维度和x保持一致. x中tensor大于2维, group_list必须传空. x中tensor为2维且传入group_list, group_list的差值需与x中tensor的第一维一一对应. 
单多单: 仅支持split_item为2或3. 必须传group_list, 且最后一个值与x中tensor的第一维相等. x、weight、y中tensor需为2维. weight中每个tensor的N轴必须相等. 
单多多: 仅支持split_item为0或1. 必须传group_list, group_list的差值需与y中tensor的第一维一一对应. x、weight、y中tensor需为2维. 
多多单: 仅支持split_item为2或3. x、weight、y中tensor需为2维. weight中每个tensor的N轴必须相等. 若传入group_list, group_list的差值需与x中tensor的第一维一一对应. 
group_list输入类型为torch.Tensor时, 各场景的限制. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 
量化、伪量化仅支持group_type为-1和0场景. 
仅per-token量化场景支持激活函数计算. 
group_type
-1: 多多多, x和weight为多张量, y为多张量. 每组数据的张量是独立的. 
0: 单单单, x、weight与y均为单张量. 
0: 单多单, x为单张量, weight为多张量, y为单张量. 
0: 多多单, x和weight为多张量, y为单张量. 每组矩阵乘法的结果连续存放在同一个张量中. 
场景限制
-1: 仅支持split_item为0或1. x中tensor要求维度一致, 支持2-6维, weight中tensor需为2维, y中tensor维度和x保持一致. group_list必须传空. 支持weight转置, 但weight中每个tensor是否转置需保持统一. x不支持转置. 
0: 仅支持split_item为2或3. weight中tensor需为3维, x、y中tensor需为2维. 必须传group_list, 且当group_list_type为0时, 最后一个值与x中tensor的第一维相等, 当group_list_type为1时, 数值的总和与x中tensor的第一维相等. group_list第1维最大支持1024, 即最多支持1024个group. 支持weight转置. x不支持转置. 
0: 仅支持split_item为2或3. 必须传group_list, 且当group_list_type为0时, 最后一个值与x中tensor的第一维相等, 当group_list_type为1时, 数值的总和与x中tensor的第一维相等, 长度最大为128. x、weight、y中tensor需为2维. weight中每个tensor的N轴必须相等. 支持weight转置, 但weight中每个tensor是否转置需保持统一. x不支持转置. 
0:  仅支持split_item为2或3. x、weight、y中tensor需为2维. weight中每个tensor的N轴必须相等. 若传入group_list, 当group_list_type为0时, group_list的差值需与x中tensor的第一维一一对应, 当group_list_type为1时, group_list的数值需与x中tensor的第一维一一对应, 且长度最大为128. 支持weight转置, 但weight中每个tensor是否转置需保持统一. x不支持转置. 
Atlas 推理系列产品: 
输入输出只支持float16的数据类型, 输出y的n轴大小需要是16的倍数. 
group_type
0: 单单单, x、weight与y均为单张量
场景限制
0: 仅支持split_item为2或3. weight中tensor需为3维, x、y中tensor需为2维. 必须传group_list, 且当group_list_type为0时, 最后一个值与x中tensor的第一维相等, 当group_list_type为1时, 数值的总和与x中tensor的第一维相等. group_list第1维最大支持1024, 即最多支持1024个group. 支持weight转置, 不支持x转置. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列产品

调用示例:

单算子模式调用
通用调用示例
import torch
import torch_npu
x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
x = [x1, x2, x3]
weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
weight = [weight1, weight2, weight3]
bias1 = torch.randn(256, device='npu', dtype=torch.float16)
bias2 = torch.randn(1024, device='npu', dtype=torch.float16)
bias3 = torch.randn(128, device='npu', dtype=torch.float16)
bias = [bias1, bias2, bias3]
group_list = None
split_item = 0
npu_out = torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=-1)
图模式调用
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class GMMModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight):
        return torch_npu.npu_grouped_matmul(x, weight, group_type=-1)

def main():
    x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
    x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
    x = [x1, x2, x3]
    
    weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
    weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
    weight = [weight1, weight2, weight3]
    
    model = GMMModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    custom_output = model(x, weight)

if __name__ == '__main__':
    main()
"""
)

_add_torch_npu_docstr(
    "npu_rms_norm_quant",
    """
接口原型
npu_rms_norm_quant(Tensor x, Tensor gamma, Tensor beta, Tensor scale, Tensor offset, float epsilon=1e-06) -> Tensor

功能描述
RmsNormQuant算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来,减少搬入搬出的操作。

参数说明
x: Device侧的Tensor类型，标准化输入张量。shape支持1-8维，数据类型支持FLOAT16、BFLOAT16，格式支持ND。不支持空Tensor。
gamma: Device侧的Tensor类型，归一化权重张量。shape为1-2维，需与x最后一维一致，数据类型与x一致。格式支持ND。不支持空Tensor。
beta: Device侧的Tensor类型，归一化偏置项。shape和数据类型与x一致。格式支持ND。不支持空Tensor。
scale: Device侧的Tensor类型，量化过程中得到y进行的scale张量,shape为1,维度为1.格式支持ND。不支持空Tensor。
offset: Device侧的Tensor类型，量化过程中得到y进行的offset张量.shape与scale保持一致,格式支持ND。不支持空Tensor。
epsilon: double类型，防止除0错误，默认值为1e-6.

输出说明
y: Device侧的Tensor类型。数据类型支持INT8。shape、数据格式需要与入参x保持一致。支持非连续的Tensor，不支持空Tensor。

约束说明
x、y的尾轴长度,以及gamma的尾轴长度必大于等于32Bytes.

支持的型号
Atlas A3训练系列产品/Atlas A3推理系列产品
Atlas A2训练系列产品/Atlas 800I A2推理产品/A200I A2 Box异构组件

调用示例
import torch
import torch_npu

eps = 1e-6
x = torch.randn(16, dtype = torch.float16).npu()
gamma = torch.randn(16, dtype = torch.float16).npu()
beta = torch.zeros(16, dtype = torch.float16).npu()
scale = torch.ones(1, dtype = torch.float16).npu()
offset = torch.zeros(1, dtype = torch.int8).npu()

y = torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, eps)
_ = y.cpu().numpy()
"""
)

_add_torch_npu_docstr(
    "npu_grouped_matmul_finalize_routing",
    """
功能描述：
GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。

接口原型：
torch_npu.npu_grouped_matmul_finalize_routing(Tensor x, Tensor weight, Tensor group_list, *, Tensor? scale=None, Tensor? bias=None, Tensor? pertoken_scale=None, Tensor? shared_input=None, Tensor? logit=None, Tensor? row_index=None, ScalarType? dtype=None, float? shared_input_weight=1.0, int? shared_input_offset=0, int? output_bs=0, int? group_list_type=1, int[]? tuning_config) -> Tensor

参数说明：
- x(Tensor, 计算输入): 必选参数，一个2D的Device侧Tensor输入，矩阵计算的左矩阵，不支持非连续的Tensor。数据类型支持int8，数据格式支持ND，维度为(m,k)。m取值范围为[1, 16*1024*8]，K取值为16整倍数。
- weight(Tensor, 计算输入): 必选参数，一个3D的Device侧Tensor输入，矩阵计算的右矩阵，不支持非连续的Tensor。数据类型支持int8、int4。a8w8场景下数据格式支持NZ，维度为(e,k,n)，e取值范围为[1, 256]，n取值为32整数倍且大于等于256，a8w4场景下数据格式支持ND，维度为(e,k,n)，k只支持2048，n只支持7168。
- group_list(Tensor, 计算输入): 必选参数，一个1D的Device侧Tensor输入，GroupedMatMul的各分组大小值，不支持非连续的Tensor。数据类型支持int64，数据格式支持ND，维度为(e,)，group_list的值的总和要求小于等于m。
- scale(Tensor, 计算输入): 可选参数，Device侧Tensor输入，矩阵计算反量化参数，对应weight矩阵，不支持非连续的Tensor。a8w8场景下是2D的Tensor，数据类型支持float32，数据格式支持ND，支持per-channel量化方式，维度为(e,n)；a8w4场景下是3D的Tensor，数据类型支持int64，维度为(e,1,n)。
- bias(Tensor, 计算输入): 可选参数，一个2D的Device侧Tensor输入，矩阵计算的bias参数，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND,只支持a8w4场景。
- offset(Tensor, 计算输入): 可选参数，Device侧Tensor输入，矩阵计算量化参数的偏移量，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND，只支持a8w4场景。
- pertoken_scale(Tensor, 计算输入): 可选参数，一个1D的Device侧Tensor输入，矩阵计算的反量化参数，对应x矩阵，per-token量化方式，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND，维度为(m,)。
- shared_input(Tensor, 计算输入): 可选参数，一个2D的Device侧Tensor输入，moe计算中共享专家的输出，需要与moe专家的输出进行combine操作，不支持非连续的Tensor。数据类型支持bfloat16，数据格式支持ND，维度为(batch/dp,n)，batch/dp取值范围[1, 2*1024]，batch取值范围[1, 16*1024]。
- logit(Tensor, 计算输入): 可选参数，一个1D的Device侧Tensor输入，moe专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND，维度为(m,)。
- row_index(Tensor*, 计算输入): 可选参数，一个1D的Device侧Tensor输入，moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引，不支持非连续的Tensor。数据类型支持int32、int64，数据格式支持ND，维度为(m,)。
- dtype(torch.dtype, 计算输入): 可选参数，指定GroupedMatMul计算的输出类型。枚举值含义：0表示float32，1表示float16，2表示bfloat16。默认值为0。
- shared_input_weight(float, 计算输入): 可选参数，float类型，指共享专家与moe专家进行combine的系数，shared_input先与该参数相乘，然后再和moe专家结果累加。默认为1.0。
- shared_input_offset(int, 计算输入): 可选参数，共享专家输出在总输出中的偏移。默认值为0.
- output_bs(int, 计算输入): 可选参数，输出的最高维大小。默认值为0。
- group_list_type(int, 计算输入): 可选参数，GroupedMatMul的分组模式，0为cumsum模式，1为count模式，默认为1。
- tuning_config:(ListInt, 计算输入): 可选参数，数组中第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值进行最优tiling。
- y(Tensor, 计算输出): 2D的Tensor，不支持非连续的Tensor，输出的数据类型固定为float32，维度为(batch, n)。

支持的芯片型号：
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 推理系列产品

调用示例：
# 单算子调用
import numpy as np
import torch
import torch_npu
from scipy.special import softmax

m, k, n = 576, 2048, 7168
batch = 72
topK = 8
group_num = 8

x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
group_list = np.array([batch] * group_num, dtype=np.int64)
shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
logit = logit.reshape(m)
row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

x_clone = torch.from_numpy(x).npu()
weight_clone = torch.from_numpy(weight).npu()
weightNz = torch_npu.npu_format_cast(weight_clone, 29)
scale_clone = torch.from_numpy(scale).npu()
pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
group_list_clone = torch.from_numpy(group_list).npu()
shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
logit_clone = torch.from_numpy(logit).npu()
row_index_clone = torch.from_numpy(row_index).npu()
shared_input_offset = batch // 2
output_bs = batch
y = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNz,
            group_list_clone, scale=scale_clone, pertoken_scale=pertoken_scale_clone,
            shared_input=shared_input_clone, logit=logit_clone, row_index=row_index_clone,
            shared_input_offset=shared_input_offset, output_bs=output_bs)

# 图模式调用
import numpy as np
import torch
import torch_npu
import torchair as tng
from scipy.special import softmax
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, weight, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs):
        output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,
                    scale=scale, pertoken_scale=pertoken_scale, shared_input=shared_input,
                    logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs)
        return output

m, k, n = 576, 2048, 7168
batch = 72
topK = 8
group_num = 8

x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
group_list = np.array([batch] * group_num, dtype=np.int64)
shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
logit = logit.reshape(m)
row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

x_clone = torch.from_numpy(x).npu()
weight_clone = torch.from_numpy(weight).npu()
weightNz = torch_npu.npu_format_cast(weight_clone, 29)
scale_clone = torch.from_numpy(scale).npu()
pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
group_list_clone = torch.from_numpy(group_list).npu()
shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
logit_clone = torch.from_numpy(logit).npu()
row_index_clone = torch.from_numpy(row_index).npu()
shared_input_offset = batch // 2
output_bs = batch

model = Model().npu()
model = torch.compile(model, backend=npu_backend, dynamic=False)
y = model(x_clone, weightNz, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone,
        logit_clone, row_index_clone, shared_input_offset, output_bs)
"""
)

_add_torch_npu_docstr(
    "npu_quant_scatter",
    """
功能描述:
先将updates进行量化, 然后将updates中的值按指定的轴axis和索引indices更新input中的值, 并将结果保存到输出tensor, input本身的数据不变. 

接口原型:
torch_npu.npu_quant_scatter(Tensor input, Tensor indices, Tensor updates, Tensor quant_scales, Tensor? quant_zero_points=None, int axis=0, int quant_axis=1, str reduce='update') -> Tensor

参数说明:
input: Tensor类型, 必选输入, 源数据张量, 数据类型支持int8, 数据格式支持ND, 支持非连续的Tensor, 维数只能是3~8维. 
indices: Tensor类型, 必选输入, 索引张量, 数据类型支持int32, 数据格式支持ND, 支持非连续的Tensor. 
updates: Tensor类型, 必选输入, 更新数据张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、float16. 
quant_scales: Tensor类型, 必选输入, 量化缩放张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、float32. 
quant_zero_points: Tensor类型, 可选输入, 量化偏移张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持int32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、int32. 
axis: int类型, 可选参数, updates上用来更新的轴, 默认值为0. 
quant_axis: int类型, 可选参数, updates上用来量化的轴, 默认值为1. 
reduce: 字符串类型, 可选参数, 表示数据操作方式; 当前只支持'update', 即更新操作. 

输出说明:
一个Tensor类型的输出, 代表input被更新后的结果. 

约束说明:
该接口支持图模式. 
indices的维数只能是1维或者2维; 如果是2维, 其第2维的大小必须是2; 不支持索引越界, 索引越界不校验; indices映射的input数据段不能重合, 若重合则会因为多核并发原因导致多次执行结果不一样. 
updates的维数需要与input的维数一样; 其第1维的大小等于indices的第1维的大小, 且不大于input的第1维的大小; 其axis轴的大小不大于input的axis轴的大小; 其余维度的大小要跟input对应维度的大小相等; 其最后一维的大小必须32B对齐. 
quant_scales的元素个数需要等于updates在quant_axis轴的大小. 
quant_zero_points的元素个数需要等于updates在quant_axis轴的大小. 
axis不能为updates的第1维或最后1维. 
quant_axis只能为updates的最后1维. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import numpy as np

data_var = np.random.uniform(0, 1, [24, 4096, 128]).astype(np.int8)
var = torch.from_numpy(data_var).to(torch.int8).npu()

data_indices = np.random.uniform(0, 1, [24]).astype(np.int32)
indices = torch.from_numpy(data_indices).to(torch.int32).npu()

data_updates = np.random.uniform(1, 2, [24, 1, 128]).astype(np.float16)
updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()

data_quant_scales = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()

data_quant_zero_points = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
quant_zero_points = torch.from_numpy(data_quant_zero_points).to(torch.bfloat16).npu()

axis = -2
quant_axis = -1
reduce = "update"

out = torch_npu.npu_quant_scatter(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
图模式调用
# 入图方式
import torch
import torch_npu
import math
import torchair as tng
import numpy as np
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
dtype_list2 =["fp16","int8","int32","fp32","fp16"]
dtype_list =[np.float16,np.int8,np.int32,np.float32]
updates_shape =[1,11,1,32]
var_shape =[1,11,1,32]
indices_shape =[1,2]
quant_scales_shape =[1,1,1,32]
quant_zero_points_shape =[1,1,1,32]
axis =-2
quant_axis=-1
reduce = "update"
updates_data = np.random.uniform(-1,1,size=updates_shape)
var_data = np.random.uniform(1,2,size=var_shape).astype(dtype_list[1])
quant_scales_data = np.random.uniform(1,2,size=quant_scales_shape)
indices_data = np.random.uniform(0,1,size=indices_shape).astype(dtype_list[2])
quant_zero_points_data = np.random.uniform(0,1,size=quant_zero_points_shape)
updates_npu = torch.from_numpy(updates_data).npu().to(torch.bfloat16).npu()
var_npu = torch.from_numpy(var_data).npu()
quant_scales_npu = torch.from_numpy(quant_scales_data).npu().to(torch.bfloat16).npu()
quant_zero_points_npu = torch.from_numpy(quant_zero_points_data).to(torch.bfloat16).npu()
indices_npu = torch.from_numpy(indices_data).npu()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
           0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
        [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
           1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
        [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
           0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
        [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
           1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
        [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
           0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
        [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
           0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
        [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
           1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
        [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
           0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
        [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
           1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
        [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
           0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
        [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
           1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
       device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
graph output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
           0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
        [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
           1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
        [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
           0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
        [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
           1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
        [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
           0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
        [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
           0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
        [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
           1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
        [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
           0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
        [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
           1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
        [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
           0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
        [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
           1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
       device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
"""
)

_add_torch_npu_docstr(
    "npu_quant_scatter_",
    """
功能描述:
先将updates进行量化, 然后将updates中的值按指定的轴axis和索引indices更新input中的值, input中的数据被改变. 

接口原型:
torch_npu.npu_quant_scatter_(Tensor(a!) input, Tensor indices, Tensor updates, Tensor quant_scales, Tensor? quant_zero_points=None, int axis=0, int quant_axis=1, str reduce='update') -> Tensor(a!)

参数说明:
input: Tensor类型, 必选输入, 源数据张量, 数据类型支持int8, 数据格式支持ND, 支持非连续的Tensor, 维数只能是3~8维. 
indices: Tensor类型, 必选输入, 索引张量, 数据类型支持int32, 数据格式支持ND, 支持非连续的Tensor. 
updates: Tensor类型, 必选输入, 更新数据张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、float16. 
quant_scales: Tensor类型, 必选输入, 量化缩放张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、float32. 
quant_zero_points: Tensor类型, 可选输入, 量化偏移张量, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持int32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持bfloat16、int32. 
axis: int类型, 可选参数, updates上用来更新的轴, 默认值为0. 
quant_axis: int类型, 可选参数, updates上用来量化的轴, 默认值为1. 
reduce: 字符串类型, 可选参数, 表示数据操作方式; 当前只支持'update', 即更新操作. 

输出说明:
一个Tensor类型的输出, 代表input被更新后的结果. 

约束说明:
该接口支持图模式. 
indices的维数只能是1维或者2维; 如果是2维, 其第2维的大小必须是2; 不支持索引越界, 索引越界不校验; indices映射的input数据段不能重合, 若重合则会因为多核并发原因导致多次执行结果不一样. 
updates的维数需要与input的维数一样; 其第1维的大小等于indices的第1维的大小, 且不大于input的第1维的大小; 其axis轴的大小不大于input的axis轴的大小; 其余维度的大小要跟input对应维度大小相等; 其最后一维的大小必须32B对齐. 
quant_scales的元素个数需要等于updates在quant_axis轴的大小. 
quant_zero_points的元素个数需要等于updates在quant_axis轴的大小. 
axis不能为updates的第1维或最后1维. 
quant_axis只能为updates的最后1维. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import numpy as np

data_var = np.random.uniform(0, 1, [24, 4096, 128]).astype(np.int8)
var = torch.from_numpy(data_var).to(torch.int8).npu()
data_indices = np.random.uniform(0, 1, [24]).astype(np.int32)
indices = torch.from_numpy(data_indices).to(torch.int32).npu()
data_updates = np.random.uniform(1, 2, [24, 1, 128]).astype(np.float16)
updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()
data_quant_scales = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()
data_quant_zero_points = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
quant_zero_points = torch.from_numpy(data_quant_zero_points).to(torch.bfloat16).npu()
axis = -2
quant_axis = -1
reduce = "update"

torch_npu.npu_quant_scatter_(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
图模式调用
# 入图方式
import torch
import torch_npu
import math
import torchair as tng
import numpy as np
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
dtype_list2 =["fp16","int8","int32","fp32","fp16"]
dtype_list =[np.float16,np.int8,np.int32,np.float32]
updates_shape =[1,11,1,32]
var_shape =[1,11,1,32]
indices_shape =[1,2]
quant_scales_shape =[1,1,1,32]
quant_zero_points_shape =[1,1,1,32]
axis =-2
quant_axis=-1
reduce = "update"
updates_data = np.random.uniform(-1,1,size=updates_shape)
var_data = np.random.uniform(1,2,size=var_shape).astype(dtype_list[1])
quant_scales_data = np.random.uniform(1,2,size=quant_scales_shape)
indices_data = np.random.uniform(0,1,size=indices_shape).astype(dtype_list[2])
quant_zero_points_data = np.random.uniform(0,1,size=quant_zero_points_shape)
updates_npu = torch.from_numpy(updates_data).npu().to(torch.bfloat16).npu()
var_npu = torch.from_numpy(var_data).npu()
quant_scales_npu = torch.from_numpy(quant_scales_data).npu().to(torch.bfloat16).npu()
quant_zero_points_npu = torch.from_numpy(quant_zero_points_data).to(torch.bfloat16).npu()
indices_npu = torch.from_numpy(indices_data).npu()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_quant_scatter_(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_quant_scatter_(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[ 0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,
           1,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1]],
        [[ 0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  1,  0,
           1,  1,  0,  1,  1,  0,  0, -1,  0,  1,  0,  1,  0,  1,  0]],
        [[ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,  1,  0,
           1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1]],
        [[ 0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,
           1,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  0,  0,  1]],
        [[ 0,  0,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  2,  0,
           1,  1,  0,  1,  1,  1,  1, -1,  0,  0,  0,  1,  1,  1,  0]],
        [[ 0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,  0,
           1,  1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  1,  1,  1,  0]],
        [[ 0, -1,  1,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  1,  2,  1,  0,
           1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0]],
        [[ 1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  2,  0,  1,  0,
           1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1]],
        [[ 0,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  1,  2,  1,  1,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  1,  0]],
        [[ 0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0, -1,  0,  1,  1,  0,  1,
           1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  0,  1,  1,  1,  0]],
        [[ 0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  1,  0,  1,  1,  1,  1,  0,
           1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1]]],
       device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
graph output with mask: tensor([[[ 0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,
           1,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1]],
        [[ 0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1,  1,  0,
           1,  1,  0,  1,  1,  0,  0, -1,  0,  1,  0,  1,  0,  1,  0]],
        [[ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  1,  1,  0,
           1,  0,  1,  0,  1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  1]],
        [[ 0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,
           1,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  0,  0,  1]],
        [[ 0,  0,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,  2,  0,
           1,  1,  0,  1,  1,  1,  1, -1,  0,  0,  0,  1,  1,  1,  0]],
        [[ 0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,  0,
           1,  1,  1,  0,  0,  1,  0, -1,  0,  0,  0,  1,  1,  1,  0]],
        [[ 0, -1,  1,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  1,  2,  1,  0,
           1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0]],
        [[ 1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  2,  0,  1,  0,
           1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1]],
        [[ 0,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  1,  2,  1,  1,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  1,  0]],
        [[ 0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  0, -1,  0,  1,  1,  0,  1,
           1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  0,  1,  1,  1,  0]],
        [[ 0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  1,  0,  1,  1,  1,  1,  0,
           1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1]]],
       device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
"""
)

_add_torch_npu_docstr(
    "npu_scatter_nd_update",
    """
功能描述:
将updates中的值按指定的索引indices更新input中的值，并将结果保存到输出tensor，input本身的数据不变。

接口原型:
torch_npu.npu_scatter_nd_update(Tensor input, Tensor indices, Tensor updates) -> Tensor

参数说明:
input：Tensor类型，必选输入，源数据张量，数据格式支持ND，支持非连续的Tensor，数据类型需要与updates一致，维数只能是1~8维。
        Atlas 推理系列加速卡产品：数据类型支持float32、float16、bool。
        Atlas 训练系列产品：数据类型支持float32、float16、bool。
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：数据类型支持float32、float16、bool、bfloat16、int64、int8。
        Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float32、float16、bool、bfloat16、int64、int8。
indices：Tensor类型，必选输入，索引张量，数据类型支持int32、int64，数据格式支持ND，支持非连续的Tensor，indices中的索引数据不支持越界。
updates：Tensor类型，必选输入，更新数据张量，数据格式支持ND，支持非连续的Tensor，数据类型需要与input一致。
        Atlas 推理系列加速卡产品：数据类型支持float32、float16、bool。
        Atlas 训练系列产品：数据类型支持float32、float16、bool。
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：数据类型支持float32、float16、bool、bfloat16、int64、int8。
        Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float32、float16、bool、bfloat16、int64、int8。

输出说明:
一个Tensor类型的输出，代表input被更新后的结果。

约束说明:
该接口支持图模式。
indices至少是2维，其最后1维的大小不能超过input的维度大小。
假设indices最后1维的大小是a，则updates的shape等于indices除最后1维外的shape加上input除前a维外的shape。举例：input的shape是(4, 5, 6)，indices的shape是(3, 2)，则updates的shape必须是(3, 6)。

支持的PyTorch版本:
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
Atlas 训练系列产品
Atlas 推理系列产品
Atlas A3 训练系列产品/Atlas A3 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import numpy as np

data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
var = torch.from_numpy(data_var).to(torch.float16).npu()
data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
indices = torch.from_numpy(data_indices).to(torch.int32).npu()
data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
updates = torch.from_numpy(data_updates).to(torch.float16).npu()

out = torch_npu.npu_scatter_nd_update(var, indices, updates)

图模式调用
import os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch.nn as nn
import torch
import numpy as np
import numpy
torch_npu.npu.set_compile_mode(jit_compile=True)

os.environ["ENABLE_ACLNN"] = "false"
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, var, indices, update):
        # 调用目标接口
        res = torch_npu.npu_scatter_nd_update(var, indices, update)
        return res
        
npu_mode = Network()
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

dtype = np.float32
x = [33 ,5]
indices = [33,25,1]
update = [33,25,5]

data_x = np.random.uniform(0, 1, x).astype(dtype)
data_indices = np.random.uniform(0, 10, indices).astype(dtype)
data_update = np.random.uniform(0, 1, update).astype(dtype)

tensor_x = torch.from_numpy(data_x).to(torch.float16)
tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
tensor_update = torch.from_numpy(data_update).to(torch.float16)

# 传参
print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu()))
"""
)

_add_torch_npu_docstr(
    "npu_scatter_nd_update_",
    """
功能描述:
将updates中的值按指定的索引indices更新input中的值，并将结果保存到输出tensor，input中的数据被改变。

接口原型:
torch_npu.npu_scatter_nd_update_(Tensor(a!) input, Tensor indices, Tensor updates) -> Tensor(a!)

参数说明:
input：Tensor类型，必选输入，源数据张量，数据格式支持ND，支持非连续的Tensor，数据类型需要与updates一致，维数只能是1~8维。
        Atlas 推理系列加速卡产品：数据类型支持float32、float16、bool。
        Atlas 训练系列产品：数据类型支持float32、float16、bool。
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：数据类型支持float32、float16、bool、bfloat16、int64、int8。
        Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float32、float16、bool、bfloat16、int64、int8。
indices：Tensor类型，必选输入，索引张量，数据类型支持int32、int64，数据格式支持ND，支持非连续的Tensor，indices中的索引数据不支持越界。
updates：Tensor类型，必选输入，更新数据张量，数据格式支持ND，支持非连续的Tensor，数据类型需要与input一致。
        Atlas 推理系列加速卡产品：数据类型支持float32、float16、bool。
        Atlas 训练系列产品：数据类型支持float32、float16、bool。
        Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：数据类型支持float32、float16、bool、bfloat16、int64、int8。
        Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持float32、float16、bool、bfloat16、int64、int8。
输出说明:
一个Tensor类型的输出，代表input被更新后的结果。

约束说明:
该接口支持图模式。
indices至少是2维，其最后1维的大小不能超过input的维度大小。
假设indices最后1维的大小是a，则updates的shape等于indices除最后1维外的shape加上input除前a维外的shape。举例：input的shape是(4, 5, 6)，indices的shape是(3, 2)，则updates的shape必须是(3, 6)。

支持的PyTorch版本:
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
Atlas 训练系列产品
Atlas 推理系列产品
Atlas A3 训练系列产品/Atlas A3 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import numpy as np

data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
var = torch.from_numpy(data_var).to(torch.float16).npu()
data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
indices = torch.from_numpy(data_indices).to(torch.int32).npu()
data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
updates = torch.from_numpy(data_updates).to(torch.float16).npu()

torch_npu.npu_scatter_nd_update_(var, indices, updates)

图模式调用
import os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch.nn as nn
import torch
import numpy as np
import numpy
torch_npu.npu.set_compile_mode(jit_compile=True)

os.environ["ENABLE_ACLNN"] = "false"
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, var, indices, update):
        # 调用目标接口
        res = torch_npu.npu_scatter_nd_update_(var, indices, update)
        return res
        
npu_mode = Network()
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

dtype = np.float32
x = [33 ,5]
indices = [33,25,1]
update = [33,25,5]

data_x = np.random.uniform(0, 1, x).astype(dtype)
data_indices = np.random.uniform(0, 10, indices).astype(dtype)
data_update = np.random.uniform(0, 1, update).astype(dtype)

tensor_x = torch.from_numpy(data_x).to(torch.float16)
tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
tensor_update = torch.from_numpy(data_update).to(torch.float16)

# 传参
print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu()))
"""
)

_add_torch_npu_docstr(
    "npu_scatter_pa_kv_cache",
    """
功能描述:
-   更新KvCache中指定位置的key和value。

-   输入输出支持以下场景：
    - 场景一：
        key:[batch, num_head, k_head_size]
        value:[batch, num_head, v_head_size]
        key_cache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
        value_cache:[num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
        slot_mapping:[batch]

    - 场景二：
        key:[batch*seq_len, num_head, k_head_size]
        value:[batch*seq_len, num_head, v_head_size]
        keyCache:[num_blocks, block_size, num_head, k_head_size]
        valueCache:[num_blocks, block_size, num_head, v_head_size]
        slotMapping:[batch * seq_len]

    其中k_head_size与v_head_size 可以不同，也可以相同。
    
    - 场景三：
        key:[batch, seq_len, num_head, k_head_size]
        value:[batch, seq_len, num_head, v_head_size]
        key_cache:[num_blocks, block_size, 1, k_head_size]
        value_cache:[num_blocks, block_size, 1, k_head_size]
        slot_mapping:[batch, num_head]
        compress_lens:[batch, num_head]
        seq_lens:[batch]
        compress_seq_offsets:[batch * num_head]

    - 上述场景根据构造的参数来区别，符合第一种入参构造走场景一，符合第二种构造走场景二，符合第三种构造走场景三。
        场景一、场景二没有compress_lens、seq_lens、compress_seq_offsets这三个可选参数。

接口原型:
torch_npu.npu_scatter_pa_kv_cache(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slot_mapping, *, Tensor? compress_lens=None, Tensor? compress_seq_offsets=None, Tensor? seq_lens=None) -> ()

参数说明:
-   key（Tensor）：必选参数。表示待更新的key值，当前step多个token的key，支持3维或4维。数据类型支持float16、float、bfloat16、int8、uint8、int16、uint16、int32、uint32、hifloat8、float8_e5m2、float8_e4m3fn，数据格式支持ND。
-   value（Tensor）：必选参数，表示待更新的value值，当前step多个token的value，支持3维或4维，数据类型和数据格式与key保持一致。
-   key_cache（Tensor）：必选参数，表示需要更新的key cache，当前layer的key cache，只支持4维，数据类型和数据格式与key保持一致。
-   value_cache（Tensor）：必选参数，表示需要更新的value cache，当前layer的value cache。只支持4维，数据类型和数据格式与key保持一致。
-   slot_mapping（Tensor）：必选参数，表示每个token key或value在cache中的存储偏移，数据类型支持int32和int64，数据格式支持ND。
-   compress_lens（Tensor）：可选参数，表示压缩量，数据类型与slot_mapping一致，数据格式支持ND。
-   compress_seq_offsets（Tensor）：可选参数，表示每个batch每个head的压缩起点，数据类型与slot_mapping一致，数据格式支持ND。
-   seq_lens（Tensor）：可选参数，表示每个batch的实际seqLens，数据类型与slot_mapping一致，数据格式支持ND。。

输出说明：
-   key_cache（Tensor）：表示Key输出到kv_cache中的Tensor（本质in-place更新）。
-   value_cache（Tensor）：表示value输出到kv_cache中的Tensor（本质in-place更新）。

约束说明:
-   输入参数不支持非连续；
-   key、value、key_cache、value_cache的数据类型必须一致；
-   slot_mapping、compress_lens、compress_seq_offset、seq_lens的数据类型必须一致；
-   slot_mapping的值范围[0,num_blocks*block_size-1]，且slot_mapping内的元素值保证不重复，重复时不保证正确性；
-   当key和value都是3维，则key和value的前两维shape必须相同；
-   当key和value都是4维，则key和value的前三维shape必须相同，且key_cache和value_cache的第三维必须是1；
-   当key和value是4维时，compress_lens、seq_lens为必选参数；当key和value是3维时，compress_lens、compress_seq_offsets、seq_lens为可选参数；
-   当key和value都是4维时，slot_mapping是二维，且slot_mapping的第一维值等于key的第一维为batch，slot_mapping的第二维值等于key的第三维为num_head(对应场景三)；
-   当key和value都是4维时，seq_lens是一维，且seq_lens的值等于key的第一维为batch(对应场景三)；
-   seq_lens和compress_lens里面的每个元素值必须满足公式：reduceSum(seq_lens[i] - compress_lens[i]) <= num_blocks * block_size (对应场景三);

支持的型号:
Atlas A3 训练系列产品/Atlas A3 推理系列产品
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件

调用示例:
# 单算子调用方式
import numpy as np
import torch
import torch_npu

# 生成随机数据, 并发送到npu
bs = 16
num_head = 4
k_head_size = 32
v_head_size = 64
num_blocks = 2
lastDim_k = 16
block_size = 32

key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
key_cache = np.random.randn(
    num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
value_cache = np.zeros(
    (num_blocks, num_head * v_head_size // lastDim_k, block_si ze, lastDim_k)).astype(np.float16)
slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

key_npu = torch.from_numpy(key).npu()
value_npu = torch.from_numpy(value).npu()
key_cache_npu = torch.from_numpy(key_cache).npu()
value_cache_npu = torch.from_numpy(value_cache).npu()
key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

# 调用ScatterPaKvCache算子
torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu_cast, value_cache_npu_cast, slot_mapping_npu)

# 执行上述代码，打印key_cache_npu_cast或value_cache_npu_cast输出类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [[ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]],
        ...,
        [[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120],
        ...,
        [[ 0.1113,  0.6255,  0.7686,  ...,  0.0247,  0.2490,  0.6909],
        [ 0.4312,  0.7954,  0.7339,  ...,  0.1154,  0.6440,  0.3342],
        [ 0.9570,  0.2869,  0.6489,  ...,  0.7451,  0.0234,  0.8843]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120]]]]
    device='npu:0', dtype=torch.float16)

# 入图方式
import numpy as np
import torch
import torch_npu

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "2"
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
bs = 16
num_head = 4
k_head_size = 32
v_head_size = 64
num_blocks = 2
lastDim_k = 16
block_size = 32

class Model(torch.nn.Module):
    def init(self):
        super().init()

    def forward(self, key, value, slot_mapping, key_cache, value_cache):
            torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)

if name=="main":
    torch_npu.npu.set_device(0)

    key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
    value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
    key_cache = np.random.randn(
        num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
    value_cache = np.zeros(
        (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
    slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

    key_npu = torch.from_numpy(key).npu()
    value_npu = torch.from_numpy(value).npu()
    key_cache_npu = torch.from_numpy(key_cache).npu()
    value_cache_npu = torch.from_numpy(value_cache).npu()
    key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
    value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
    slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    cpu_model = Model().npu()
    # 图模式调用
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    model(key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast)


# 执行上述代码，打印key_cache_npu_cast或value_cache_npu_cast输出类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [[ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]],
        ...,
        [[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120],
        ...,
        [[ 0.1113,  0.6255,  0.7686,  ...,  0.0247,  0.2490,  0.6909],
        [ 0.4312,  0.7954,  0.7339,  ...,  0.1154,  0.6440,  0.3342],
        [ 0.9570,  0.2869,  0.6489,  ...,  0.7451,  0.0234,  0.8843]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120]]]]
    device='npu:0', dtype=torch.float16)
"""
)

_add_torch_npu_docstr(
    "npu_anti_quant",
    """
功能描述:
算子功能: 对张量x进行反量化操作, 即将整数恢复为浮点数. 
计算公式为: anti_quant(x)=quant((x+offset)*scale)

接口原型:
torch_npu.npu_anti_quant(Tensor x, Tensor scale, *, Tensor? offset=None, ScalarType? dst_dtype=None, ScalarType? src_dtype=None) -> Tensor

参数说明:
x: Tensor类型, 即输入参数中的x. 数据格式支持ND, 支持非连续的Tensor. 输入最大支持8维. 
Atlas 推理系列产品: 数据类型支持int8. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、int32, 其中int32类型数据的每个值是由8个int4数值拼成. 
Atlas A3 训练系列产品: 数据类型支持int8、int32, 其中int32类型数据的每个值是由8个int4数值拼成. 
scale: Tensor类型, 反量化参数scale. 仅支持1维Tensor, shape为(n,). 其中n可以为1, 如果n不为1, 当x为int8类型, 必须与输入x的尾轴维度大小相同; 当x为int32类型时, 必须为输入x的尾轴维度大小的8倍. 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、bfloat16. 
Atlas A3 训练系列产品: 数据类型支持float32、bfloat16. 
offset: Tensor类型, 可选参数, 反量化参数offset. 仅支持1维Tensor, 数据类型和shape必须与scale一致. 数据格式支持ND, 支持非连续的Tensor. 
dst_dtype: ScalarType类型, 可选参数, 指定输出的数据类型, 默认值为float16. 
Atlas 推理系列产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16. 
src_dtype: ScalarType类型, 可选参数, 指定源输入的数据类型, 默认值为int8. 
Atlas 推理系列产品: 数据类型支持int8. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持quint4x2或int8. 
Atlas A3 训练系列产品: 数据类型支持quint4x2或int8. 

输出说明:
一个Tensor类型的输出, 代表antiquant的计算结果. 

约束说明:
该接口支持推理、训练场景下使用. 
该接口支持图模式. 
x、scale这两个输入中不能含有None. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
x_tensor = torch.tensor([1,2,3,4], dtype=torch.int8).npu()
scale = torch.tensor([2.0], dtype=torch.float).npu()
offset = torch.tensor([2.0], dtype=torch.float).npu()
out = torch_npu.npu_anti_quant(x_tensor, scale, offset=offset, dst_dtype=torch.float16)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = 'pbtxt'
npu_backend = tng.get_npu_backend(compiler_config=config)
x_tensor = torch.tensor([1,2,3,4], dtype=torch.int8).npu()
scale = torch.tensor([2.0], dtype=torch.float).npu()
offset = torch.tensor([2.0], dtype=torch.float).npu()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,scale,offset):
        return torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=torch.float16)
cpu_model = Model()
model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
output = model(x_tensor,scale,offset)
"""
)

_add_torch_npu_docstr(
    "npu_mm_all_reduce_base",
    """
功能描述:
TP切分场景下, 实现mm和all_reduce的融合, 融合算子内部实现计算和通信流水并行. 
使用该接口时, 请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本, 否则将会引发报错, 比如BUS ERROR等.

接口原型:
torch_npu.npu_mm_all_reduce_base(Tensor x1, Tensor x2, str hcom, *, str reduce_op='sum', Tensor? bias=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? x3=None, Tensor? dequant_scale=None Tensor? pertoken_scale=None, Tensor? comm_quant_scale_1=None, Tensor? comm_quant_scale_2=None, int comm_turn=0, int antiquant_group_size=0) -> Tensor

参数说明:
x1: Tensor类型, 数据类型支持int8、float16、bfloat16. 数据格式支持ND, 输入shape支持2维或者3维. 
x2: Tensor类型, 数据类型支持float16、int8、bfloat16, 数据格式支持NZ(昇腾亲和排布格式)、ND. 非量化场景, 数据类型需要和x1保持一致, 输入shape维度第0维和x1的最后一维保持一致. 
hcom: String类型, 通信域handle名, 通过get_hccl_comm_name接口获取. 
*: 代表其之前的变量是位置相关, 按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
reduce_op: String类型, reduce操作类型, 当前版本仅支持'sum', 默认值: 'sum'. 
bias: Tensor类型, 可选输入, 数据类型支持int32、float16、bfloat16, 数据格式支持ND. bias当前仅支持一维, 且维度大小与output/x2的最后一维大小相同. 
antiquant_scale: Tensor类型, 可选输入, 伪量化场景对x2进行去量化的系数, 数据类型支持float16、bfloat16, 数据格式支持ND. 伪量化场景数据类型需要和x1保持一致. 
per-tensor场景: shape为[1]. 
per-channel场景: shape为[1,n]或者[n], n为x2最后一维的大小. 
per-group场景: shape为[ceil(k, antiquant_group_size), n]. 其中k为x2第一维的大小, n为x2最后一维的大小, antiquant_group_size为伪量化场景对输入x2进行反量化计算的groupSize输入. 
ceil(k, antiquant_group_size)的计算逻辑为: (k+antiquant_group_siz-1)/antiquant_group_size, 并对计算结果取整数部分. 
antiquant_offset: Tensor类型, 可选输入, 伪量化场景对x2进行去量化的系数, 数据类型支持float16、bfloat16, 数据格式支持ND. 数据类型、shape需要和antiquant_scale保持一致. 
x3: Tensor类型, 可选输入, matmul计算后的偏移. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16, 数据格式支持ND. 数据类型、shape需要和输出output保持一致. 
dequant_scale: Tensor类型, 可选输入, matmul计算后的去量化系数. 数据类型支持int64、uint64、bfloat16、float32; 数据格式支持ND. 
per-tensor场景: shape为[1]. 
per-channel场景: shape为[n]/[1,n], n为x2最后一维的大小. 
pertoken_scale: Tensor类型, 可选输入, matmul计算后的per-token去量化系数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32. 当x1为[m,k]时pertoken_scale shape为[m]; 当x1为[b, s, k]时pertoken_scale shape为[b*s]. 
comm_quant_scale_1: Tensor类型, 可选输入, alltoall通信前后的量化、去量化系数. 支持float16、bfloat16, 支持ND格式. x2为[k, n]时shape为[1, n]或[n], 用户需保证每张卡上数据保持一致且正确. 
comm_quant_scale_2: Tensor类型, 可选输入, allgather通信前后的量化、去量化系数. 支持float16、bfloat16, 支持ND格式. x2为[k, n]时shape为[1, n]或[n], 用户需保证每张卡上数据保持一致且正确. 
comm_turn: int类型, 表示rank间通信切分粒度, 默认值: 0, 表示默认的切分方式. 当前版本仅支持输入0. 
antiquant_group_size: int类型, 表示伪量化pre-group算法模式下, 对输入x2进行反量化计算的groupSize输入, 描述一组反量化参数对应的待反量化数据量在k轴方向的大小. 当伪量化算法模式不为pre-group时传入0; 当伪量化算法模式为pre-group时传入值的范围为[32, min(k-1, INT_MAX)]且值要求是32的倍数, 其中k为x2第一维的大小. 默认值0, 为0则表示非per-group场景. 

输出说明
Tensor类型, 数据类型非量化场景以及伪量化场景与x1保持一致, 全量化场景输出数据类型为float16或bfloat16. shape第0维度和x1的0维保持一致, 若x1为2维, shape第1维度和x2的1维保持一致, 若x1为3维, shape第1维度和x1的1维保持一致, shape第2维度和x2的1维保持一致. 

约束说明
增量场景不使能该融合算子, 全量场景使能该融合算子. 
该接口支持推理场景下使用. 
该接口支持图模式. 
输入x1可为2维或者3维、x2必须是2维, 分别为(b, s, k)/(m, k), (k, n), k轴满足mm算子入参要求, k轴相等. bias当前仅支持一维, 且维度大小与output的最后一维大小相同. x3的shape与output的shape相同. 
x1不支持输入转置后的tensor, x2转置后输入, 需要满足shape的第一维大小与x1的最后一维相同, 满足matmul的计算条件. 
antiquant_group_size中k值的范围与matmul一致, 为[1,65535], INT_MAX大于(k-1). 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 
数据类型支持bfloat16. 
x1、x2不支持为空tensor. 
支持1、2、4、8卡, 并且仅支持hccs链路all mesh组网. 
非量化场景下, m、k、n的取值范围均为[1, 2147483647]. 
comm_quant_scale_1, comm_quant_scale_2的shape应保持一致, dtype与输出的dtype保持一致, 且只在全量化场景支持. 
全量化场景: m取值范围均为[1, 2147483647], x1、x2的最后一维范围为[1, 65535], 即k的取值范围为[1, 65535]、仅当x2(shape=[n,k])为转置时n可以大于65535. 
伪量化场景: m取值范围均为[1, 2147483647], k、n的取值范围为[1, 65535]. 
Atlas A2 训练系列产品: 一个模型中的通算融合算子(AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce), 仅支持相同通信域. 
在长序列场景, 随着b/s或者m的增大, 可能出现内存不足或者计算超时. 
不同场景下数据类型支持情况: 
表1 非量化场景产品型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
x1: float16
x2: float16
bias: float16
x3: float16
output(输出): float16
antiquant_scale: None
antiquant_offset: None
dequant_scale: None
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
x1: bfloat16
x2: bfloat16
bias: bfloat16
x3: bfloat16
output(输出): bfloat16
antiquant_scale: None
antiquant_offset: None
dequant_scale: None
表2 伪量化场景产品型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
x1: float16
x2: int8
bias: float16
x3: float16
output(输出): float16
antiquant_scale: float16
antiquant_offset: float16
dequant_scale: None
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
x1: bfloat16
x2: int8
bias: bfloat16
x3: bfloat16
output(输出): bfloat16
antiquant_scale: bfloat16
antiquant_offset: bfloat16
dequant_scale: None
表3 全量化场景产品型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
x1: int8, int8, int8, int8
x2: int8, int8, int8, int8
bias: int32, int32, int32, int32
x3: float16, bfloat16, float16, bfloat16
output(输出): float16, bfloat16, float16, bfloat16
antiquant_scale: None, None, None, None
antiquant_offset: None, None, None, None
dequant_scale: uint64或int64, bfloat16, float32, bfloat16
pertoken_scale: None, None, float32, float32
全量化场景: 若dequant_scale需要以FP32类型传入, 在调用torch_npu.npu_mm_all_reduce_base前, 需通过torch_npu.npu_trans_quant_param接口对dequant_scale进行处理为int64类型(处理方法见对应的接口使用说明). 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.1
PyTorch 1.11.0

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例:
单算子模式调用
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    input_ = torch.randn(x1_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op='sum')
    print("output: ", output)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16

    mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
图模式调用
非量化、伪量化、全量化使能NZ调用示例如下: 
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
class MM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
        output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                      x2=x2,
                                                      hcom=hcom,
                                                      reduce_op=reduce_op,
                                                      bias=bias,
                                                      antiquant_scale=antiquant_scale,
                                                      antiquant_offset=antiquant_offset,
                                                      x3=x3,
                                                      dequant_scale=dequant_scale
                                                      )
        return output_npu

class MM_ALLREDUCE_A8W8_GRAPH_Model(MM_ALLREDUCE_GRAPH_Model):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
        output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                      x2=x2.t(),
                                                      hcom=hcom,
                                                      reduce_op=reduce_op,
                                                      bias=bias,
                                                      antiquant_scale=antiquant_scale,
                                                      antiquant_offset=antiquant_offset,
                                                      x3=x3,
                                                      dequant_scale=dequant_scale
                                                      )
        return output_npu

def define_model(model, graph_type):
    import torchair
    if graph_type == 1:  # 传统入图模式, 静态shape+在线编译场景
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  # ACLNN入图模式, 动态shape+二进制
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    else:
        print("Error type")
    return model

def get_graph(input, weight, hcomm_info, dequant_scale, bias, antiquant_scale, antiquant_offset, x3):
    model = MM_ALLREDUCE_A8W8_GRAPH_Model()
    model = define_model(model, 2) # 1:静态入图;2:动态入图;
    output = model(x1=input, x2=weight, hcom=hcomm_info, reduce_op="sum", bias=bias, antiquant_scale=antiquant_scale,
                   antiquant_offset=antiquant_offset, x3=x3, dequant_scale=dequant_scale)
    return output

def run_mc2_a16w16(x1_shape, x2_shape, hcom_info):
    np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
    np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.float16)
    input = torch.tensor(np_input).npu()
    weight = torch.tensor(np_weight).npu()
    output_a16w16 = get_graph(input, weight, hcom_info, None, None, None, None, None)
    return output_a16w16

def run_mc2_a8w8(x1_shape, x2_shape, hcom_info):
    np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.int8)
    np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
    input = torch.tensor(np_input).npu()
    weight = torch.tensor(np_weight).npu()
    weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
    dequant_scale = torch.randn(x2_shape[0], dtype=torch.float32).uniform_(float(-10), float(10)).npu()
    dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
    output_a8w8 = get_graph(input, weight_nz, hcom_info, dequant_scale, None, None, None, None)
    return output_a8w8

def run_mc2_a16w8(x1_shape, x2_shape, hcom_info):
    np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
    np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
    input = torch.tensor(np_input).npu()
    weight = torch.tensor(np_weight).npu()
    weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
    antiquant_scale = torch.randn(x2_shape[0], dtype=torch.float16).uniform_(float(-1), float(1)).npu()
    antiquant_offset = torch.ones(x2_shape[0], dtype=torch.float16).npu()
    output_a16w8 = get_graph(input, weight_nz, hcom_info, None, None, antiquant_scale, antiquant_offset, None)
    return output_a16w8

def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, op_type):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)
    output = None
    # 非量化调用
    if op_type == "a16w16":
        output = run_mc2_a16w16(x1_shape, x2_shape, hcom_info)
    # 伪量化调用
    if op_type == "a16w8":
        output = run_mc2_a16w8(x1_shape, x2_shape, hcom_info)
    # 全量化调用
    if op_type == "a8w8":
        output = run_mc2_a8w8(x1_shape, x2_shape, hcom_info)
    print("output:", output)
if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [1280, 5120]
    x2_shape = [640, 5120]
    op_type = "a16w8" # Options: a16w16, a16w8, a8w8
    mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, op_type), nprocs=worksize)
"""
)

_add_torch_npu_docstr(
    "npu_ffn",
    """
功能描述:
算子功能: 该FFN算子提供MoeFFN和FFN的计算功能. 在没有专家分组(expert_tokens为空)时是FFN, 有专家分组时是MoeFFN. 
计算公式: 
out=activation(xW1+b1)W2+b2
激活层为geglu/swiglu/reglu时, 性能使能需要满足门槛要求, 即整网中FFN结构所对应的小算子中vector耗时30us且占比10%以上的用例方可尝试FFN融合算子; 或在不知道小算子性能的情况下, 尝试使能FFN, 若性能劣化则不使能FFN. 

接口原型:
torch_npu.npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, int[]? expert_tokens=None, int[]? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None, Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None, Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None, int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor

参数说明:
x: Tensor类型, 输入参数, 公式中的x, 数据类型支持float16、bfloat16、int8, 数据格式支持ND, 支持输入的维度最少是2维[M, K1], 最多是8维. 
weight1: Tensor类型, 专家的权重数据, 公式中的W1, 数据类型支持float16、bfloat16、int8, 数据格式支持ND, 输入在有/无专家时分别为[E, K1, N1]/[K1, N1]. 
weight2: Tensor类型, 专家的权重数据, 公式中的W2, 数据类型支持float16、bfloat16、int8, 数据格式支持ND, 输入在有/无专家时分别为[E, K2, N2]/[K2, N2]. 
M表示token个数, 对应transform中的BS(B: Batch, 表示输入样本批量大小, S: Seq-Length, 表示输入样本序列长度); K1表示第一个matmul的输入通道数, 对应transform中的H(Head-Size, 表示隐藏层的大小); N1表示第一个matmul的输出通道数; K2表示第二个matmul的输入通道数; N2表示第二个matmul的输出通道数, 对应transform中的H; E表示有专家场景的专家数. 
expert_tokens: List类型, 可选参数. 代表各专家的token数, 数据类型支持int32, 数据格式支持ND, 若不为空时可支持的最大长度为256个. 
expert_tokens_index: List类型, 可选参数. 代表各专家计算token的索引值, 数据类型支持int32, 数据格式支持ND, 若不为空时可支持的最大长度为256个. 
bias1: Tensor类型, 可选参数. 权重数据修正值, 公式中的b1, 数据类型支持float16、float32、int32, 数据格式支持ND, 输入在有/无专家时分别为[E, N1]/[N1]. 
bias2: Tensor类型, 可选参数. 权重数据修正值, 公式中的b2, 数据类型支持float16、float32、int32, 数据格式支持ND, 输入在有/无专家时分别为[E, N2]/[N2]. 
activation: string类型, 代表使用的激活函数, 即输入参数中的activation. 当前仅支持fastgelu、gelu、relu、silu、geglu、swiglu、reglu. 
scale: Tensor类型, 可选参数, 量化参数, 量化缩放系数, 数据类型支持float32, 数据格式支持ND. per-tensor下输入在有/无专家时均为一维向量, 输入元素个数在有/无专家时分别为[E]/[1]; per-channel下输入在有/无专家时为二维向量/一维向量, 输入元素个数在有/无专家时分别为[E, N1]/[N1]. 
offset: Tensor类型, 可选参数, 量化参数, 量化偏移量, 数据类型支持float32, 数据格式支持ND, 一维向量, 输入元素个数在有/无专家时分别为[E]/[1]. 
deq_scale1: Tensor类型, 可选参数, 量化参数, 第一组matmul的反量化缩放系数, 数据类型支持int64、float32、bfloat16, 数据格式支持ND, 输入在有/无专家时分别为[E, N1]/[N1]. 
deq_scale2: Tensor类型, 可选参数, 量化参数, 第二组matmul的反量化缩放系数, 数据类型支持int64、float32、bfloat16, 数据格式支持ND, 输入在有/无专家时分别为[E, N2]/[N2]. 
antiquant_scale1: Tensor类型, 可选参数, 伪量化参数, 第一组matmul的缩放系数, 数据类型支持float16、bfloat16, 数据格式支持ND, per-channel下输入在有/无专家时分别为[E, N1]/[N1]. 
antiquant_scale2: Tensor类型, 可选参数, 伪量化参数, 第二组matmul的缩放系数, 数据类型支持float16、bfloat16, 数据格式支持ND, per-channel下输入在有/无专家时分别为[E, N2]/[N2]. 
antiquant_offset1: Tensor类型, 可选参数, 伪量化参数, 第一组matmul的偏移量, 数据类型支持float16、bfloat16, 数据格式支持ND, per-channel下输入在有/无专家时分别为[E, N1]/[N1]. 
antiquant_offset2: Tensor类型, 可选参数, 伪量化参数, 第二组matmul的偏移量, 数据类型支持float16、bfloat16, 数据格式支持ND, per-channel下输入在有/无专家时分别为[E, N2]/[N2]. 
inner_precise: int类型, 可选参数, 表示高精度或者高性能选择. 数据类型支持int64. 该参数仅对float16生效, bfloat16和int8不区分高精度和高性能. 
inner_precise为0时, 代表开启高精度模式, 算子内部采用float32数据类型计算. 
inner_precise为1时, 代表高性能模式. 
inner_precise参数在bfloat16非量化场景, 只能配置为0; float16非量化场景, 可以配置为0或者1; 量化或者伪量化场景, 0和1都可配置, 但是配置后不生效. 
output_dtype: ScalarType类型, 可选参数, 该参数只在量化场景生效, 其他场景不生效. 表示输出Tensor的数据类型, 支持输入float16、bfloat16. 默认值为None, 代表输出Tensor数据类型为float16. 

输出说明:
一个Tensor类型的输出, 公式中的输出y, 数据类型支持float16、bfloat16, 数据格式支持ND, 输出维度与x一致. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
有专家时, 专家数据的总数需要与x的M保持一致. 
激活层为geglu/swiglu/reglu时, 仅支持无专家分组时的float16高性能场景(float16场景指类型为Tensor的必选参数数据类型都为float16的场景), 且N1=2*K2. 
激活层为gelu/fastgelu/relu/silu时, 支持有专家或无专家分组的float16高精度及高性能场景, bfloat16场景, 量化场景及伪量化场景, 且N1=K2. 
所有场景下需满足K1=N2、K1<65536、K2<65536、M轴在32Byte对齐后小于int32的最大值. 
非量化场景不能输入量化参数和伪量化参数, 量化场景不能输入伪量化参数, 伪量化场景不能输入量化参数. 
量化场景参数类型: x为int8、weight为int8、bias为int32、scale为float32、offset为float32, 其余参数类型根据y不同分两种情况: 
y为float16, deqScale支持数据类型uint64、int64、float32. 
y为bfloat16, deqScale支持数据类型bfloat16. 
要求deqScale1与deqScale2的数据类型保持一致. 
量化场景支持scale的per-channel模式参数类型: x为int8、weight为int8、bias为int32、scale为float32、offset为float32, 其余参数类型根据y不同分两种情况: 
y为float16, deqScale支持数据类型uint64、int64. 
y为bfloat16, deqScale支持数据类型bfloat16. 
要求deqScale1与deqScale2的数据类型保持一致. 
伪量化场景支持两种不同参数类型: 
y为float16、x为float16、bias为float16、antiquant_scale为float16、antiquant_offset为float16、weight支持数据类型int8. 
y为bfloat16、x为bfloat16、bias为float32、antiquant_scale为bfloat16、antiquant_offset为bfloat16、weight支持数据类型int8. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11.0

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import logging
import os
cpu_x = torch.randn((1, 1280), device='npu', dtype=torch.float16)
cpu_weight1 = torch.randn(1280, 10240, device='npu', dtype=torch.float16)
cpu_weight2 = torch.randn(10240, 1280, device='npu', dtype=torch.float16)
activation = "fastgelu"
npu_out = torch_npu.npu_ffn(cpu_x.npu(), cpu_weight1.npu(), cpu_weight2.npu(), activation, inner_precise=1)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
import os
os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, weight1, weight2, activation, expert):
        return torch_npu.npu_ffn(x, weight1, weight2, activation,  expert_tokens=expert, inner_precise=1)
cpu_model = MyModel()
cpu_x = torch.randn((1954, 2560),device='npu',dtype=torch.float16)
cpu_weight1 = torch.randn((16, 2560, 5120),device='npu',dtype=torch.float16)
cpu_weight2 = torch.randn((16, 5120, 2560),device='npu',dtype=torch.float16)
activation = "fastgelu"
expert = [227, 62, 78, 126, 178, 27, 122, 1, 19, 182, 166, 118, 66, 217, 122, 243]
model = cpu_model.npu()
model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
npu_out = model(cpu_x.npu(), cpu_weight1.npu(), cpu_weight2.npu(), activation, expert)
"""
)

_add_torch_npu_docstr(
    "npu_incre_flash_attention",
    """
功能描述:
增量FA实现, 实现对应公式: 
atten_out=softmax(scale*(query*key)+atten_mask)*value

接口原型:
torch_npu.npu_incre_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? padding_mask=None, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? kv_padding_size=None, int num_heads=1, float scale_value=1.0, str input_layout="BSH", int num_key_value_heads=0, int block_size=0, int inner_precise=1) -> Tensor

参数说明:
query: Tensor类型, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
key: Tensor类型, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16、bfloat16、int8. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
value: Tensor类型, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16、int8. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
*: 代表其之前的变量是位置相关, 需要按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
padding_mask: Tensor类型, 预留参数, 暂未使用, 默认值为None. 
pse_shift: Tensor类型, 表示在attention结构内部的位置编码参数, 数据格式支持ND. 如不使用该功能时可不传或传入None. 
Atlas 推理系列加速卡产品: 仅支持None. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
atten_mask: Tensor类型, 取值为1代表该位不参与计算(不生效), 为0代表该位参与计算, 默认值为None, 即全部参与计算; 数据类型支持bool、int8、uint8, 数据格式支持ND. 
actual_seq_lengths: int型数组, 其shape为(B,)或(1,), 形如[1, 2, 3], 代表key、value中有效的S序列长度, 默认值为None, 即全部有效, 类型为List int; 数据类型为int64, 数据格式支持ND. 
dequant_scale1: Tensor类型, 数据类型支持float32, 数据格式支持ND, 表示BMM1后面反量化的量化因子, 支持per-tensor(scalar).  如不使用该功能时可不传或传入None. Atlas 推理系列加速卡产品暂不使用该参数. 
quant_scale1: Tensor类型, 数据类型支持float32, 数据格式支持ND, 表示BMM2前面量化的量化因子, 支持per-tensor(scalar).  如不使用该功能时可不传或传入None. Atlas 推理系列加速卡产品暂不使用该参数. 
dequant_scale2: Tensor类型, 数据类型支持float32, 数据格式支持ND, 表示BMM2后面反量化的量化因子, 支持per-tensor(scalar).  如不使用该功能时可不传或传入None. Atlas 推理系列加速卡产品暂不使用该参数. 
quant_scale2: Tensor类型, 数据格式支持ND, 表示输出量化的量化因子, 支持per-tensor(scalar)和per-channel(list).  如不使用该功能时可不传或传入None. 
Atlas 推理系列加速卡产品: 当前版本不支持. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、bfloat16. 
quant_offset2: Tensor类型, 数据格式支持ND, 表示输出量化的量化偏移, 支持per-tensor(scalar)和per-channel(list).  如不使用该功能时可不传或传入None. 
Atlas 推理系列加速卡产品: 当前版本不支持. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、bfloat16. 
antiquant_scale: Tensor类型, 数据格式支持ND, 表示量化因子, 支持per-channel(list), 由shape决定, BNSD场景下shape为(2, N, 1, D), BSH场景下shape为(2, H), BSND场景下shape为(2, N, D).  如不使用该功能时可不传或传入None. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
antiquant_offset: Tensor类型, 数据格式支持ND, 表示量化偏移, 支持per-channel(list), 由shape决定, BNSD场景下shape为(2, N, 1, D), BSH场景下shape为(2, H), BSND场景下shape为(2, N, D).  如不使用该功能时可不传或传入None. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 
block_table: Tensor类型, 数据类型支持int32, 数据格式支持ND. block_table为2维Tensor, 表示PageAttention中KV存储使用的block映射表, 具体约束和使用方法可见约束说明.  如不使用该功能时可不传或传入None. 
kv_padding_size: Tensor类型, 数据类型支持int64, 数据格式支持ND, 表示kv左padding场景使能时, 最后一个有效token到S的距离.  如不使用该功能时可传入None. 
num_heads: int类型, 代表query的头数, 即query的N, 默认值为1; 数据类型为int64. 
scale_value: float类型, 代表缩放系数, 用来约束梯度, 其默认值为1.0, 典型值为$\frac{1}{\sqrt{D}}$; 数据类型为float32. 
input_layout: 字符串类型, 代表query、key、value的布局, 根据输入的query、key、value的shape确定, 三维Tensor是BSH, 四维Tensor是BNSD或BSND, 默认值为BSH, 不支持其他值; 数据类型为string. 
query、key、value数据排布格式支持从多种维度解读, 其中B(Batch)表示输入样本批量大小、S(Seq-Length)表示输入样本序列长度、H(Head-Size)表示隐藏层的大小、N(Head-Num)表示多头数、D(Head-Dim)表示隐藏层最小的单元尺寸, 且满足D=H/N. 
num_key_value_heads: int类型, 代表key、value的头数, 用于支持GQA(Grouped-Query Attention, 分组查询注意力)场景, 默认值为0, 表示与query的头数相同, 否则表示key、value的头数, 需要能被query的头数(num_heads)整除; num_heads与num_key_value_heads的比值不能大于64. 数据类型为int64. 
block_size: int类型, PageAttention中KV存储每个block中最大的token个数, 默认为0, 通常为128、256等值, 数据类型支持int64. 
inner_precise: int类型, 代表高精度/高性能选择, 0代表高精度, 1代表高性能, 默认值为1(高性能),  数据类型支持int64. 

输出说明:
atten_out: Tensor类型, 计算的最终结果, shape与query保持一致. 
非量化场景下, 输出数据类型与query的数据类型保持一致. 
量化场景下, 若传入quant_scale2, 则输出数据类型为int8. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
query、key、value的维度必须保持一致, key、value的shape必须保持一致. 
num_heads的值要等于query的N. 
input_layout的值与query的shape相关, 三维是BSH, 四维是BNSD或BSND. 
num_key_value_heads的值要等于key、value的N, 需要能被query的头数(num_heads)整除. 
query, key, value输入, 功能使用限制如下: 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品支持B轴小于等于65535, 支持N轴小于等于256, 支持S轴小于等于262144, 支持D轴小于等于512. 
Atlas 推理系列加速卡产品支持B轴小于等于256, 支持N轴小于等于256, 支持S轴小于等于65536, 支持D轴小于等于512. 
query、key、value输入均为int8的场景暂不支持. 
int8量化相关入参数量与输入、输出数据格式的综合限制: 
query、key、value输入为float16, 输出为int8的场景: 入参quant_scale2必填, quant_offset2可选, 不能传入dequant_scale1、quant_scale1、dequant_scale2(即为None)参数. 
pse_shift功能使用限制如下: 
pse_shift数据类型需与query数据类型保持一致. 
仅支持D轴对齐, 即D轴可以被16整除. 
page attention使用限制: 
page attention使能必要条件是block_table存在且有效, 且传入每个batch对应的actual_seq_lengths. page attention使能场景下, key、value是按照block_table中的索引在一片连续内存中排布, 支持key、value数据类型为float16、bfloat16、int8. 
page attention使能场景下, 输入kv cache排布格式为(blocknum, numKvHeads, blocksize, headDims)或(blocknum, blocksize, H), blocknum不应小于每个batch所需block个数的总和. 通常情况下, kv cache排布格式为(blocknum, numKvHeads, blocksize, headDims)时, 性能比kv cache排布格式为(blocknum, blocksize, H)时更好. 
page attention使能场景下, 支持kv cache排布格式为(blocknum, numKvHeads, blocksize, headDims), 但此时query layout仅支持BNSD. 
page attention使能场景下, 当输入kv cache排布格式为(blocknum, blocksize, H), 且H(H=numKvHeads * headDims)超过64k时, 受硬件指令约束, 会被拦截报错. 
page attention场景下, 必须传入输入actual_seq_lengths, 每个batch的actualSeqLength表示每个batch对sequence真实长度, 该值除以属性输入blocksize即表示每个batch所需block数量. 
page attention场景下, block_table必须为二维Tensor, 第一维长度需等于batch数, 第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为每个batch中最大actual_seq_lengths对应的block数量). 例如, batch数为2, 属性blocksize=128, 当每个batch的actualSeqLength为512时, 表明每个batch至少需要4个block, 因此block_table的排布可以为(2, 4). 
page attention使能场景下, block_size是用户自定义的参数, 该参数的取值会影响page attention的性能, 通常为128或256. key、value输入类型为float16、bfloat16时block_size需要16对齐; key、value输入类型为int8时block_size需要32对齐. 通常情况下, page attention可以提高吞吐量, 但会带来性能上的下降. 
quant_scale2、quant_offset2为一组参数, 其中quant_offset2可选, 传入该组参数后算子输出数据类型会推导为int8, 若不期望int8输出, 请勿传入该组参数. 
kv左padding场景使用限制: 
kvCache的搬运起点计算公式为: Smax-kv_padding_size-actual_seq_lengths. kvCache的搬运终点计算公式为: Smax-kv_padding_size. 其中kvCache的搬运起点或终点小于0时, 返回数据结果为全0. 
kv左padding场景kv_padding_size小于0时将被置为0. 
kv左padding场景使能需要同时存在kv_padding_size和actual_seq_lengths参数, 否则默认为kv右padding场景. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列加速卡产品

调用示例:
单算子调用
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
k = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
v = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)

# 调用IFA算子
out = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale)

# 执行上述代码的输出类似如下
tensor([[[ 0.3149, -0.2460,  0.7939,  ...,  0.5737, -0.4929, -0.1500]],
        [[ 0.8115,  1.3789,  0.6484,  ..., -0.9092, -0.6206, -0.7412]]],
       device='npu:0', dtype=torch.float16)


图模式调用
# 入图方式
import torch
import torch_npu
import math

import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
q = torch.randn(2, 1, 40 * 128, dtype=torch.float16).npu()
k = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
v = torch.randn(2, 2048, 40 * 128, dtype=torch.float16).npu()
atten = torch.randn(2, 1, 1, 2048).bool().npu()
scale_value = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_incre_flash_attention(q, k, v, num_heads=40, input_layout="BSH", scale_value=scale_value, atten_mask=atten)
    print("single op output with mask:", single_op, single_op.shape)
    print("graph output with mask:", graph_output, graph_output.shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
        [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
       device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
graph output with mask: tensor([[[ 0.2488, -0.6572,  1.0928,  ...,  0.1694,  0.1142, -2.2266]],
        [[-0.9595, -0.9609, -0.6602,  ...,  0.7959,  1.7920,  0.0783]]],
       device='npu:0', dtype=torch.float16) torch.Size([2, 1, 5120])
"""
)

_add_torch_npu_docstr(
    "npu_prompt_flash_attention",
    """
功能描述:
全量FA实现, 实现对应公式: 
atten_out=softmax(scale*(Q*K)+atten_mask)*V

接口原型:
torch_npu.npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, padding_mask=None, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, Tensor? deq_scale1=None, Tensor? quant_scale1=None, Tensor? deq_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, int num_heads=1, float scale_value=1.0, int pre_tokens=2147473647, int next_tokens=0, str input_layout="BSH", int num_key_value_heads=0, int[]? actual_seq_lengths_kv=None, int sparse_mode=0) -> Tensor

参数说明:
query、key、value数据排布格式支持从多种维度解读, 其中B(Batch)表示输入样本批量大小、S(Seq-Length)表示输入样本序列长度、H(Head-Size)表示隐藏层的大小、N(Head-Num)表示多头数、D(Head-Dim)表示隐藏层最小的单元尺寸, 且满足D=H/N、T表示所有Batch输入样本序列长度的累加和. 
query: Tensor类型, 公式中的输入Q, 数据类型与key的数据类型需满足数据类型推导规则, 即保持与key、value的数据类型一致. 不支持非连续的Tensor, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8. 
key: Tensor类型, 公式中的输入K, 数据类型与query的数据类型需满足数据类型推导规则, 即保持与query、value的数据类型一致. 不支持非连续的Tensor, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8. 
value: Tensor类型, 公式中的输入V, 数据类型与query的数据类型需满足数据类型推导规则, 即保持与query、key的数据类型一致. 不支持非连续的Tensor, 数据格式支持ND. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8. 
*: 代表其之前的变量是位置相关, 需要按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
pse_shift: Tensor类型, 可选参数. 不支持非连续的Tensor, 数据格式支持ND. 输入shape类型需为(B, N, Q_S, KV_S)或(1, N, Q_S, KV_S), 其中Q_S为query的shape中的S, KV_S为key和value的shape中的S. 对于pse_shift的KV_S为非32字节对齐的场景, 建议padding到32字节来提高性能, 多余部分的填充值不做要求. 如不使用该功能时可传入None. 综合约束请见约束说明. 
Atlas 推理系列加速卡产品: 暂不支持该参数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16. 当pse_shift为float16时, 要求query为float16或int8; 当pse_shift为bfloat16时, 要求query为bfloat16. 在query、key、value为float16且pse_shift存在的情况下, 默认走高精度模式. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16. 当pse_shift为float16时, 要求query为float16或int8; 当pse_shift为bfloat16时, 要求query为bfloat16. 在query、key、value为float16且pse_shift存在的情况下, 默认走高精度模式. 
padding_mask: 预留参数, 暂未使用, 默认值为None. 
atten_mask: Tensor类型, 代表下三角全为0上三角全为负无穷的倒三角mask矩阵, 数据类型支持bool、int8和uint8. 数据格式支持ND, 不支持非连续的Tensor. 如果不使用该功能可传入None. 通常建议shape输入(Q_S, KV_S)、(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S), 其中Q_S为query的shape中的S, KV_S为key和value的shape中的S, 对于attenMask的KV_S为非32字节对齐的场景, 建议padding到32字节对齐来提高性能, 多余部分填充成1. 综合约束请见7.2.1.79-约束说明. 
actual_seq_lengths: int类型数组, 代表不同Batch中query的有效seqlen, 数据类型支持int64. 如果不指定seqlen可以传入None, 表示和query的shape的s长度相同. 限制: 该入参中每个batch的有效Sequence Length应该不大于query中对应batch的seqlen. seqlen的传入长度为1时, 每个Batch使用相同seqlen; 传入长度大于等于Batch数时取seqlen的前Batch个数. 其它长度不支持. 
Atlas 推理系列加速卡产品: 暂不支持该参数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持TND格式. 当query的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
Atlas A3 训练系列产品: 支持TND格式. 当query的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
deq_scale1: Tensor类型, 表示BMM1后面的反量化因子, 支持per-tensor. 数据类型支持uint64、float32, 数据格式支持ND.  如不使用该功能时可传入None. Atlas 推理系列加速卡产品暂不支持该参数. 
quant_scale1: Tensor类型, 数据类型支持float32. 数据格式支持ND, 表示BMM2前面的量化因子, 支持per-tensor.  如不使用该功能时可传入None. Atlas 推理系列加速卡产品暂不支持该参数. 
deq_scale2: Tensor类型, 数据类型支持uint64、float32. 数据格式支持ND, 表示BMM2后面的反量化因子, 支持per-tensor.  如不使用该功能时可传入None. Atlas 推理系列加速卡产品暂不支持该参数. 
quant_scale2: Tensor类型, 数据格式支持ND, 表示输出的量化因子, 支持per-tensor、per-channel. 如不使用该功能时可传入None. 
Atlas 推理系列加速卡产品: 暂不支持该参数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、bfloat16. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32 . per-channel格式, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于N*D(建议输出layout为BSH时, quant_scale2 shape传入(1, 1, H)或(H,); 输出为BNSD时, 建议传入(1, N, 1, D)或(N, D); 输出为BSND时, 建议传入(1, 1, N, D)或(N, D)). 
Atlas A3 训练系列产品: 数据类型支持float32、bfloat16. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32 . per-channel格式, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于N*D(建议输出layout为BSH时, quant_scale2 shape传入(1, 1, H)或(H,); 输出为BNSD时, 建议传入(1, N, 1, D)或(N, D); 输出为BSND时, 建议传入(1, 1, N, D)或(N, D)). 
quant_offset2: Tensor类型, 数据格式支持ND, 表示输出的量化偏移, 支持per-tensor、per-channel. 若传入quant_offset2, 需保证其类型和shape信息与 quant_scale2一致. 如不使用该功能时可传入None. 
Atlas 推理系列加速卡产品: 暂不支持该参数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float32、bfloat16. 
Atlas A3 训练系列产品: 数据类型支持float32、bfloat16. 
num_heads: int类型数组, 代表query的head个数, 数据类型支持int64. 
scale_value: 浮点型, 公式中d开根号的倒数, 代表缩放系数, 作为计算流中Muls的scalar值, 数据类型支持float. 数据类型与query的数据类型需满足数据类型推导规则. 用户不特意指定时可传入默认值1.0. 
pre_tokens: int类型, 用于稀疏计算, 表示attention需要和前几个Token计算关联, 数据类型支持int64. 用户不特意指定时可传入默认值2147483647. Atlas 推理系列加速卡产品仅支持默认值2147483647. 
next_tokens: int类型, 用于稀疏计算, 表示attention需要和后几个Token计算关联. 数据类型支持int64. 用户不特意指定时可传入默认值0. Atlas 推理系列加速卡产品仅支持0和2147483647. 
input_layout: 字符串类型, 用于标识输入query、key、value的数据排布格式, 当前支持BSH、BSND、BNSD、BNSD、BNSD_BSND(输入为BNSD时, 输出格式为BSND). 用户不特意指定时可传入默认值"BSH". 支持TND(不支持pse、全量化、后量化). 
num_key_value_heads: int类型, 代表key、value中head个数, 用于支持GQA(Grouped-Query Attention, 分组查询注意力)场景, 数据类型支持int64. 用户不特意指定时可传入默认值0, 表示key/value和query的head个数相等. 限制: 需要满足num_heads整除num_key_value_heads, num_heads与num_key_value_heads的比值不能大于64, 且在BSND、BNSD、BNSD_BSND场景下, 需要与shape中的key/value的N轴shape值相同, 否则报错. Atlas 推理系列加速卡产品仅支持默认值0. 
actual_seq_lengths_kv: int类型数组, 代表不同batch中key/value的有效seqlenKV. 数据类型支持int64. 限制: 该入参中每个batch的有效seqlenKV应该不大于key/value中对应batch的seqlenKV. seqlenKV的传入长度为1时, 每个Batch使用相同seqlenKV; 传入长度大于等于Batch数时取seqlenKV的前Batch个数, 其它长度不支持. 
Atlas 推理系列加速卡产品: 暂不支持该参数. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持TND格式. 当key/value的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKV和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
Atlas A3 训练系列产品: 支持TND格式. 当key/value的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKV和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
sparse_mode: int类型, 表示sparse的模式, 数据类型支持int64. Atlas 推理系列加速卡产品仅支持默认值0. 
sparse_mode为0时, 代表defaultMask模式, 如果atten_mask未传入则不做mask操作, 忽略preTokens和nextTokens(内部赋值为INT_MAX); 如果传入, 则需要传入完整的atten_mask矩阵(S1 * S2), 表示pre_tokens和next_tokens之间的部分需要计算. 
sparse_mode为1时, 代表allMask. 
sparse_mode为2时, 代表leftUpCausal模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为3时, 代表rightDownCausal模式的mask, 均对应以左顶点为划分的下三角场景, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为4时, 代表band模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为5、6、7、8时, 分别代表prefix、global、dilated、block_local, 均暂不支持. 用户不特意指定时可传入默认值0.

输出说明
atten_out: Tensor类型, 计算的最终结果, shape与query保持一致. 
Atlas 推理系列加速卡产品: 数据类型支持float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
该接口与PyTorch配合使用时, 需要保证CANN相关包与PyTorch相关包的版本匹配. 
入参为空的处理: 算子内部需要判断参数query是否为空, 如果是空则直接返回. 参数query不为空Tensor, 参数key、value为空tensor(即S2为0), 则填充全零的对应shape的输出(填充attention_out). attention_out为空Tensor时, AscendCLNN框架会处理. 
query、key、value输入, 功能使用限制如下: 
轴约束
Atlas 推理系列加速卡产品: 支持B轴小于等于128. 支持N轴小于等于256. 支持S轴小于等于65535(64k). 支持D轴小于等于512. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品&Atlas A3 训练系列产品: 
{支持B轴小于等于65536(64k), D轴32byte不对齐时仅支持到128. 
支持N轴小于等于256. 
S支持小于等于20971520(20M). 长序列场景下, 如果计算量过大可能会导致PFA算子执行超时(aicore error类型报错, errorStr为timeout or trap error), 此场景下建议做S切分处理, 注: 这里计算量会受B、S、N、D等的影响, 值越大计算量越大. 典型的会超时的长序列(即B、S、N、D的乘积较大)场景包括但不限于: 
B=1, Q_N=20, Q_S=1048576, D = 256, KV_N=1, KV_S=1048576. 
B=1, Q_N=2, Q_S=10485760, D = 256, KV_N=2, KV_S=10485760. 
B=20, Q_N=1, Q_S=1048576, D = 256, KV_N=1, KV_S=1048576. 
B=1, Q_N=10, Q_S=1048576, D = 512, KV_N=1, KV_S=1048576. 
支持D轴小于等于512. input_layout为BSH或者BSND时, 要求N*D小于65535. 
TND场景下query, key, value输入的综合限制: 
B=1, Q_N=20, Q_S=1048576, D = 256, KV_N=1, KV_S=1048576. 
T小于等于65536;
N等于8/16/32/64/128, 且Q_N、K_N、V_N相等;
Q_D、K_D等于192, V_D等于128/192;
数据类型仅支持BFLOAT16;
sparse模式仅支持sparse=0且不传mask, 或sparse=3且传入mask; 
当sparse=3时, 要求每个batch单独的actualSeqLengths < actualSeqLengthsKv. }
参数sparse_mode当前仅支持值为0、1、2、3、4的场景, 取其它值时会报错. 
sparse_mode=0时, atten_mask如果为None, 则忽略入参pre_tokens、next_tokens(内部赋值为INT_MAX). 
sparse_mode=2、3、4时, atten_mask的shape需要为(S, S)或(1, S, S)或(1, 1, S, S), 其中S的值需要固定为2048, 且需要用户保证传入的atten_mask为下三角, 不传入atten_mask或者传入的shape不正确报错. 
sparse_mode=1、2、3的场景忽略入参pre_tokens、next_tokens并按照相关规则赋值. 
int8量化相关入参数量与输入、输出数据格式的综合限制: 
输入为int8, 输出为int8的场景: 入参deq_scale1、quant_scale1、deq_scale2、quant_scale2需要同时存在, quant_offset2可选, 不传时默认为0. 
输入为int8, 输出为float16的场景: 入参deq_scale1、quant_scale1、deq_scale2需要同时存在, 若存在入参quant_offset2或quant_scale2(即不为None), 则报错并返回. 
输入为float16或bfloat16, 输出为int8的场景: 入参quant_scale2需存在, quant_offset2可选, 不传时默认为0, 若存在入参deq_scale1或quant_scale1或deq_scale2(即不为None), 则报错并返回. 
入参quant_offset2和quant_scale2支持per-tensor/per-channel两种格式和float32/bfloat16两种数据类型. 若传入quant_offset2, 需保证其类型和shape信息与quant_scale2一致. 当输入为bfloat16时, 同时支持float32和bfloat16, 否则仅支持float32. per-channel格式, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于N*D. 当输出layout为BSH时, quant_scale2 shape建议传入(1, 1, H)或(H,); 当输出为BNSD时, 建议传入(1, N, 1, D)或(N, D); 当输出为BSND时, 建议传入(1, 1, N, D)或(N, D). per-tensor格式, 建议D轴对齐到32Byte. 
per-channel格式, 入参quant_scale2和quant_offset2暂不支持左padding、Ring Attention或者D非32Byte对齐的场景. 
输出为int8时, 暂不支持sparse为band且pre_tokens/next_tokens为负数. 
pse_shift功能使用限制如下: 
支持query数据类型为float16或bfloat16或int8场景下使用该功能. 
query, key, value数据类型为float16且pse_shift存在时, 强制走高精度模式, 对应的限制继承自高精度模式的限制. 
Q_S需大于等于query的S长度, KV_S需大于等于key的S长度. 
输出为int8, 入参quant_offset2传入非None和非空tensor值, 并且sparse_mode、pre_tokens和next_tokens满足以下条件, 矩阵会存在某几行不参与计算的情况, 导致计算结果误差, 该场景会拦截: 
sparseMode=0, atten_mask如果非None, 每个batch actual_seq_lengths-actual_seq_lengths_kv-pre_tokens>0或nextTokens<0时, 满足拦截条件. 
sparseMode=1或2, 不会出现满足拦截条件的情况. 
sparseMode=3, 每个batch actual_seq_lengths_kv- actual_seq_lengths<0, 满足拦截条件. 
sparseMode= 4, preTokens<0或每个batch next_tokens+actual_seq_lengths_kv-actual_seq_lengths<0时, 满足拦截条件. 
kv伪量化参数分离当前暂不支持. 
暂不支持D不对齐场景. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品
Atlas 推理系列加速卡产品

调用示例:
单算子调用
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# 调用PFA算子
out = torch_npu.npu_prompt_flash_attention(q, k, v, 
actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16)
图模式调用
# 入图方式
import torch
import torch_npu
import math

import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
    print("single op output with mask:", single_op, single_op.shape)
    print("graph output with mask:", graph_output, graph_output.shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
"""
)

_add_torch_npu_docstr(
    "npu_fused_infer_attention_score",
    """
功能描述:
算子功能: 适配增量&全量推理场景的FlashAttention算子, 既可以支持全量计算场景(PromptFlashAttention), 也可支持增量计算场景(IncreFlashAttention). 当Query矩阵的S为1, 进入IncreFlashAttention分支, 其余场景进入PromptFlashAttention分支. 
计算公式: 
attention_out = softmax(scale*(query*key)+atten_mask)*value

接口原型:
torch_npu.npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, Tensor? actual_shared_prefix_len=None,Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, bool softmax_lse_flag=False, int key_antiquant_mode=0, int value_antiquant_mode=0) -> (Tensor, Tensor)

参数说明:
query、key、value数据排布格式支持从多种维度解读, 其中B(Batch)表示输入样本批量大小、S(Seq-Length)表示输入样本序列长度、H(Head-Size)表示隐藏层的大小、N(Head-Num)表示多头数、D(Head-Dim)表示隐藏层最小的单元尺寸, 且满足D=H/N、T表示所有Batch输入样本序列长度的累加和. 
query: Tensor类型, attention结构的Query输入, 数据类型支持float16、bfloat16、int8, 不支持非连续的Tensor, 数据格式支持ND. 
key: Tensor类型, attention结构的Key输入, 不支持非连续的Tensor, 数据格式支持ND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
value: Tensor类型, attention结构的Value输入, 不支持非连续的Tensor, 数据格式支持ND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
*: 代表其之前的变量是位置相关, 需要按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
pse_shift: Tensor类型, 在attention结构内部的位置编码参数, 数据类型支持float16、bfloat16, 数据类型与query的数据类型需满足数据类型推导规则. 不支持非连续的Tensor, 数据格式支持ND. 如不使用该功能时可传入None. 
Q_S不为1, 要求在pse_shift为float16类型时, 此时的query为float16或int8类型; 而在pse_shift为bfloat16类型时, 要求此时的query为bfloat16类型. 输入shape类型需为(B, Q_N, Q_S, KV_S)或(1, Q_N, Q_S, KV_S), 其中Q_S为query的shape中的S, KV_S为key和value的shape中的S. 对于pse_shift的KV_S为非32对齐的场景, 建议padding到32字节来提高性能, 多余部分的填充值不做要求. 
Q_S为1, 要求在pse_shift为float16类型时, 此时的query为float16类型; 而在pse_shift为bfloat16类型时, 要求此时的query为bfloat16类型. 输入shape类型需为(B, Q_N, 1, KV_S)或(1, Q_N, 1, KV_S), KV_S为key和value的shape中的S. 对于pse_shift的KV_S为非32对齐的场景, 建议padding到32字节来提高性能, 多余部分的填充值不做要求. 
atten_mask: Tensor类型, 对QK的结果进行mask, 用于指示是否计算Token间的相关性, 数据类型支持bool、int8和uint8. 不支持非连续的Tensor, 数据格式支持ND. 如果不使用该功能可传入None. 
Q_S不为1时建议shape输入(Q_S, KV_S)、(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S). 
Q_S为1时建议shape输入(B, KV_S)、(B, 1, KV_S)、(B, 1, 1, KV_S). 
其中Q_S为query的shape中的S, KV_S为key和value的shape中的S, 但如果Q_S、KV_S非16或32对齐, 可以向上取到对齐的S. 综合约束请见约束说明. 
actual_seq_lengths: int类型数组, 代表不同Batch中query的有效seqlen, 数据类型支持int64. 如果不指定seqlen可以传入None, 表示和query的shape的s长度相同. 限制: 该入参中每个batch的有效seqlen应该不大于query中对应batch的seqlen, Q_S为1时该参数无效. seqlen的传入长度为1时, 每个Batch使用相同seqlen; 传入长度大于等于Batch时取seqlen的前Batch个数. 其他长度不支持. 当query的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
actual_seq_lengths_kv: int类型数组, 代表不同Batch中key/value的有效seqlenKv, 数据类型支持int64. 如果不指定None, 表示和key/value的shape的S长度相同. 不同O_S值有不同的约束, 具体参见约束说明. 
dequant_scale1: Tensor类型, 数据类型支持uint64、float32. 数据格式支持ND, 表示BMM1后面的反量化因子, 支持per-tensor. 如不使用该功能时传入None. 
quant_scale1: Tensor类型, 数据类型支持float32. 数据格式支持ND, 表示BMM2前面的量化因子, 支持per-tensor. 如不使用该功能时可传入None, 综合约束请见约束说明. 
dequant_scale2: Tensor类型, 数据类型支持uint64、float32. 数据格式支持ND, 表示BMM2后面的反量化因子, 支持per-tensor. 如不使用该功能时传入None. 
quant_scale2: Tensor类型, 数据类型支持float32、bfloat16. 数据格式支持ND, 表示输出的量化因子, 支持per-tensor、per-channel. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32 . per-channel格式, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于Q_N*D(建议输出layout为BSH时, quant_scale2shape传入(1, 1, H)或(H,); 输出为BNSD时, 建议传入(1, Q_N, 1, D)或(Q_N, D); 输出为BSND时, 建议传入(1, 1, Q_N, D)或(Q_N, D)). 如不使用该功能时可传入None, 综合约束请见约束说明. 
quant_offset2: Tensor类型, 数据类型支持float32、bfloat16. 数据格式支持ND, 表示输出的量化偏移, 支持per-tensor、per-channel. 若传入quant_offset2, 需保证其类型和shape信息与quantScale2 一致. 如不使用该功能时可传入None, 综合约束请见约束说明. 
antiquant_scale: Tensor类型, 数据类型支持float16、bfloat16. 数据格式支持ND, 表示伪量化因子, 支持per-tensor、per-channel, Q_S为1时只支持per-channel, Q_S大于等于2时只支持float16, 如不使用该功能时可传入None, 综合约束请见约束说明. 
antiquant_offset: Tensor类型, 数据类型支持float16、bfloat16. 数据格式支持ND, 表示伪量化偏移, 支持per-tensor、per-channel, Q_S为1时只支持per-channel, Q_S大于等于2时只支持float16, 如不使用该功能时可传入None, 综合约束请见约束说明. 
block_table: Tensor类型, 数据类型支持int32. 数据格式支持ND. 表示PageAttention中KV存储使用的block映射表, 如不使用该功能可传入None. 
query_padding_size: Tensor类型, 数据类型支持int64. 数据格式支持ND. 表示Query中每个batch的数据是否右对齐, 且右对齐的个数是多少. 仅支持Q_S大于1, 其余场景该参数无效. 用户不特意指定时可传入默认值None. 
kv_padding_size: Tensor类型, 数据类型支持int64. 数据格式支持ND. 表示key、value中每个batch的数据是否右对齐, 且右对齐的个数是多少. 表示key、value中每个batch的数据是否右对齐, 且右对齐的个数是多少. 用户不特意指定时可传入默认值None. 
key_antiquant_scale: Tensor类型. 数据格式支持ND, kv伪量化参数分离时表示key的反量化因子. 如不使用该功能时可传入None, 综合约束请见约束说明. 通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、float32. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、float32. 
key_antiquant_offset: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示key的反量化偏移. 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 
value_antiquant_scale: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示value的反量化因子. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、float32. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、float32. 
value_antiquant_offset: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示value的反量化偏移, 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 
key_shared_prefix: Tensor类型, attention结构中Key的系统前缀部分的参数, 数据类型支持float16、bfloat16、int8, 不支持非连续的Tensor, 数据格式支持ND. 综合约束请见约束说明. 
value_shared_prefix: Tensor类型, attention结构中Value的系统前缀部分的输入, 数据类型支持float16、bfloat16、int8, 不支持非连续的Tensor, 数据格式支持ND. 综合约束请见约束说明. 
actual_shared_prefix_len: Tensor类型, 代表key_shared_prefix/value_shared_prefix的有效Sequence Length. 数据类型支持: int64. 如果不指定seqlen可以传入None, 表示和key_shared_prefix/value_shared_prefix的s长度相同. 限制: 该入参中的有效Sequence Length应该不大于key_shared_prefix/value_shared_prefix中的Sequence Length. 
query_rope: Tensor类型, 表示MLA(Multi-head Latent Attention)结构中的query的rope信息, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND. 
key_rope: Tensor类型, 表示MLA(Multi-head Latent Attention)结构中的key的rope信息, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND. 
key_rope_antiquant_scale: Tensor类型, 预留参数, 暂未使用, 使用默认值即可. 表示MLA(Multi-head Latent Attention)结构中的key Rope对应的反量化因子, 支持per-channel, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND, D维度与key_rope的D维度保持一致. 仅支持Q_S等于1-16, 其余场景该参数无效. 
num_heads: 整型, 代表query的head个数, 数据类型支持int64, 在BNSD场景下, 需要与shape中的query的N轴shape值相同, 否则执行异常. 
scale: 浮点型, 公式中d开根号的倒数, 代表缩放系数, 作为计算流中Muls的scalar值, 数据类型支持float. 数据类型与query的数据类型需满足数据类型推导规则. 用户不特意指定时可传入默认值1.0. 
pre_tokens: 整型, 用于稀疏计算, 表示attention需要和前几个Token计算关联, 数据类型支持int64. 用户不特意指定时可传入默认值2147483647, Q_S为1时该参数无效. 
next_tokens: 整型, 用于稀疏计算, 表示attention需要和后几个Token计算关联. 数据类型支持int64. 用户不特意指定时可传入默认值2147483647, Q_S为1时该参数无效. 
input_layout: 字符串类型, 用于标识输入query、key、value的数据排布格式, 用户不特意指定时可传入默认值"BSH". 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持BSH、BSND、BNSD、BNSD_BSND、TND(不支持左padding、tensorlist、pse、page attention、prefix、伪量化、全量化、后量化, 综合约束请见约束说明). 当为TND时, 不支持图模式配置Tiling调度优化功能(tiling_schedule_optimize=True). 
Atlas A3 训练系列产品: 支持BSH、BSND、BNSD、BNSD_BSND、TND(不支持左padding、tensorlist、pse、page attention、prefix、伪量化、全量化、后量化, 综合约束请见约束说明). 当为TND时, 不支持图模式配置Tiling调度优化功能(tiling_schedule_optimize=True). 
其中BNSD_BSND含义指当输入为BNSD, 输出格式为BSND, 仅支持Q_S大于1. 
num_key_value_heads: 整型, 代表key、value中head个数, 用于支持GQA(Grouped-Query Attention, 分组查询注意力)场景, 数据类型支持int64. 用户不特意指定时可传入默认值0, 表示key/value和query的head个数相等, 需要满足num_heads整除num_key_value_heads, num_heads与num_key_value_heads的比值不能大于64. 在BSND、BNSD、BNSD_BSND(仅支持Q_S大于1)场景下, 还需要与shape中的key/value的N轴shape值相同, 否则执行异常. 
sparse_mode: 整型, 表示sparse的模式. 数据类型支持int64. Q_S为1且不带rope输入时该参数无效. 
sparse_mode为0时, 代表defaultMask模式, 如果atten_mask未传入则不做mask操作, 忽略pre_tokens和next_tokens(内部赋值为INT_MAX); 如果传入, 则需要传入完整的atten_mask矩阵(S1*S2), 表示pre_tokens和next_tokens之间的部分需要计算. 
sparse_mode为1时, 代表allMask, 必须传入完整的attenmask矩阵(S1*S2). 
sparse_mode为2时, 代表leftUpCausal模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为3时, 代表rightDownCausal模式的mask, 对应以右顶点为划分的下三角场景, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为4时, 代表band模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为5、6、7、8时, 分别代表prefix、global、dilated、block_local, 均暂不支持. 用户不特意指定时可传入默认值0. 综合约束请见约束说明. 
inner_precise: 整型, 一共4种模式: 0、1、2、3. 一共两位bit位, 第0位(bit0)表示高精度或者高性能选择, 第1位(bit1)表示是否做行无效修正. 数据类型支持int64. Q_S>1时, sparse_mode为0或1, 并传入用户自定义mask的情况下, 建议开启行无效; Q_S为1时该参数仅支持innerPrecise为0和1. 综合约束请见约束说明. 
inner_precise为0时, 代表开启高精度模式, 且不做行无效修正. 
inner_precise为1时, 代表高性能模式, 且不做行无效修正. 
inner_precise为2时, 代表开启高精度模式, 且做行无效修正. 
inner_precise为3时, 代表高性能模式, 且做行无效修正. 
bfloat16和int8不区分高精度和高性能, 行无效修正对float16、bfloat16和int8均生效. 当前0、1为保留配置值, 当计算过程中“参与计算的mask部分”存在某整行全为1的情况时, 精度可能会有损失. 此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度, 但是该配置会导致性能下降. 
block_size: 整型, PageAttention中KV存储每个block中最大的token个数, 默认为0, 数据类型支持int64. 
antiquant_mode: 整型, 表示伪量化方式, 传入0时表示为per-channel(per-channel包含per-tensor), 传入1时表示per-token. Q_S大于等于2时该参数无效, 用户不特意指定时可传入默认值0, 传入0和1之外的其他值会执行异常. 
softmax_lse_flag: 布尔型, 表示是否输出softmax_lse, 支持S轴外切(增加输出). true表示输出softmax_lse, false表示不输出; 用户不特意指定时可传入默认值false. 
key_antiquant_mode: 整型, 表示key的伪量化方式. Q_S大于等于2时仅支持传入值为1, 用户不特意指定时可传入默认值0, 取值除了key_antiquant_mode为0并且value_antiquant_mode为1的场景外, 需要与value_antiquant_mode一致. 综合约束请见约束说明. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持取值0、1、2、3、4、5. 
Atlas A3 训练系列产品: 支持取值0、1、2、3、4、5. 
key_antiquant_mode为0时, 代表per-channel模式(per-channel包含per-tensor). 
key_antiquant_mode为1时, 代表per-token模式. 
key_antiquant_mode为2时, 代表per-tensor叠加per-head模式. 
key_antiquant_mode为3时, 代表per-token叠加per-head模式. 
key_antiquant_mode为4时, 代表per-token叠加使用page attention模式管理scale/offset模式. 
key_antiquant_mode为5时, 代表per-token叠加per head并使用page attention模式管理scale/offset模式. 
value_antiquant_mode: 整型, 表示value的伪量化方式, 模式编号与key_antiquant_mode一致. Q_S大于等于2时仅支持传入值为1, 用户不特意指定时可传入默认值0, 取值除了key_antiquant_mode为0并且value_antiquant_mode为1的场景外, 需要与key_antiquant_mode一致. 综合约束请见约束说明. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持取值0、1、2、3、4、5. 
Atlas A3 训练系列产品: 支持取值0、1、2、3、4、5. 

输出说明
attention_out: Tensor类型, 公式中的输出, 数据类型支持float16、bfloat16、int8. 数据格式支持ND. 限制: 当input_layout为BNSD_BSND时, 输入query的shape是BNSD, 输出shape为BSND; 其余情况该参数的shape需要与入参query的shape保持一致. 
softmaxLse: Tensor类型, ring attention算法对query乘key的结果, 先取max得到softmax_max. query乘key的结果减去softmax_max, 再取exp, 最后取sum, 得到softmax_sum, 最后对softmax_sum取log, 再加上softmax_max得到的结果. 数据类型支持float32, softmax_lse_flag为True时, 一般情况下, 输出shape为(B, Q_N, Q_S, 1)的Tensor, 当input_layout为TND时, 输出shape为(T,Q_N,1)的Tensor; softmax_lse_flag为False时, 则输出shape为[1]的值为0的Tensor. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
该接口与PyTorch配合使用时, 需要保证CANN相关包与PyTorch相关包的版本匹配. 
入参为空的处理: 算子内部需要判断参数query是否为空, 如果是空则直接返回. 参数query不为空Tensor, 参数key、value为空tensor(即S2为0), 则填充全零的对应shape的输出(填充attention_out). attention_out为空Tensor时, 框架会处理. 
参数key、value中对应tensor的shape需要完全一致; 非连续场景下key、value的tensorlist中的batch只能为1, 个数等于query的B, N和D需要相等. 
int8量化相关入参数量与输入、输出数据格式的综合限制: 
输入为int8, 输出为int8的场景: 入参dequant_scale1、quant_scale1、dequant_scale2、quant_scale2需要同时存在, quant_offset2可选, 不传时默认为0. 
输入为int8, 输出为float16的场景: 入参dequant_scale1、quant_scale1、dequant_scale2需要同时存在, 若存在入参quant_offset2或quant_scale2(即不为None), 则报错并返回. 
输入全为float16或bfloat16, 输出为int8的场景: 入参quant_scale2需存在, quant_offset2可选, 不传时默认为0, 若存在入参dequant_scale1或quant_scale1或dequant_scale2(即不为None), 则报错并返回. 
入参quant_offset2和quant_scale2支持per-tensor或per-channel格式, 数据类型支持float32、bfloat16. 
antiquant_scale和antiquant_offset参数约束: 
支持per-channel、per-tensor和per-token三种模式: 
per-channel模式: 两个参数BNSD场景下shape为(2, KV_N, 1, D), BSND场景下shape为(2, KV_N, D), BSH场景下shape为(2, H). 参数数据类型和query数据类型相同, antiquant_mode置0, 当key、value数据类型为int8时支持. 
per-tensor模式: 两个参数的shape均为(2,), 数据类型和query数据类型相同, antiquant_mode置0, 当key、value数据类型为int8时支持. 
per-token模式: 两个参数的shape均为(2, B, KV_S), 数据类型固定为float32, antiquant_mode置1, 当key、value数据类型为int8时支持. 
算子运行在何种模式根据参数的shape进行判断, dim为1时运行per-tensor模式, 否则运行per-channel模式. 
支持对称量化和非对称量化: 
非对称量化模式下, antiquant_scale和antiquant_offset参数需同时存在. 
对称量化模式下, antiquant_offset可以为空(即None); 当antiquant_offset参数为空时, 执行对称量化, 否则执行非对称量化. 
query_rope和key_rope参数约束及支持特性: 
query_rope和key_rope要求同时配置或同时不配置, 不支持只配置其中一个.
query_rope的数据类型、数据格式与query一致.
key_rope的数据类型、数据格式与key一致.
sparse: Q_S等于1时支持sparse=0且不传mask或sparse=4且传入mask, Q_S大于1时支持sparse=3或sparse=4且传入mask;
sparse不为4时:
query_rope配置时要求query的S为1-16、N为32、64、128, D为512, shape中B、N、S与query一致, D为64. 
key_rope配置时要求key的N为1, D为512, key_rope的shape中B、N、S与key一致, D为64. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持key、value的input_layout格式为ND或NZ. 当input_layout为NZ时, 输入参数key和value的格式为[blockNum, KV_N, D/16, blockSize, 16]. 
Atlas A3 训练系列产品: 支持key、value的input_layout格式为ND或NZ. 当input_layout为NZ时, 输入参数key和value的格式为[blockNum, KV_N, D/16, blockSize, 16]. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: input_layout形状支持BSH、BSND、BNSD, 当数据格式为NZ时input_layout不支持BNSD. 
Atlas A3 训练系列产品: input_layout形状支持BSH、BSND、BNSD, 当数据格式为NZ时input_layout不支持BNSD. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 该场景下, 必须开启PageAttention, 此时block_size支持16、128, 其中数据格式为NZ时block_size不支持配置16. 
Atlas A3 训练系列产品: 该场景下, 必须开启PageAttention, 此时block_size支持16、128, 其中数据格式为NZ时block_size不支持配置16. 
sparse为4时:
query_rope配置时要求query的每batch的S不大于key的每batch的S、N为128, D为512, query_rope的shape中B、N、S与query一致, D为64. 
key_rope配置时要求key的S不大于131088, N为1, D为512, key_rope的shape中B、N、S与key一致, D为64.
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 仅支持key、value的input_layout格式为ND.
Atlas A3 训练系列产品: 仅支持key、value的input_layout格式为ND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: input_layout形状仅支持BSND. 
Atlas A3 训练系列产品: input_layout形状仅支持BSND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持开启PageAttention, 此时block_size支持64、128. 
Atlas A3 训练系列产品: 支持开启PageAttention, 此时block_size支持64、128. 
TND场景下query、key、value输入的综合限制: 
T小于等于65536;
N等于8/16/32/64/128, 且Q_N、K_N、V_N相等;
Q_D、K_D等于192, V_D等于128/192;
数据类型仅支持BFLOAT16; 
sparse模式仅支持sparse=0且不传mask, 或sparse=3且传入mask; 
当sparse=3时, 要求每个batch单独的actual_seq_lengths < actual_seq_lengths_kv. 
sparse模式支持sparse\_mode=4且传入mask；当sparse\_mode=4时，要求preTokens >= -actual\_seq\_qlen、nextTokens >= -actual\_seq\_kvlen、preTokens + nextTokens >= 0；
当Q_S大于1时: 
query、key、value输入, 功能使用限制如下: 
支持B轴小于等于65536, D轴32byte不对齐时仅支持到128. 
支持N轴小于等于256, 支持D轴小于等于512; input_layout为BSH或者BSND时, 要求N*D小于65535. 
S支持小于等于20971520(20M). 部分长序列场景下, 如果计算量过大可能会导致PFA算子执行超时(aicore error类型报错, errorStr为timeout or trap error), 此场景下建议做S切分处理(注: 这里计算量会受B、S、N、D等的影响, 值越大计算量越大), 典型的会超时的长序列(即B、S、N、D的乘积较大)场景包括但不限于: 
B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152. 
B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520. 
B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152. 
B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152. 
query、key、value输入类型包含int8时, D轴需要32对齐; 输入类型全为float16、bfloat16时, D轴需16对齐. 
actual_seq_lengths_kv: 该参数传入时应为非负数, 在input_layout不同时, 其含义与拦截条件不同: 一般情况下, 该入参为可选入参, 该入参中每个Batch的有效seqlenKv应该不大于key/value中对应Batch的seqlenKv. 当本参数的传入长度为1时, 每个Batch使用相同seqlenKv; 传入长度大于等于Batch时取seqlenKv的前Batch个数. 其他长度不支持. 当key/value的input_layout为TND时, 该入参必须传入, 且该入参元素的数量等于Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKv和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
参数sparse_mode当前仅支持值为0、1、2、3、4的场景, 取其它值时会报错. 
sparse_mode=0时, atten_mask如果为None, 或者在左padding场景传入atten_mask, 则忽略入参pre_tokens、next_tokens(内部赋值为INT_MAX). 
sparse_mode=2、3、4时, atten_mask的shape需要为(S, S)或(1, S, S)或(1, 1, S, S), 其中S的值需要固定为2048, 且需要用户保证传入的atten_mask为下三角, 不传入atten_mask或者传入的shape不正确报错. 
sparse_mode=1、2、3的场景忽略入参pre_tokens、next_tokens并按照相关规则赋值. 
kvCache反量化的合成参数场景仅支持int8反量化到float16. 入参key、value的data range与入参antiquant_scale的data range乘积范围在(-1, 1)内, 高性能模式可以保证精度, 否则需要开启高精度模式来保证精度. 
page attention场景:
page attention的使能必要条件是block_table存在且有效, 同时key、value是按照block_table中的索引在一片连续内存中排布, 支持key、value数据类型为float16、bfloat16、int8. 在该场景下key、value的input_layout参数无效. block_table中填充的是blockid, 当前不会对blockid的合法性进行校验, 需用户自行保证. 
block_size是用户自定义的参数, 该参数的取值会影响page attention的性能, 在使能page attention场景下, block_size最小为128, 最大为512, 且要求是128的倍数. 通常情况下, page attention可以提高吞吐量, 但会带来性能上的下降. 
page attention场景下, 当输入kv cache排布格式为(blocknum, blocksize, H), 且KV_N*D超过65535时, 受硬件指令约束, 会被拦截报错. 可通过使能GQA(减小KV_N)或调整kv cache排布格式为(blocknum, KV_N, blocksize, D)解决. 当query的input_layout为BNSD、TND时, kv cache排布支持(blocknum, blocksize, H)和(blocknum, KV_N, blocksize, D)两种格式, 当query的input_layout为BSH、BSND时, kv cache排布只支持(blocknum, blocksize, H)一种格式. blocknum不能小于根据actual_seq_lengths_kv和blockSize计算的每个batch的block数量之和. 且key和value的shape需保证一致. 
page attention不支持伪量化场景, 不支持tensorlist场景, 不支持左padding场景. 
page attention场景下, 必须传入actual_seq_lengths_kv. 
page attention场景下, block_table必须为二维, 第一维长度需等于B, 第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为不同batch中最大actual_seq_lengths_kv对应的block数量). 
page atte两种格式和float32/bfloat1ntion场景下, 不支持输入query为int8的场景. 
page attention使能场景下, 以下场景输入需满足KV_S>=maxBlockNumPerSeq*blockSize: 
传入attenMask时, 如mask shape为 (B, 1, Q_S, KV_S). 
传入pseShift时, 如pseShift shape为(B, Q_N, Q_S, KV_S). 
query左padding场景: 
query左padding场景query的搬运起点计算公式为: Q_S-query_padding_size-actual_seq_lengths. query的搬运终点计算公式为: Q_S-query_padding_size. 其中query的搬运起点不能小于0, 终点不能大于Q_S, 否则结果将不符合预期. 
query左padding场景kv_padding_size小于0时将被置为0. 
query左padding场景需要与actual_seq_lengths参数一起使能, 否则默认为query右padding场景. 
query左padding场景不支持PageAttention, 不能与block_table参数一起使能. 
kv左padding场景: 
kv左padding场景key和value的搬运起点计算公式为: KV_S-kv_padding_size-actual_seq_lengths_kv. key和value的搬运终点计算公式为: KV_S-kv_padding_size. 其中key和value的搬运起点不能小于0, 终点不能大于KV_S, 否则结果将不符合预期. 
kv左padding场景kv_padding_size小于0时将被置为0. 
kv左padding场景需要与actual_seq_lengths_kv参数一起使能, 否则默认为kv右padding场景. 
kv左padding场景不支持PageAttention, 不能与block_table参数一起使能. 
入参quant_scale2和quant_offset2支持per-tensor、per-channel量化, 支持float32、bfloat16类型. 若传入quant_offset2, 需保证其类型和shape信息与quant_scale2一致. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32. per-channel场景下, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于N*D. 当输出layout为BSH时, quant_scale2 shape建议传入(1, 1, H)或(H,); 当输出layout为BNSD时, 建议传入(1, Q_N, 1, D)或(Q_N, D); 当输出为BSND时, 建议传入(1, 1, Q_N, D)或(Q_N, D). 
输出为int8, quant_scale2和quant_offset2为per-channel时, 暂不支持左padding、Ring Attention或者D非32Byte对齐的场景. 
输出为int8时, 暂不支持sparse为band且preTokens/nextTokens为负数. 
pse_shift功能使用限制如下: 
支持query数据类型为float16或bfloat16或int8场景下使用该功能. 
query、key、value数据类型为float16且pse_shift存在时, 强制走高精度模式, 对应的限制继承自高精度模式的限制. 
Q_S需大于等于query的S长度, KV_S需大于等于key的S长度. prefix场景KV_S需大于等于actual_shared_prefix_len与key的S长度之和. 
输出为int8, 入参quant_offset2传入非None和非空tensor值, 并且sparse_mode、pre_tokens和next_tokens满足以下条件, 矩阵会存在某几行不参与计算的情况, 导致计算结果误差, 该场景会拦截: 
sparse_mode=0, atten_mask如果非None, 每个batch actual_seq_lengths-actual_seq_lengths_kv-pre_tokens>0或next_tokens<0时, 满足拦截条件. 
sparse_mode=1或 2, 不会出现满足拦截条件的情况. 
sparse_mode=3, 每个batch actual_seq_lengths_kv-actual_seq_lengths<0, 满足拦截条件. 
sparse_mode=4, pre_tokens<0或每个batch next_tokens+actual_seq_lengths_kv-actual_seq_lengths<0时, 满足拦截条件. 
prefix相关参数约束: 
key_shared_prefix和value_shared_prefix要么都为空, 要么都不为空. 
key_shared_prefix和value_shared_prefix都不为空时, key_shared_prefix、value_shared_prefix、key、value的维度相同、dtype保持一致. 
key_shared_prefix和value_shared_prefix都不为空时, key_shared_prefix的shape第一维batch必须为1, layout为BNSD和BSND情况下N、D轴要与key一致、BSH情况下H要与key一致, value_shared_prefix同理. key_shared_prefix和value_shared_prefix的S应相等. 
当actual_shared_prefix_len存在时, actual_shared_prefix_len的shape需要为[1], 值不能大于key_shared_prefix和value_shared_prefix的S. 
公共前缀的S加上key或value的S的结果, 要满足原先key或value的S的限制. 
prefix不支持PageAttention场景、不支持左padding场景、不支持tensorlist场景. 
prefix场景不支持query、key、value数据类型同时为int8. 
prefix场景, sparse为0或1时, 如果传入attenmask, 则S2需大于等于actual_shared_prefix_len与key的S长度之和. 
prefix场景, 不支持输入qkv全部为int8的场景. 
kv伪量化参数分离: 
key_antiquant_mode和value_antiquant_mode需要保持一致. 
key_antiquant_scale和value_antiquant_scale要么都为空, 要么都不为空; key_antiquant_offset和value_antiquant_offset要么都为空, 要么都不为空. 
key_antiquant_scale和value_antiquant_scale都不为空时, 其shape需要保持一致; key_antiquant_offset和value_antiquant_offset都不为空时, 其shape需要保持一致. 
仅支持per-token模式, 且该模式下要求两个参数的shape均为(B, KV_S), 数据类型固定为float32. 
当伪量化参数和KV分离量化参数同时传入时, 以KV分离量化参数为准. 
key_antiquant_scale与value_antiquant_scale非空场景, 要求query的s小于等于16. 
key_antiquant_scale与value_antiquant_scale非空场景, 要求query的dtype为bfloat16, key、value的dtype为int8, 输出的dtype为bfloat16. 
key_antiquant_scale与value_antiquant_scale非空场景, 不支持tensorlist、左padding、page attention、prefix特性. 
当Q_S等于1时: 
query、key、value输入, 功能使用限制如下: 
支持B轴小于等于65536, 支持N轴小于等于256, 支持S轴小于等于262144, 支持D轴小于等于512. 
query、key、value输入类型均为int8的场景暂不支持. 
在int4(int32)伪量化场景下, PyTorch入图调用仅支持KV int4拼接成int32输入(建议通过dynamicQuant生成int4格式的数据, 因为dynamicQuant就是一个int32包括8个int4). 
在int4(int32)伪量化场景下, 若KV int4拼接成int32输入, 那么KV的N、D或者H是实际值的八分之一(prefix同理). 并且, int4伪量化仅支持D 64对齐(int32支持D 8对齐). 
actual_seq_lengths_kv: 该参数应为非负数, 在input_layout不同时, 其含义与拦截条件不同: 一般情况下, 该入参为可选入参, 该入参中每个Batch的有效Sequence Length应该不大于key/value中对应Batch的seqlenKv. 当本参数的传入长度为1时, 每个Batch使用相同seqlenKv; 传入长度大于等于Batch时取seqlenKv的前Batch个数. 其他长度不支持. 当input_layout为TND时, 该入参必须传入, 在非PA场景下, 第b个值表示前b个Batch的S轴累加长度, 其值应递增(大于等于前一个值)排列, 且该入参元素的数量代表总Batch数, 在PA场景下, 其长度等于key/value的Batch值, 代表每个Batch的实际长度, 值不大于KV_S. 
page attention场景: 
使能必要条件是block_table存在且有效, 同时key、value是按照block_table中的索引在一片连续内存中排布, 在该场景下key、value的input_layout参数无效. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持key、value数据类型为float16、bfloat16、int8. 
Atlas A3 训练系列产品: 支持key、value数据类型为float16、bfloat16、int8. 
该场景下, block_size是用户自定义的参数, 该参数的取值会影响page attention的性能. block_size需要传入非0值，且最大不超过512, key、value输入类型为float16、bfloat16时需要16对齐, key、value输入类型为int8时需要32对齐, 推荐使用128. 通常情况下, page attention可以提高吞吐量, 但会带来性能上的下降. 
参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 不支持Q为bfloat16、float16、key、value为int4(int32)的场景. 
Atlas A3 训练系列产品: 不支持Q为bfloat16、float16、key、value为int4(int32)的场景. 
page attention场景下, blockTable必须为二维, 第一维长度需等于B, 第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为不同batch中最大actual_seq_lengths_kv对应的block数量). 
page attention场景下, 当query的input_layout为BNSD、TND时, kv cache排布支持(blocknum, blocksize, H)和(blocknum, KV_N, blocksize, D)两种格式, 当query的input_layout为BSH、BSND时, kv cache排布只支持(blocknum, blocksize, H)一种格式. blocknum不能小于根据actual_seq_lengths_kv和blockSize计算的每个batch的block数量之和. 且key和value的shape需保证一致. 
page attention场景下, kv cache排布为(blocknum, KV_N, blocksize, D)时性能通常优于kv cache排布为(blocknum, blocksize, H)时的性能, 建议优先选择(blocknum, KV_N, blocksize, D)格式. 
page attention场景下, 当输入kv cache排布格式为(blocknum, blocksize, H), 且 numKvHeads * headDim 超过64k时, 受硬件指令约束, 会被拦截报错. 可通过使能GQA(减小 numKvHeads)或调整kv cache排布格式为(blocknum, numKvHeads, blocksize, D)解决. 
page attention不支持tensorlist场景, 不支持左padding场景. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 不支持Q为BF16/FP16且KV为INT4(INT32)的场景. 
Atlas A3 训练系列产品: 不支持Q为BF16/FP16且KV为INT4(INT32)的场景. 
page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围. 
age attention场景下，使能`atten_mask`，当`sparse_mode`不为2、3、4时，传入的`atten_mask`的最后一维需要大于等于`block_table`的第二维 * `block_size`.
page attention场景下，使能`pse_shift`，传入的`pse_shift`的最后一维需要大于等于`block_table`的第二维 * `block_size`.
page attention场景下，以下场景输入S需要大于等于`block_table`的第二维 * `block_size`：
使能伪量化per-token模式：输入参数`antiqunant_scale`和`antiquant_offset`的shape均为\(2, B, S\).
使能per-token叠加per-head模式：两个参数的shape均为\(B, N, S\)，数据类型固定为`float32`。支持`key`、`value`数据类型为`int8`、`int4`\(`int32`\)。
kv左padding场景: 
kvCache的搬运起点计算公式为: Smax-kv_padding_size-actual_seq_lengths. kvCache的搬运终点计算公式为: Smax-kv_padding_size. 其中kvCache的搬运起点或终点小于0时, 返回数据结果为全0. 
kv_padding_size小于0时将被置为0. 
使能需要同时存在actual_seq_lengths参数, 否则默认为kv右padding场景. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: kv左padding场景不支持Q为bfloat16/float16、KV为int4(int32)的场景. 
Atlas A3 训练系列产品: kv左padding场景不支持Q为bfloat16/float16、KV为int4(int32)的场景. 
kv伪量化参数分离: 
除了key_antiquant_mode为0并且value_antiquant_mode为1的场景外, key_antiquant_mode和value_antiquant_mode取值需要保持一致. 
key_antiquant_scale和value_antiquant_scale要么都为空, 要么都不为空; key_antiquant_offset和value_antiquant_offset要么都为空, 要么都不为空. 
key_antiquant_scale和value_antiquant_scale都不为空时, 除了key_antiquant_mode为0并且value_antiquant_mode为1的场景外, 其shape需要保持一致; key_antiquant_offset和value_antiquant_offset都不为空时, 除了key_antiquant_mode为0并且value_antiquant_mode为1的场景外, 其shape需要保持一致. 
int4(int32)伪量化场景不支持后量化. 
管理scale/offset的量化模式如下: 
注意scale、offset两个参数指key_antiquant_scale、key_antiquant_scale、value_antiquant_offset、value_antiquant_offset. 
场景下scale和offset条件
per-channel模式: 两个参数shape支持(1, KV_N, 1, D), (1, KV_N, D), (1, H), 数据类型和query数据类型相同. 
per-tensor模式: 两个参数的shape均为(1,), 数据类型和query数据类型相同. 
per-token模式: 两个参数的shape均为(1, B, KV_S), 数据类型固定为float32. 
per-tensor叠加per-head模式: 两个参数的shape均为(KV_N,), 数据类型和query数据类型相同. 
per-token叠加per-head模式: 两个参数的shape均为(B, KV_N, KV_S), 数据类型固定为float32. 
per-token叠加使用page attention模式: 两个参数的shape均为(blocknum, blocksize), 数据类型固定为float32. 
per-token叠加per head并使用page attention模式: 两个参数的shape均为(blocknum, KV_N, blocksize), 数据类型固定为float32. 
key支持per-channel叠加value支持per-token模式: 对于key支持per-channel, 两个参数的shape可支持(1, KV_N, 1, D)、(1, KV_N, D)、(1, H), 且参数数据类型和query数据类型相同. 对于value支持per-token, 两个参数的shape均为(1, B, KV_S)并且数据类型固定为float32. 
场景下key和value条件
per-channel模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int4(int32)或int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int4(int32)或int8时支持. 
per-tensor模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int8时支持. 
per-token模式: key、value数据类型为int4(int32)或int8时支持. 
per-tensor叠加per-head模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int8时支持. 
per-token叠加per-head模式: key、value数据类型为int4(int32)或int8时支持. 
per-token叠加使用page attention模式: key、value数据类型为int8时支持. 
per-token叠加per head并使用page attention模式: key、value数据类型为int8时支持. 
key支持per-channel叠加value支持per-token模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int4(int32)或int8时支持; 当key和value的数据类型为int8时, 仅支持query和输出的dtype为float16. Atlas A3 训练系列产品: 当key、value数据类型为int4(int32)或int8时支持; 当key和value的数据类型为int8时, 仅支持query和输出的dtype为float16. 
支持的产品: Atlas A2 训练系列产品/Atlas 800I A2 推理产品. Atlas A3 训练系列产品
pse_shift功能使用限制如下: 
pse_shift数据类型需与query数据类型保持一致. 仅支持D轴对齐, 即D轴可以被16整除. 

支持的PyTorch版本
PyTorch 2.1
PyTorch 2.3
PyTorch 2.4

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import math
# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# 调用FIA算子
out, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, 
actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
num_heads = 8, input_layout = "BNSD", scale = scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出out类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ..
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16)
图模式调用
# 入图方式
import torch
import torch_npu
import math
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
"""
)

_add_torch_npu_docstr(
    "_npu_fused_infer_attention_score_get_max_workspace",
    """
功能描述:
算子功能：用于npu_fused_infer_attention_score算子aclgraph tilling下沉场景，获取最大workspace size并创建一个此size大小的tensor。

接口原型:
torch_npu._npu_fused_infer_attention_score_get_max_workspace(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False) -> Tensor

参数说明:
输入与npu_fused_infer_attention_score一致
输出类型为Tensor, 由aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize返回最大的Size，返回创建的workspace tensor。

约束说明:
当Q_S等于1时：请参考Incre_Flash_Attention限制
当Q_S大于1时：请参考Prompt_Flash_Attention限制

支持的芯片型号:
Atlas A2 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)

# 调用FIA算子
out = torch_npu._npu_fused_infer_attention_score_get_max_workspace(q, k, v, num_heads = 8, input_layout = "BNSD", scale = scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出类似如下
tensor([0., 0., ..., 0., 0., 0.],
        device='npu:0', dtype=torch.float16)

# 入图方式
暂不支持入图
"""
)

_add_torch_npu_docstr(
    "npu_fused_infer_attention_score.out",
    """
功能描述:
算子功能：npu_fused_infer_attention_score.out算子实现，可用于aclgraph tilling下沉场景（需传入workspace tensor），输入参数相比npu_fused_infer_attention_score增加workspace、attention_out、softmax_lse。
计算公式：atten_out = softmax(scale*(query*key)+atten_mask)*value

接口原型:
torch_npu.npu_fused_infer_attention_score.out(Tensor query, Tensor key, Tensor value, *, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_lengths=None, SymInt[]? actual_seq_lengths_kv=None, Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, Tensor? quant_scale2=None, Tensor? quant_offset2=None, Tensor? antiquant_scale=None, Tensor? antiquant_offset=None, Tensor? key_antiquant_scale=None, Tensor? key_antiquant_offset=None, Tensor? value_antiquant_scale=None, Tensor? value_antiquant_offset=None, Tensor? block_table=None, Tensor? query_padding_size=None, Tensor? kv_padding_size=None, Tensor? key_shared_prefix=None, Tensor? value_shared_prefix=None, SymInt[]? actual_shared_prefix_len=None, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? key_rope_antiquant_scale=None, int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, int block_size=0, int antiquant_mode=0, int key_antiquant_mode=0, int value_antiquant_mode=0, bool softmax_lse_flag=False, Tensor? workspace=None, Tensor(a!) attention_out, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))

参数说明:
在torch_npu.npu_fused_infer_attention_score的基础上增加下面三个参数：
workspace(可选): 一维Device侧的Input Tensor，数据类型与Query一致；
attention_out（aclTensor*，计算输出）: 计算的最终结果Attention output tensor, shape与Query一致；
softmax_lse（aclTensor*，计算输出）: 也是一个输出结果，当前预留，暂不支持；

约束说明:
当Q_S等于1时：请参考Incre_Flash_Attention限制
当Q_S大于1时：请参考Prompt_Flash_Attention限制

支持的芯片型号:
Atlas A2 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
workspace = torch.randn(2000000, dtype=torch.float16).npu()
output = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
softmax_lse = torch.randn(1, dtype=torch.float16).npu()
scale = 1/math.sqrt(128.0)

# 调用FIA算子
out = torch_npu.npu_fused_infer_attention_score.out(q, k, v, workspace=workspace, out=[output, softmax_lse], num_heads = 8, input_layout = "BNSD", scale = scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出output类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.float16)

# 入图方式
暂不支持入图
"""
)

_add_torch_npu_docstr(
    "npu_fused_infer_attention_score_v2",
    """
功能描述:
算子功能: 适配增量&全量推理场景的FlashAttention算子, 既可以支持全量计算场景(PromptFlashAttention), 也可支持增量计算场景(IncreFlashAttention). 当Query矩阵的S为1, 进入IncreFlashAttention分支, 其余场景进入PromptFlashAttention分支. 
计算公式: 
attention_out = softmax(softmax_scale*(query*key)+atten_mask)*value

接口原型:
torch_npu.npu_fused_infer_attention_score_v2(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_qlen=None, SymInt[]? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor)

参数说明:
query、key、value数据排布格式支持从多种维度解读, 其中B(Batch)表示输入样本批量大小、S(Seq-Length)表示输入样本序列长度、H(Head-Size)表示隐藏层的大小、N(Head-Num)表示多头数、D(Head-Dim)表示隐藏层最小的单元尺寸, 且满足D=H/N、T表示所有Batch输入样本序列长度的累加和. 
query: Tensor类型, attention结构的Query输入, 数据类型支持float16、bfloat16、int8, 不支持非连续的Tensor, 数据格式支持ND. 
key: Tensor类型, attention结构的Key输入, 不支持非连续的Tensor, 数据格式支持ND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
value: Tensor类型, attention结构的Value输入, 不支持非连续的Tensor, 数据格式支持ND. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、int8、int4(int32). 
*: 代表其之前的变量是位置相关, 需要按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
query_rope: Tensor类型, 表示MLA(Multi-head Latent Attention)结构中的query的rope信息, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND. 
key_rope: Tensor类型, 表示MLA(Multi-head Latent Attention)结构中的key的rope信息, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND. 
pse_shift: Tensor类型, 在attention结构内部的位置编码参数, 数据类型支持float16、bfloat16, 数据类型与query的数据类型需满足数据类型推导规则. 不支持非连续的Tensor, 数据格式支持ND. 如不使用该功能时可传入None. 
Q_S不为1, 要求在pse_shift为float16类型时, 此时的query为float16或int8类型; 而在pse_shift为bfloat16类型时, 要求此时的query为bfloat16类型. 输入shape类型需为(B, Q_N, Q_S, KV_S)或(1, Q_N, Q_S, KV_S), 其中Q_S为query的shape中的S, KV_S为key和value的shape中的S. 对于pse_shift的KV_S为非32对齐的场景, 建议padding到32字节来提高性能, 多余部分的填充值不做要求. 
Q_S为1, 要求在pse_shift为float16类型时, 此时的query为float16类型; 而在pse_shift为bfloat16类型时, 要求此时的query为bfloat16类型. 输入shape类型需为(B, Q_N, 1, KV_S)或(1, Q_N, 1, KV_S), KV_S为key和value的shape中的S. 对于pse_shift的KV_S为非32对齐的场景, 建议padding到32字节来提高性能, 多余部分的填充值不做要求. 
atten_mask: Tensor类型, 对QK的结果进行mask, 用于指示是否计算Token间的相关性, 数据类型支持bool、int8和uint8. 不支持非连续的Tensor, 数据格式支持ND. 如果不使用该功能可传入None. 
Q_S不为1时建议shape输入(Q_S, KV_S)、(B, Q_S, KV_S)、(1, Q_S, KV_S)、(B, 1, Q_S, KV_S)、(1, 1, Q_S, KV_S). 
Q_S为1时建议shape输入(B, KV_S)、(B, 1, KV_S)、(B, 1, 1, KV_S). 
其中Q_S为query的shape中的S, KV_S为key和value的shape中的S, 但如果Q_S、KV_S非16或32对齐, 可以向上取到对齐的S. 综合约束请见约束说明. 
actual_seq_qlen: int类型数组, 代表不同Batch中query的有效seqlen, 数据类型支持int64. 如果不指定seqlen可以传入None, 表示和query的shape的s长度相同. 限制: 该入参中每个batch的有效seqlen应该不大于query中对应batch的seqlen, Q_S为1时该参数无效. seqlen的传入长度为1时, 每个Batch使用相同seqlen; 传入长度大于等于Batch时取seqlen的前Batch个数. 其他长度不支持. 当query的input_layout为TND时, 该入参必须传入, 且以该入参元素的数量作为Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlen和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
actual_seq_kvlen: int类型数组, 代表不同Batch中key/value的有效seqlenKv, 数据类型支持int64. 如果不指定None, 表示和key/value的shape的S长度相同. 不同O_S值有不同的约束, 具体参见约束说明. 
block_table: Tensor类型, 数据类型支持int32. 数据格式支持ND. 表示PageAttention中KV存储使用的block映射表, 如不使用该功能可传入None. 
dequant_scale_query: Tensor类型. 数据格式支持ND, query的反量化参数. 仅支持per-token叠加per-head. 如不使用该功能时可传入None, 综合约束请见约束说明.
dequant_scale_key: Tensor类型. 数据格式支持ND, kv伪量化参数分离时表示key的反量化因子. 如不使用该功能时可传入None, 综合约束请见约束说明. 通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、float32. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、float32. 
dequant_offset_key: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示key的反量化偏移. 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 
dequant_scale_value: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示value的反量化因子. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 通常支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理scale、per-token叠加per head并使用page attention模式管理scale. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、bfloat16、float32. 
Atlas A3 训练系列产品: 数据类型支持float16、bfloat16、float32. 
dequant_offset_value: Tensor类型, 数据类型支持float16、bfloat16、float32. 数据格式支持ND, kv伪量化参数分离时表示value的反量化偏移, 支持per-channel、per-tensor、per-token、per-tensor叠加per-head、per-token叠加per-head、per-token叠加使用page attention模式管理offset、per-token叠加per head并使用page attention模式管理offset. Q_S大于等于2时仅支持per-token模式, 如不使用该功能时可传入None, 综合约束请见约束说明. 
dequant_scale_key_rope: Tensor类型, 预留参数, 暂未使用, 使用默认值即可. 表示MLA(Multi-head Latent Attention)结构中的key Rope对应的反量化因子, 支持per-channel, 数据类型支持float16、bfloat16, 不支持非连续的Tensor, 数据格式支持ND, D维度与key_rope的D维度保持一致. 仅支持Q_S等于1-16, 其余场景该参数无效. 
quant_scale_out: Tensor类型, 数据类型支持float32、bfloat16. 数据格式支持ND, 表示输出的量化因子, 支持per-tensor、per-channel. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32 . per-channel格式, 当输出layout为BSH时, 要求quant_scale2所有维度的乘积等于H; 其他layout要求乘积等于Q_N*D(建议输出layout为BSH时, quant_scale2shape传入(1, 1, H)或(H,); 输出为BNSD时, 建议传入(1, Q_N, 1, D)或(Q_N, D); 输出为BSND时, 建议传入(1, 1, Q_N, D)或(Q_N, D)). 如不使用该功能时可传入None, 综合约束请见约束说明. 
quant_offset_out: Tensor类型, 数据类型支持float32、bfloat16. 数据格式支持ND, 表示输出的量化偏移, 支持per-tensor、per-channel. 若传入quant_offset_out, 需保证其类型和shape信息与quant_scale_out 一致. 如不使用该功能时可传入None, 综合约束请见约束说明. 
learnable_sink: Tensor类型, 数据类型支持bfloat16, 数据格式支持ND, shape输入为(Q_N,), 通过可学习的"Sink Token"起到吸收Attention Score的作用, 如果不使用该功能可传入None, 综合约束请见约束说明.
num_query_heads: 整型, 代表query的head个数, 数据类型支持int64, 在BNSD场景下, 需要与shape中的query的N轴shape值相同, 否则执行异常. 
num_key_value_heads: 整型, 代表key、value中head个数, 用于支持GQA(Grouped-Query Attention, 分组查询注意力)场景, 数据类型支持int64. 用户不特意指定时可传入默认值0, 表示key/value和query的head个数相等, 需要满足num_query_heads整除num_key_value_heads, num_query_heads与num_key_value_heads的比值不能大于64. 在BSND、BNSD、BNSD_BSND(仅支持Q_S大于1)场景下, 还需要与shape中的key/value的N轴shape值相同, 否则执行异常. 
softmax_scale: 浮点型, 公式中d开根号的倒数, 代表缩放系数, 作为计算流中Muls的scalar值, 数据类型支持float. 数据类型与query的数据类型需满足数据类型推导规则. 用户不特意指定时可传入默认值1.0. 
pre_tokens: 整型, 用于稀疏计算, 表示attention需要和前几个Token计算关联, 数据类型支持int64. 用户不特意指定时可传入默认值2147483647, Q_S为1时该参数无效. 
next_tokens: 整型, 用于稀疏计算, 表示attention需要和后几个Token计算关联. 数据类型支持int64. 用户不特意指定时可传入默认值2147483647, Q_S为1时该参数无效. 
input_layout: 字符串类型, 用于标识输入query、key、value的数据排布格式, 用户不特意指定时可传入默认值"BSH". 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持BSH、BSND、BNSD、BNSD_BSND、TND(不支持tensorlist、pse、page attention、伪量化、全量化、后量化, 综合约束请见约束说明). 当为TND时, 不支持图模式配置Tiling调度优化功能(tiling_schedule_optimize=True). 
Atlas A3 训练系列产品: 支持BSH、BSND、BNSD、BNSD_BSND、TND(不支持tensorlist、pse、page attention、伪量化、全量化、后量化, 综合约束请见约束说明). 当为TND时, 不支持图模式配置Tiling调度优化功能(tiling_schedule_optimize=True). 
其中BNSD_BSND含义指当输入为BNSD, 输出格式为BSND, 仅支持Q_S大于1. 
sparse_mode: 整型, 表示sparse的模式. 数据类型支持int64. Q_S为1且不带rope输入时该参数无效. 
sparse_mode为0时, 代表defaultMask模式, 如果atten_mask未传入则不做mask操作, 忽略pre_tokens和next_tokens(内部赋值为INT_MAX); 如果传入, 则需要传入完整的atten_mask矩阵(S1*S2), 表示pre_tokens和next_tokens之间的部分需要计算. 
sparse_mode为1时, 代表allMask, 必须传入完整的attenmask矩阵(S1*S2). 
sparse_mode为2时, 代表leftUpCausal模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为3时, 代表rightDownCausal模式的mask, 对应以右顶点为划分的下三角场景, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为4时, 代表band模式的mask, 需要传入优化后的atten_mask矩阵(2048*2048). 
sparse_mode为5、6、7、8时, 分别代表prefix、global、dilated、block_local, 均暂不支持. 用户不特意指定时可传入默认值0. 综合约束请见约束说明. 
block_size: 整型, PageAttention中KV存储每个block中最大的token个数, 默认为0, 数据类型支持int64. 
query_quant_mode: 整型, 表示query的伪量化方式。仅支持传入3，代表模式3：代表per-token叠加per-head模式.
key_quant_mode: 整型, 表示key的伪量化方式. Q_S大于等于2时仅支持传入值为1, 用户不特意指定时可传入默认值0, 取值除了key_quant_mode为0并且value_quant_mode为1的场景外, 需要与value_quant_mode一致. 综合约束请见约束说明. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持取值0、1、2、3、4、5. 
Atlas A3 训练系列产品: 支持取值0、1、2、3、4、5. 
key_quant_mode为0时, 代表per-channel模式(per-channel包含per-tensor). 
key_quant_mode为1时, 代表per-token模式. 
key_quant_mode为2时, 代表per-tensor叠加per-head模式. 
key_quant_mode为3时, 代表per-token叠加per-head模式. 
key_quant_mode为4时, 代表per-token叠加使用page attention模式管理scale/offset模式. 
key_quant_mode为5时, 代表per-token叠加per head并使用page attention模式管理scale/offset模式. 
value_quant_mode: 整型, 表示value的伪量化方式, 模式编号与key_quant_mode一致. Q_S大于等于2时仅支持传入值为1, 用户不特意指定时可传入默认值0, 取值除了key_quant_mode为0并且value_quant_mode为1的场景外, 需要与key_quant_mode一致. 综合约束请见约束说明. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持取值0、1、2、3、4、5. 
Atlas A3 训练系列产品: 支持取值0、1、2、3、4、5. 
inner_precise: 整型, 一共4种模式: 0、1、2、3. 一共两位bit位, 第0位(bit0)表示高精度或者高性能选择, 第1位(bit1)表示是否做行无效修正. 数据类型支持int64. Q_S>1时, sparse_mode为0或1, 并传入用户自定义mask的情况下, 建议开启行无效; Q_S为1时该参数仅支持innerPrecise为0和1. 综合约束请见约束说明. 
inner_precise为0时, 代表开启高精度模式, 且不做行无效修正. 
inner_precise为1时, 代表高性能模式, 且不做行无效修正. 
inner_precise为2时, 代表开启高精度模式, 且做行无效修正. 
inner_precise为3时, 代表高性能模式, 且做行无效修正. 
bfloat16和int8不区分高精度和高性能, 行无效修正对float16、bfloat16和int8均生效. 当前0、1为保留配置值, 当计算过程中“参与计算的mask部分”存在某整行全为1的情况时, 精度可能会有损失. 此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度, 但是该配置会导致性能下降.
return_softmax_lse: 布尔型, 表示是否输出softmax_lse, 支持S轴外切(增加输出). true表示输出softmax_lse, false表示不输出; 用户不特意指定时可传入默认值false. 
query_dtype: 整型, 表示query的数据类型，预留参数，暂未使用，使用默认值即可.
key_dtype: 整型, 表示key的数据类型，预留参数，暂未使用，使用默认值即可.
value_dtype: 整型, 表示value的数据类型，预留参数，暂未使用，使用默认值即可.
query_rope_dtyp: 整型, 表示query_repo的数据类型，预留参数，暂未使用，使用默认值即可.
key_rope_dtype: 整型, 表示key_rope的数据类型，预留参数，暂未使用，使用默认值即可.
key_shared_prefix_dtype: 整型, 表示key_shared_prefix的数据类型，预留参数，暂未使用，使用默认值即可.
value_shared_prefix_dtype: 整型, 表示value_shared_prefix的数据类型，预留参数，暂未使用，使用默认值即可.
dequant_scale_query_dtype: 整型, 表示dequant_scale_query的数据类型，预留参数，暂未使用，使用默认值即可.
dequant_scale_key_dtype: 整型, 表示dequant_scale_key的数据类型，预留参数，暂未使用，使用默认值即可.
dequant_scale_value_dtype: 整型, 表示dequant_scale_value的数据类型，预留参数，暂未使用，使用默认值即可.
dequant_scale_key_rope_dtype: 整型, 表示dequant_scale_key_rope的数据类型，预留参数，暂未使用，使用默认值即可.

输出说明
attention_out: Tensor类型, 公式中的输出, 数据类型支持float16、bfloat16、int8. 数据格式支持ND. 限制：该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致. 
softmaxLse: Tensor类型, ring attention算法对query乘key的结果, 先取max得到softmax_max. query乘key的结果减去softmax_max, 再取exp, 最后取sum, 得到softmax_sum, 最后对softmax_sum取log, 再加上softmax_max得到的结果. 数据类型支持float32, return_softmax_lse为True时, 一般情况下, 输出shape为(B, Q_N, Q_S, 1)的Tensor, 当input_layout为TND时, 输出shape为(T,Q_N,1)的Tensor; return_softmax_lse为False时, 则输出shape为[1]的值为0的Tensor. 

约束说明:
该接口支持推理场景下使用. 
该接口支持图模式. 
该接口与PyTorch配合使用时, 需要保证CANN相关包与PyTorch相关包的版本匹配. 
入参为空的处理: 算子内部需要判断参数query是否为空, 如果是空则直接返回. 参数query不为空Tensor, 参数key、value为空tensor(即S2为0), 则填充全零的对应shape的输出(填充attention_out). attention_out为空Tensor时, 框架会处理. 
参数key、value中对应tensor的shape需要完全一致; 非连续场景下key、value的tensorlist中的batch只能为1, 个数等于query的B, N和D需要相等. 
int8量化相关入参数量与输出数据格式的综合限制: 
输出为int8的场景: 入参quant_scale_out需要存在, quant_offset_out可选, 不传时默认为0. 
输出为float16的场景: 若存在入参quant_offset_out或quant_scale_out(即不为None), 则报错并返回. 
入参quant_offset_out和quant_scale_out支持per-tensor或per-channel格式, 数据类型支持float32、bfloat16. 
query_rope和key_rope输入时即为MLA场景，参数约束如下：
query_rope的数据类型、数据格式与query一致。
key_rope的数据类型、数据格式与key一致。
query_rope和key_rope要求同时配置或同时不配置，不支持只配置其中一个。
当query_rope和key_rope非空时，支持如下特性：
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/Atlas A3 推理系列产品：query的d只支持512、128；
当query的d等于512时：
sparse：Q_S等于1时只支持sparse=0且不传mask，Q_S大于1时只支持sparse=3且传入mask；
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/Atlas A3 推理系列产品约束如下：
query_rope配置时要求query的s为1-16、n为1/2/4/8/16/32/64/128，query_rope的shape中d为64，其余维度与query一致；
key_rope配置时要求key的n为1，d为512，keyRope的shape中d为64，其余维度与key一致；
支持key、value、keyRope的input_layout格式为ND或NZ。当input_layout为NZ时，数据类型为float16或bfloat16时，输入参数key和value的格式为[blockNum, KV_N, D/16, blockSize, 16]，数据类型为int8时，输入参数key和value的格式为[blockNum, KV_N, D/32, blockSize, 32]；
input_layout形状支持BSH、BSND、BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、TND_NTD，当数据格式为NZ时input_layout不支持BNSD、BNSD_NBSD。
该场景下，必须开启PageAttention，此时block_size支持16、128，其中数据格式为NZ时block_size不支持配置16。
不支持开启softmax_lse、左padding、tensorlist、pse、prefix、伪量化、后量化、空Tensor。
支持全量化场景，即输入query/key/value全为int8，query_rope和key_rope为bfloat16，输出为bfloat16的场景：
入参dequant_scale_query、dequant_scale_key、dequant_scale_value需要同时存在，且其数据类型仅支持FP32。
不支持传入quant_scale_out、quant_offset_out、dequant_offset_key、dequant_offset_value（即不为nullptr），否则报错并返回。
query_quant_mode仅支持per-token叠加per-head模式，key_quant_mode和value_quant_mode仅支持per-tensor模式。
支持key、value、keyRope的input_layout格式为NZ。
当query的d等于128时：
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/Atlas A3 推理系列产品约束如下：
inputLayout：TND、NTD_TND。  
query_rope配置时要求query_rope的shape中d为64，其余维度与query一致。  
keyRope配置时要求keyRope的shape中d为64，其余维度与key一致。  
不支持左padding、tensorlist、pse、page attention、prefix、伪量化、全量化、后量化、空Tensor。
其余约束同TND、NTD_TND场景下的综合限制保持一致。
TND、TND_NTD、NTD_TND场景下query、key、value输入的综合限制：
T小于等于1M;
sparse模式仅支持sparse=0且不传mask，或sparse=3且传入mask；
actual_seq_qlen和actual_seq_kvlen必须传入，且以该入参元素的数量作为Batch值。该入参中每个元素的值表示当前Batch与之前所有Batch的Sequence Length和，因此后一个元素的值必须大于等于前一个元素的值；
当query的d等于512时：
支持TND、TND_NTD;
必须开启page attention，此时actual_seq_kvlen长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV_S；
支持query每个batch的s为1-16；
要求query的n为1/2/4/8/16/32/64/128，key、value的n为1；
要求query_rope和keyRope不等于空，query_rope和keyRope的d为64；
不支持开启softmax_lse、左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor。

当query的d不等于512时：
当query_rope和key_rope为空时：TND场景，要求Q_D、K_D、V_D等于128，或者Q_D、K_D等于192，V_D等于128/192；NTD_TND场景，要求Q_D、K_D等于128/192，V_D等于128。当query_rope和key_rope不为空时，要求Q_D、K_D、V_D等于128；
Q_N、K_N、V_N：需要满足K_N、V_N相等，Q_N整除K_N，Q_N与K_N的比值不能大于64；
支持TND、NTD_TND；
数据类型仅支持BFLOAT16；
当sparse=3时，要求每个batch单独的actual_seq_qlen<actual_seq_kvlen；
sparse模式支持sparse\_mode=4且传入mask；当sparse\_mode=4时，要求preTokens >= -actual\_seq\_qlen、nextTokens >= -actual\_seq\_kvlen、preTokens + nextTokens >= 0；
不支持左padding、tensorlist、pse、prefix、伪量化、全量化、后量化、空Tensor；
不支持图模式配置Tiling调度优化（tiling_schedule_optimize=True）、reduce-overhead执行模式（config.mode="reduce-overhead"）。
actual_seq_qlen和actual_seq_kvlen的元素个数不大于4096。

GQA伪量化场景下KV为NZ格式时的参数约束如下：
支持per-channel和per-token模式，query数据类型固定为bfloat16，key&value固定为int8；query&key&value的D仅支持128；query Sequence Length仅支持1-16；
input_layout仅支持BSH、BSND、BNSD；
仅支持page_attention场景，blockSize仅支持128或512；
key&value仅支持NZ输入，输入格式为[blockNum, KV_N, D/32, blockSize, 32]；
dequant_scale_key和dequant_scale_value的dtype：per-channel模式下，仅支持bfloat16类型；per-token模式下，仅支持float32类型；
dequant_scale_key和dequant_scale_value的shape：per-channel模式下，当layout为BSH时，必须传入[H]；layout为BNSD时，必须传入[KV_N,1,D]；输出为BSND时，必须传入[KV_N, D]；per-token模式下，必须传入[B,KV_S]，S需要大于等于blockTable的第二维*blockSize；
仅支持KV分离；
仅支持高性能模式；
当MTP等于0时，支持sparse_mode=0且不传mask；当MTP大于0、小于16时，支持sparse_mode=3且传入优化后的atten_mask矩阵，atten_mask矩阵shape必须传入（2048*2048）；
不支持配置dequant_offset_key和dequant_offset_value;
不支持配置query_rope和key_rope；
不支持左padding、tensorlist、pse、prefix、后量化；
num_query_heads与num_key_value_heads支持组合有(10,1)、(64,8)、(80,8)、(128,16)。

learnable_sink的参数约束如下：
仅支持TND、NTD\_TND；
仅支持value的d小于等于128；

当Q_S大于1时: 
query、key、value输入, 功能使用限制如下: 
支持B轴小于等于65536, D轴32byte不对齐时仅支持到128. 
支持N轴小于等于256, 支持D轴小于等于512; input_layout为BSH或者BSND时, 要求N*D小于65535. 
S支持小于等于20971520(20M). 部分长序列场景下, 如果计算量过大可能会导致PFA算子执行超时(aicore error类型报错, errorStr为timeout or trap error), 此场景下建议做S切分处理(注: 这里计算量会受B、S、N、D等的影响, 值越大计算量越大), 典型的会超时的长序列(即B、S、N、D的乘积较大)场景包括但不限于: 
B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152. 
B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520. 
B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152. 
B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152. 
query、key、value输入类型包含int8时, D轴需要32对齐; 输入类型全为float16、bfloat16时, D轴需16对齐. 
actual_seq_kvlen: 该参数传入时应为非负数, 在input_layout不同时, 其含义与拦截条件不同: 一般情况下, 该入参为可选入参, 该入参中每个Batch的有效seqlenKv应该不大于key/value中对应Batch的seqlenKv. 当本参数的传入长度为1时, 每个Batch使用相同seqlenKv; 传入长度大于等于Batch时取seqlenKv的前Batch个数. 其他长度不支持. 当key/value的input_layout为TND时, 该入参必须传入, 且该入参元素的数量等于Batch值. 该入参中每个元素的值表示当前Batch与之前所有Batch的seqlenKv和, 因此后一个元素的值必须大于等于前一个元素的值, 且不能出现负值. 
参数sparse_mode当前仅支持值为0、1、2、3、4的场景, 取其它值时会报错. 
sparse_mode=0时, atten_mask如果为None, 则忽略入参pre_tokens、next_tokens(内部赋值为INT_MAX). 
sparse_mode=2、3、4时, atten_mask的shape需要为(S, S)或(1, S, S)或(1, 1, S, S), 其中S的值需要固定为2048, 且需要用户保证传入的atten_mask为下三角, 不传入atten_mask或者传入的shape不正确报错. 
sparse_mode=1、2、3的场景忽略入参pre_tokens、next_tokens并按照相关规则赋值. 
page attention场景:
page attention的使能必要条件是block_table存在且有效, 同时key、value是按照block_table中的索引在一片连续内存中排布, 支持key、value数据类型为float16、bfloat16、int8. 在该场景下key、value的input_layout参数无效. block_table中填充的是blockid, 当前不会对blockid的合法性进行校验, 需用户自行保证. 
block_size是用户自定义的参数, 该参数的取值会影响page attention的性能, 在使能page attention场景下, block_size最小为128, 最大为512, 且要求是128的倍数. 通常情况下, page attention可以提高吞吐量, 但会带来性能上的下降. 
page attention场景下, 当输入kv cache排布格式为(blocknum, blocksize, H), 且KV_N*D超过65535时, 受硬件指令约束, 会被拦截报错. 可通过使能GQA(减小KV_N)或调整kv cache排布格式为(blocknum, KV_N, blocksize, D)解决. 当query的input_layout为BNSD、TND时, kv cache排布支持(blocknum, blocksize, H)和(blocknum, KV_N, blocksize, D)两种格式, 当query的input_layout为BSH、BSND时, kv cache排布只支持(blocknum, blocksize, H)一种格式. blocknum不能小于根据actual_seq_kvlen和blockSize计算的每个batch的block数量之和. 且key和value的shape需保证一致. 
page attention不支持伪量化场景, 不支持tensorlist场景. 
page attention场景下, 必须传入actual_seq_kvlen. 
page attention场景下, block_table必须为二维, 第一维长度需等于B, 第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为不同batch中最大actual_seq_kvlen对应的block数量). 
page attention场景下，支持两种格式和float32/bfloat16，不支持输入query为int8的场景。
page attention使能场景下, 以下场景输入需满足KV_S>=maxBlockNumPerSeq*blockSize: 
传入attenMask时, 如mask shape为 (B, 1, Q_S, KV_S). 
传入pseShift时, 如pseShift shape为(B, Q_N, Q_S, KV_S). 
入参quant_scale_out和quant_offset_out支持per-tensor、per-channel量化, 支持float32、bfloat16类型. 若传入quant_offset_out, 需保证其类型和shape信息与quant_scale_out一致. 当输入为bfloat16时, 同时支持float32和bfloat16 , 否则仅支持float32. per-channel场景下, 当输出layout为BSH时, 要求quant_scale_out所有维度的乘积等于H; 其他layout要求乘积等于Q_N*D. 当输出layout为BSH时, quant_scale_out shape建议传入(1, 1, H)或(H,); 当输出layout为BNSD时, 建议传入(1, Q_N, 1, D)或(Q_N, D); 当输出为BSND时, 建议传入(1, 1, Q_N, D)或(Q_N, D). 
输出为int8, quant_scale_out和quant_offset_out为per-channel时, 暂不支持Ring Attention或者D非32Byte对齐的场景. 
输出为int8时, 暂不支持sparse为band且preTokens/nextTokens为负数. 
pse_shift功能使用限制如下: 
支持query数据类型为float16或bfloat16或int8场景下使用该功能. 
query、key、value数据类型为float16且pse_shift存在时, 强制走高精度模式, 对应的限制继承自高精度模式的限制. 
Q_S需大于等于query的S长度, KV_S需大于等于key的S长度.
输出为int8, 入参quant_offset_out传入非None和非空tensor值, 并且sparse_mode、pre_tokens和next_tokens满足以下条件, 矩阵会存在某几行不参与计算的情况, 导致计算结果误差, 该场景会拦截: 
sparse_mode=0, atten_mask如果非None, 每个batch actual_seq_qlen-actual_seq_kvlen-pre_tokens>0或next_tokens<0时, 满足拦截条件. 
sparse_mode=1或 2, 不会出现满足拦截条件的情况. 
sparse_mode=3, 每个batch actual_seq_kvlen-actual_seq_qlen<0, 满足拦截条件. 
sparse_mode=4, pre_tokens<0或每个batch next_tokens+actual_seq_kvlen-actual_seq_qlen<0时, 满足拦截条件. 
kv伪量化参数分离: 
key_quant_mode和value_quant_mode需要保持一致. 
dequant_scale_key和dequant_scale_value要么都为空, 要么都不为空; dequant_offset_key和dequant_offset_value要么都为空, 要么都不为空. 
dequant_scale_key和dequant_scale_value都不为空时, 其shape需要保持一致; dequant_offset_key和dequant_offset_value都不为空时, 其shape需要保持一致. 
仅支持per-token和per-channel模式，per-token模式下要求两个参数的shape均为(B, KV_S)，数据类型固定为float32；per-channel模式下要求两个参数的shape为（KV_N, D），(KV_N, D)，(H)，数据类型固定为bfloat16,H为KV_N*D.
当伪量化参数和KV分离量化参数同时传入时, 以KV分离量化参数为准. 
dequant_scale_key与dequant_scale_value非空场景, 要求query的s小于等于16. 
dequant_scale_key与dequant_scale_value非空场景, 要求query的dtype为bfloat16, key、value的dtype为int8, 输出的dtype为bfloat16. 
dequant_scale_key与dequant_scale_value非空场景, 不支持tensorlist、page attention特性. 
当Q_S等于1时: 
query、key、value输入, 功能使用限制如下: 
支持B轴小于等于65536, 支持N轴小于等于256, 支持S轴小于等于262144, 支持D轴小于等于512. 
query、key、value输入类型均为int8的场景暂不支持. 
在int4(int32)伪量化场景下, PyTorch入图调用仅支持KV int4拼接成int32输入(建议通过dynamicQuant生成int4格式的数据, 因为dynamicQuant就是一个int32包括8个int4). 
在int4(int32)伪量化场景下, 若KV int4拼接成int32输入, 那么KV的N、D或者H是实际值的八分之一. 并且, int4伪量化仅支持D 64对齐(int32支持D 8对齐). 
actual_seq_kvlen: 该参数应为非负数, 在input_layout不同时, 其含义与拦截条件不同: 一般情况下, 该入参为可选入参, 该入参中每个Batch的有效Sequence Length应该不大于key/value中对应Batch的seqlenKv. 当本参数的传入长度为1时, 每个Batch使用相同seqlenKv; 传入长度大于等于Batch时取seqlenKv的前Batch个数. 其他长度不支持. 当input_layout为TND时, 该入参必须传入, 在非PA场景下, 第b个值表示前b个Batch的S轴累加长度, 其值应递增(大于等于前一个值)排列, 且该入参元素的数量代表总Batch数, 在PA场景下, 其长度等于key/value的Batch值, 代表每个Batch的实际长度, 值不大于KV_S. 
page attention场景: 
使能必要条件是block_table存在且有效, 同时key、value是按照block_table中的索引在一片连续内存中排布, 在该场景下key、value的input_layout参数无效. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 支持key、value数据类型为float16、bfloat16、int8. 
Atlas A3 训练系列产品: 支持key、value数据类型为float16、bfloat16、int8. 
该场景下, block_size是用户自定义的参数, 该参数的取值会影响page attention的性能. key、value输入类型为float16、bfloat16时需要16对齐, key、value输入类型为int8时需要32对齐, 推荐使用128. 通常情况下, page attention可以提高吞吐量, 但会带来性能上的下降. 
参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 不支持Q为bfloat16、float16、key、value为int4(int32)的场景. 
Atlas A3 训练系列产品: 不支持Q为bfloat16、float16、key、value为int4(int32)的场景. 
page attention场景下, blockTable必须为二维, 第一维长度需等于B, 第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为不同batch中最大actual_seq_kvlen对应的block数量). 
page attention场景下, 当query的input_layout为BNSD、TND时, kv cache排布支持(blocknum, blocksize, H)和(blocknum, KV_N, blocksize, D)两种格式, 当query的input_layout为BSH、BSND时, kv cache排布只支持(blocknum, blocksize, H)一种格式. blocknum不能小于根据actual_seq_kvlen和blockSize计算的每个batch的block数量之和. 且key和value的shape需保证一致. 
page attention场景下, kv cache排布为(blocknum, KV_N, blocksize, D)时性能通常优于kv cache排布为(blocknum, blocksize, H)时的性能, 建议优先选择(blocknum, KV_N, blocksize, D)格式. 
page attention使能场景下, 当输入kv cache排布格式为(blocknum, blocksize, H), 且 numKvHeads * headDim 超过64k时, 受硬件指令约束, 会被拦截报错. 可通过使能GQA(减小 numKvHeads)或调整kv cache排布格式为(blocknum, numKvHeads, blocksize, D)解决. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 不支持Q为BF16/FP16且KV为INT4(INT32)的场景. 
Atlas A3 训练系列产品: 不支持Q为BF16/FP16且KV为INT4(INT32)的场景. 
page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围. 
kv伪量化参数分离: 
除了key_quant_mode为0并且value_quant_mode为1的场景外, key_quant_mode和value_quant_mode取值需要保持一致. 
dequant_scale_key和dequant_scale_value要么都为空, 要么都不为空; dequant_offset_key和dequant_offset_value要么都为空, 要么都不为空. 
dequant_scale_key和dequant_scale_value都不为空时, 除了key_quant_mode为0并且value_quant_mode为1的场景外, 其shape需要保持一致; dequant_offset_key和dequant_offset_value都不为空时, 除了key_quant_mode为0并且value_quant_mode为1的场景外, 其shape需要保持一致. 
int4(int32)伪量化场景不支持后量化. 
管理scale/offset的量化模式如下: 
注意scale、offset两个参数指dequant_scale_key、dequant_scale_key、dequant_offset_value、dequant_offset_value. 
场景下scale和offset条件
per-channel模式: 两个参数shape支持(1, KV_N, 1, D), (1, KV_N, D), (1, H), 数据类型和query数据类型相同. 
per-tensor模式: 两个参数的shape均为(1,), 数据类型和query数据类型相同. 
per-token模式: 两个参数的shape均为(1, B, KV_S), 数据类型固定为float32. 
per-tensor叠加per-head模式: 两个参数的shape均为(KV_N,), 数据类型和query数据类型相同. 
per-token叠加per-head模式: 两个参数的shape均为(B, KV_N, KV_S), 数据类型固定为float32. 
per-token叠加使用page attention模式: 两个参数的shape均为(blocknum, blocksize), 数据类型固定为float32. 
per-token叠加per head并使用page attention模式: 两个参数的shape均为(blocknum, KV_N, blocksize), 数据类型固定为float32. 
key支持per-channel叠加value支持per-token模式: 对于key支持per-channel, 两个参数的shape可支持(1, KV_N, 1, D)、(1, KV_N, D)、(1, H), 且参数数据类型和query数据类型相同. 对于value支持per-token, 两个参数的shape均为(1, B, KV_S)并且数据类型固定为float32. 
场景下key和value条件
per-channel模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int4(int32)或int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int4(int32)或int8时支持. 
per-tensor模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int8时支持. 
per-token模式: key、value数据类型为int4(int32)或int8时支持. 
per-tensor叠加per-head模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int8时支持. Atlas A3 训练系列产品: 当key、value数据类型为int8时支持. 
per-token叠加per-head模式: key、value数据类型为int4(int32)或int8时支持. 
per-token叠加使用page attention模式: key、value数据类型为int8时支持. 
per-token叠加per head并使用page attention模式: key、value数据类型为int8时支持. 
key支持per-channel叠加value支持per-token模式: Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 当key、value数据类型为int4(int32)或int8时支持; 当key和value的数据类型为int8时, 仅支持query和输出的dtype为float16. Atlas A3 训练系列产品: 当key、value数据类型为int4(int32)或int8时支持; 当key和value的数据类型为int8时, 仅支持query和输出的dtype为float16. 
支持的产品: Atlas A2 训练系列产品/Atlas 800I A2 推理产品. Atlas A3 训练系列产品
pse_shift功能使用限制如下: 
pse_shift数据类型需与query数据类型保持一致. 仅支持D轴对齐, 即D轴可以被16整除. 

支持的PyTorch版本
PyTorch 2.1
PyTorch 2.3
PyTorch 2.4

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
import math
# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# 调用FIA算子
out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, 
actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv,
num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出out类似如下
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ..
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16)
图模式调用
# 入图方式
import torch
import torch_npu
import math
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
"""
)

_add_torch_npu_docstr(
    "_npu_fused_infer_attention_score_v2_get_max_workspace",
    """
功能描述:
算子功能：用于npu_fused_infer_attention_score_v2算子aclgraph tilling下沉场景，获取最大workspace size并创建一个此size大小的tensor。

接口原型:
torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_qlen=None, SymInt[]? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None) -> Tensor

参数说明:
输入与npu_fused_infer_attention_score_v2一致
输出类型为Tensor, 由aclnnFusedInferAttentionScoreV4GetMaxWorkspaceSize返回最大的Size，返回创建的workspace tensor。

约束说明:
当Q_S等于1时：请参考Incre_Flash_Attention限制
当Q_S大于1时：请参考Prompt_Flash_Attention限制

支持的芯片型号:
Atlas A2 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)

# 调用FIA算子
out = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出类似如下
tensor([0., 0., ..., 0., 0., 0.],
        device='npu:0', dtype=torch.float16)

# 入图方式
暂不支持入图
"""
)

_add_torch_npu_docstr(
    "npu_fused_infer_attention_score_v2.out",
    """
功能描述:
算子功能：npu_fused_infer_attention_score_v2.out算子实现，可用于aclgraph tilling下沉场景（需传入workspace tensor），输入参数相比npu_fused_infer_attention_score_v2增加workspace、attention_out、softmax_lse。
计算公式：atten_out = softmax(softmax_scale*(query*key)+atten_mask)*value

接口原型:
torch_npu.npu_fused_infer_attention_score_v2.out(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, SymInt[]? actual_seq_qlen=None, SymInt[]? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None, Tensor? workspace=None, Tensor(a!) attention_out, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))

参数说明:
在torch_npu.npu_fused_infer_attention_score_v2的基础上增加下面三个参数：
workspace(可选): 一维Device侧的Input Tensor，数据类型与Query一致；
attention_out（aclTensor*，计算输出）: 计算的最终结果Attention output tensor, shape与Query一致；
softmax_lse（aclTensor*，计算输出）: 也是一个输出结果，当前预留，暂不支持；

约束说明:
当Q_S等于1时：请参考Incre_Flash_Attention限制
当Q_S大于1时：请参考Prompt_Flash_Attention限制

支持的芯片型号:
Atlas A2 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
workspace = torch.randn(2000000, dtype=torch.float16).npu()
output = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
softmax_lse = torch.randn(1, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)

# 调用FIA算子
out = torch_npu.npu_fused_infer_attention_score_v2.out(q, k, v, workspace=workspace, out=[output, softmax_lse], num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

# 执行上述代码的输出output类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.float16)

# 入图方式
暂不支持入图
"""
)

_add_torch_npu_docstr(
    "npu_mla_prolog",
    """
功能描述:
推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为四路，首先对输入x乘以WeightDq进行下采样和RmsNorm后分成两路，第一路乘以WeightUq和WeightUk经过两次上采样后得到query；第二路乘以WeightQr后经过旋转位置编码（ROPE)得到query_rope；第三路是输入x乘以WeightDkv进行下采样和RmsNorm后传入Cache中得到kvCache；第四路是输入x乘以Wkr后经过旋转位置编码后传入另一个Cache中得到krCache。

接口原型:
torch_npu.npu_mla_prolog(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor)

参数说明:
-   token_x（Tensor）：必选参数，对应公式中x。shape支持2维和3维，格式为(T, He)和(B, S, He)，dtype支持bfloat16，数据格式支持ND。
-   weight_dq（Tensor）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为(He, Hcq)，dtype支持bfloat16，数据格式支持FRACTAL_NZ（可通过torch_npu.npu_format_cast将ND格式转为FRACTAL_NZ格式）。
-   weight_uq_qr（Tensor）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为(Hcq, N*(D+Dr))，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
    -   当weight_uq_qr为int8类型时，weight_uq_qr是一个per-tensor的量化后的输入，表示当前为部分量化场景。

        此时若kv_cache、kr_cache为bfloat16类型，对应kv_cache_out、kr_cache_out为非量化输出，此时dequant_scale_w_uq_qr字段必须传入，smooth_scales_cq字段可选传入。

        此时若kv_cache、kr_cache为int8类型，对应kv_cache_out、kr_cache_out为量化输出，此时dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr字段必须传入，smooth_scales_cq字段可选传入。

    -   当weight_uq_qr为bfloat16类型时，表示当前为非量化场景。

        此时dequant_scale_w_uq_qr、quant_scale_ckv、quant_scale_ckr、smooth_scales_cq字段不能传入（即为none）。

-   weight_uk（Tensor）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为(N, D, Hckv)，dtype支持bfloat16，数据格式支持ND。
-   weight_dkv_kr（Tensor）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为(He, Hckv+Dr)，dtype支持bfloat16，数据格式支持FRACTAL_NZ。
-   rmsnorm_gamma_cq（Tensor）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hcq,)，dtype支持bfloat16，数据格式支持ND。
-   rmsnorm_gamma_ckv（Tensor）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hckv,)，dtype支持bfloat16，数据格式支持ND。
-   rope_sin（Tensor）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   rope_cos（Tensor）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   cache_index（Tensor）：必选参数，表示用于存储kv_cache和kr_cache的索引。shape支持1维和2维，格式为(T,)和(B, S)，dtype支持int64，数据格式支持ND。
    - cache_index的取值范围为[0,BlockNum*BlockSize)，当前不会对cache_index传入值的合法性进行校验，需用户自行保证。
-   kv_cache（Tensor）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache（Tensor）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。
-   dequant_scale_x（Tensor）：预留参数，暂未使用，使用默认值即可。
-   dequant_scale_w_dq（Tensor）：预留参数，暂未使用，使用默认值即可。
-   dequant_scale_w_uq_qr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数维per-channel。shape支持2维，格式为(1, N*(D+Dr))，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dkv_kr（Tensor）：预留参数，暂未使用，使用默认值即可。
-   quant_scale_ckv（Tensor）：可选参数，用于对输出到kv_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Hckv)，dtype支持float，数据格式支持ND。
-   quant_scale_ckr（Tensor）：可选参数，用于对输出到kr_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Dr)，dtype支持float，数据格式支持ND。
-   smooth_scales_cq（Tensor）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   rmsnorm_epsilon_cq（float）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   rmsnorm_epsilon_ckv（float）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   cache_mode（str）：可选参数，表示kvCache的模式，支持"PA_BSND"、"PA_NZ"，其用户不特意指定时可传入默认值“PA_BSND”。


输出说明：
-   query（Tensor）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为(T, N, Hckv)和(B, S, N, Hckv)，dtype支持bfloat16，数据格式支持ND。
-   query_rope（Tensor）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为(T, N, Dr)和(B, S, N, Dr)，dtype支持bfloat16，数据格式支持ND。
-   kv_cache_out（Tensor）：表示Key输出到kv_cache中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache_out（Tensor）：表示Key的位置编码输出到kr_cache中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。

约束说明:
-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   接口参数中shape格式字段含义：
    -   B：Batch表示输入样本批量大小，取值范围为0~65536。
    -   S：Seq-Length表示输入样本序列长度，取值范围为0~16。
    -   He：Head-Size表示隐藏层的大小，取值为7168。

    -   Hcq：q低秩矩阵维度，取值为1536。
    -   N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    -   Hckv：kv低秩矩阵维度，取值为512。
    -   D：qk不含位置编码维度，取值为128。
    -   Dr：qk位置编码维度，取值为64。
    -   Nkv：kv的head数，取值为1。
    -   BlockNum：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
    -   BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    -   T：BS合轴后的大小，取值范围：0~1048576。注：若采用BS合轴，此时token_x、rope_sin、rope_cos均为2维，cache_index为1维，query、query_rope为3维。
-   shape约束
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        -   如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        -   如果Skv取值为0，则query、query_rope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。


支持的芯片型号:
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例:
# 单算子调用方式
import math
import torch
import torch_npu

# 生成随机数据, 并发送到npu
B = 8
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 1024
S = 1
Nkv = 1
BlockSize = 128
BlockNum = math.ceil(B * Skv / BlockSize)
T = 8

token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)

w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)

w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)

rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()

cache_index = torch.rand(B, S).to(torch.int64).npu()

kv_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Hckv, dtype=torch.bfloat16).npu()
kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)

kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)

rmsnorm_epsilon_cq = 1.0e-5
rmsnorm_epsilon_ckv = 1.0e-5
cache_mode = "PA_BSND"

query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = torch_npu.npu_mla_prolog(
    token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast,
    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
    cache_index, kv_cache, kr_cache,
    rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
    rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode
)


# 执行上述代码的输出类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

# 入图方式

import torch
import torch_npu
import math

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "2"
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
B = 8
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 1024
S = 2
Nkv = 1
BlockNum = 32
BlockSize = 128
token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
cache_index = torch.rand(B, S).to(torch.int64).npu()
kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
rmsnorm_epsilon_cq = 1.0e-5
rmsnorm_epsilon_ckv = 1.0e-5
cache_mode = "PA_BSND"

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return torch_npu.npu_mla_prolog(
            token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache)

def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()

if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
single op output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

graph output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],        device='npu:0', dtype=torch.bfloat16)
"""
)

_add_torch_npu_docstr(
    "npu_mla_prolog_v2",
    """
功能描述:
推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分五路，首先对输入x乘以WeightDq进行下采样和RmsNorm后分成两路，第一路乘以WeightUq和WeightUk经过两次上采样后得到query；第二路乘以WeightQr后经过旋转位置编码（ROPE)得到query_rope；第三路是输入x乘以WeightDkv进行下采样和RmsNorm后传入Cache中得到kvCache；第四路是输入x乘以Wkr后经过旋转位置编码后传入另一个Cache中得到krCache；第五路是输出query经过DynamicQuant后得到的量化参数。

接口原型:
torch_npu.npu_mla_prolog_v2(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, *, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor, Tensor)

参数说明:
-   token_x（Tensor）：必选参数，对应公式中x。shape支持2维和3维，格式为(T, He)和(B, S, He)，dtype支持bfloat16和int8，数据格式支持ND。
-   weight_dq（Tensor）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为(He, Hcq)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ（可通过torch_npu.npu_format_cast将ND格式转为FRACTAL_NZ格式）。
-   weight_uq_qr（Tensor）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为(Hcq, N*(D+Dr))，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   weight_uk（Tensor）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为(N, D, Hckv)，dtype支持bfloat16，数据格式支持ND。
-   weight_dkv_kr（Tensor）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为(He, Hckv+Dr)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   rmsnorm_gamma_cq（Tensor）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hcq,)，dtype支持bfloat16，数据格式支持ND。
-   rmsnorm_gamma_ckv（Tensor）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hckv,)，dtype支持bfloat16，数据格式支持ND。
-   rope_sin（Tensor）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   rope_cos（Tensor）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   cache_index（Tensor）：必选参数，表示用于存储kv_cache和kr_cache的索引。shape支持1维和2维，格式为(T)和(B, S)，dtype支持int64，数据格式支持ND。
    - cache_index的取值范围为[0,BlockNum*BlockSize)，当前不会对cache_index传入值的合法性进行校验，需用户自行保证。
-   kv_cache（Tensor）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache（Tensor）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。
-   dequant_scale_x（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为pertoken。其shape支持2维，格式为(T, 1)和(BS, 1)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dq（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为perchannel。其shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_uq_qr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数维perchannel。shape支持2维，格式为(1, N*(D+Dr))，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dkv_kr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化算法为perchannel。其shape支持2维，格式为(1, Hckv+Dr)，dtype支持float，数据格式支持ND。
-   quant_scale_ckv（Tensor）：可选参数，用于对输出到kv_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Hckv)，dtype支持float，数据格式支持ND。
-   quant_scale_ckr（Tensor）：可选参数，用于对输出到kr_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Dr)，dtype支持float，数据格式支持ND。
-   smooth_scales_cq（Tensor）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   rmsnorm_epsilon_cq（float）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   rmsnorm_epsilon_ckv（float）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   cache_mode（str）：可选参数，表示kvCache的模式，支持"PA_BSND"、"PA_NZ"，其用户不特意指定时可传入默认值“PA_BSND”。

输出说明：
-   query（Tensor）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为(T, N, Hckv)和(B, S, N, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   query_rope（Tensor）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为(T, N, Dr)和(B, S, N, Dr)，dtype支持bfloat16，数据格式支持ND。
-   kv_cache_out（Tensor）：表示Key输出到kv_cache中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache_out（Tensor）：表示Key的位置编码输出到kr_cache中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。
-   dequant_scale_q_nope（Tensor）：表示Query的输出Tensor的反量化参数。其shape支持1维和3维，全量化kv_cache量化场景下，其shape为(T, N, 1)和(B*S, N, 1)；其他场景下，其shape为(1)，dtype支持float，数据格式支持ND。

约束说明:
-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   接口参数中shape格式字段含义：
    -   B：Batch表示输入样本批量大小，取值范围为0~65536。
    -   S：Seq-Length表示输入样本序列长度，取值范围为0~16。
    -   He：Head-Size表示隐藏层的大小，取值为7168。

    -   Hcq：q低秩矩阵维度，取值为1536。
    -   N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    -   Hckv：kv低秩矩阵维度，取值为512。
    -   D：qk不含位置编码维度，取值为128。
    -   Dr：qk位置编码维度，取值为64。
    -   Nkv：kv的head数，取值为1。
    -   BlockNum：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
    -   BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    -   T：BS合轴后的大小，取值范围：0~1048576。

-   shape约束：
    -   若token_x的维度采用BS合轴，即(T, He)，则rope_sin和rope_cos的shape为(T, Dr)，cache_index的shape为(T,)，dequant_scale_x的shape为(T, 1)，query的shape为(T, N, Hckv)，query_rope的shape为(T, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(T, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   若token_x的维度不采用BS合轴，即(B, S, He)，则rope_sin和rope_cos的shape为(B, S, Dr)，cache_index的shape为(B, S)，dequant_scale_x的shape为(B*S, 1)，query的shape为(B, S, N, Hckv)，query_rope的shape为(B, S, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(B*S, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        -   如果B、S、T取值为0，则query、query_rope、dequant_scale_q_nope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        -   如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。

支持的芯片型号:
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
B = 32
He=7168
Hcq=1536
Hckv=512
N=32
D=128
Dr=64
Skv=6144
S=2
Nkv=1
block_size=128
block_num=math.ceil(B*Skv/block_size)
BS = B * S

token_x = torch.rand(B, S, He).to(torch.int8).npu()
torch_npu.get_npu_format(token_x)
w_dq = torch.rand(He, Hcq).to(torch.int8).npu()
w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
torch_npu.get_npu_format(w_dq_cast)
w_uq_qr = torch.rand(Hcq, N*(D+Dr)).to(torch.int8).npu()
w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
w_dkv_kr = torch.rand(He,Hckv+Dr).to(torch.int8).npu()
w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
sin = torch.rand(B,S,Dr, dtype=torch.bfloat16).npu()
cos = torch.rand(B,S,Dr, dtype=torch.bfloat16).npu()
cache_index = torch.rand(B,S).to(torch.int64).npu()
kv_cache = torch.rand(1, block_numblock_sizeNkvHckv).to(torch.int8).npu()
kv_cache = kv_cache.view(block_num, block_size, Nkv, Hckv)
kr_cache = torch.rand(1, block_numblock_sizeNkvDr, dtype=torch.bfloat16).npu()
kr_cache = kr_cache.view(block_num, block_size, Nkv, Dr)
dequant_scale_x = torch.rand(BS, 1, dtype=torch.float).npu()
dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float).npu()
dequant_scale_w_uq_qr = torch.rand(1,N*(D+Dr), dtype=torch.float).npu()
dequant_scale_w_dkv_kr = torch.rand(1,Hckv+Dr, dtype=torch.float).npu()
quant_scale_ckv = torch.rand(1,Hckv, dtype=torch.float).npu()
cache_mode = "PA_NZ"

# 调用MlaPrologV2算子
query, query_rope, kvcache, krcache,dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x, w_dq, w_uq_qr, w_uk,
    w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x, 
    dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
    dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, cache_mode=cache_mode)

# 执行上述代码的输出类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

# 入图方式

import torch
import torch_npu
import math

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "2"
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
B = 32
He=7168
Hcq=1536
Hckv=512
N=32
D=128
Dr=64
Skv=6144
S=1
Nkv=1
block_size=128
block_num=math.ceil(B*Skv/block_size)
BS = B * S

class Model_ds(torch.nn.Module):
    def init(self):
        super().init()
    def forward(self, token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x,
        dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr,
        quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, cache_mode = "PA_BSND"):
            query, query_rope, kvcache, krcache,dequant_scale_q_nope = torch_npu.npu_mla_prolog_v2(token_x,
            w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
            sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None,
            smooth_scales_cq=None, cache_mode = cache_mode)

            return query, query_rope, kvcache, krcache, dequant_scale_q_nope

if name=="main":
    torch_npu.npu.set_device(0)

    token_x = torch.rand(B, S, He).to(torch.int8).npu()
    torch_npu.get_npu_format(token_x)
    w_dq = torch.rand(He, Hcq).to(torch.int8).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    torch_npu.get_npu_format(w_dq_cast)
    w_uq_qr = torch.rand(Hcq, N*(D+Dr)).to(torch.int8).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He,Hckv+Dr).to(torch.int8).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    sin = torch.rand(B,S,Dr, dtype=torch.bfloat16).npu()
    cos = torch.rand(B,S,Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B,S).to(torch.int64).npu()
    kv_cache = torch.rand(1, block_num*block_size*Nkv*Hckv).to(torch.int8).npu()
    kv_cache = kv_cache.view(block_num, block_size, Nkv, Hckv)
    kr_cache = torch.rand(1, block_num*block_size*Nkv*Dr, dtype=torch.bfloat16).npu()
    kr_cache = kr_cache.view(block_num, block_size, Nkv, Dr)
    dequant_scale_x = torch.rand(BS, 1, dtype=torch.float).npu()
    dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float).npu()
    dequant_scale_w_uq_qr = torch.rand(1,N*(D+Dr), dtype=torch.float).npu()
    dequant_scale_w_dkv_kr = torch.rand(1,Hckv+Dr, dtype=torch.float).npu()
    quant_scale_ckv = torch.rand(1,Hckv, dtype=torch.float).npu()
    cache_mode = "PA_NZ" # PA_BSND

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    cpu_model = Model_ds().npu()
    # 图模式调用
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    query, query_rope, kvcache, krcache,dequant_scale_q_nope = model(token_x, w_dq, w_uq_qr, w_uk,
        w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x, 
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None, 
        smooth_scales_cq=None, cache_mode=cache_mode)

    # 单算子调用
    query, query_rope, kvcache, krcache,dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x, w_dq, w_uq_qr, w_uk,
        w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache, dequant_scale_x=dequant_scale_x, 
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, cache_mode=cache_mode)

# 执行上述代码的输出类似如下
single op output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

graph output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],        device='npu:0', dtype=torch.bfloat16)
"""
)

_add_torch_npu_docstr(
    "npu_mla_prolog_v3",
    """
功能描述:
推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分五路，首先对输入x乘以WeightDq进行下采样和RmsNorm后分成两路，第一路乘以WeightUq和WeightUk经过两次上采样后得到query；第二路乘以WeightQr后经过旋转位置编码（ROPE)得到query_rope；第三路是输入x乘以WeightDkv进行下采样和RmsNorm后传入Cache中得到kvCache；第四路是输入x乘以Wkr后经过旋转位置编码后传入另一个Cache中得到krCache；第五路是输出query经过DynamicQuant后得到的量化参数。

接口原型:
torch_npu.npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND", float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)

参数说明:
-   token_x（Tensor）：必选参数，对应公式中x。shape支持2维和3维，格式为(T, He)和(B, S, He)，dtype支持bfloat16和int8，数据格式支持ND。
-   weight_dq（Tensor）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为(He, Hcq)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ（可通过torch_npu.npu_format_cast将ND格式转为FRACTAL_NZ格式）。
-   weight_uq_qr（Tensor）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为(Hcq, N*(D+Dr))，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   weight_uk（Tensor）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为(N, D, Hckv)，dtype支持bfloat16，数据格式支持ND。
-   weight_dkv_kr（Tensor）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为(He, Hckv+Dr)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   rmsnorm_gamma_cq（Tensor）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hcq,)，dtype支持bfloat16，数据格式支持ND。
-   rmsnorm_gamma_ckv（Tensor）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hckv,)，dtype支持bfloat16，数据格式支持ND。
-   rope_sin（Tensor）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   rope_cos（Tensor）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   kv_cache（Tensor）：必选参数，表示用于cache索引的aclTensor（本质in-place更新）。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache（Tensor）：必选参数，表示用于key位置编码的cache（本质in-place更新）。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。
-   cache_index（Tensor）：可选参数，表示用于存储kv_cache和kr_cache的索引。shape支持1维和2维，格式为(T)和(B, S)，dtype支持int64，数据格式支持ND。
-   cache_index的取值范围为[0,BlockNum*BlockSize)，当前不会对cache_index传入值的合法性进行校验，需用户自行保证。
-   dequant_scale_x（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为pertoken。其shape支持2维，格式为(T, 1)和(BS, 1)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dq（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为perchannel。其shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_uq_qr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数维perchannel。shape支持2维，格式为(1, N*(D+Dr))，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dkv_kr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化算法为perchannel。其shape支持2维，格式为(1, Hckv+Dr)，dtype支持float，数据格式支持ND。
-   quant_scale_ckv（Tensor）：可选参数，用于对输出到kv_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Hckv)，dtype支持float，数据格式支持ND。
-   quant_scale_ckr（Tensor）：可选参数，用于对输出到kr_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Dr)，dtype支持float，数据格式支持ND。
-   smooth_scales_cq（Tensor）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   rmsnorm_epsilon_cq（float）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   rmsnorm_epsilon_ckv（float）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   cache_mode（str）：可选参数，表示kvCache的模式，支持"PA_BSND"、"PA_NZ"，其用户不特意指定时可传入默认值“PA_BSND”。
-   qc_qr_scale（float）：可选参数，表示Query的尺度矫正参数，不传入的时候默认值为1.0。
-   kc_scale（float）：可选参数，表示Key的尺度矫正参数，不传入的时候默认值为1.0。

输出说明：
-   query（Tensor）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为(T, N, Hckv)和(B, S, N, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   query_rope（Tensor）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为(T, N, Dr)和(B, S, N, Dr)，dtype支持bfloat16，数据格式支持ND。
-   dequant_scale_q_nope（Tensor）：表示Query的输出Tensor的反量化参数。其shape支持1维和3维，全量化kv_cache量化场景下，其shape为(T, N, 1)和(B*S, N, 1)；其他场景下，其shape为(1)，dtype支持float，数据格式支持ND。
-   query_norm（Tensor）：预留输出，默认生成shape为(1,)的零张量，dtype支持bfloat16和int8，数据格式支持ND。
-   dequant_scale_q_norm（Tensor）：预留输出，默认生成shape为(1,)的零张量，dtype支持float，数据格式支持ND。

约束说明:
-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   接口参数中shape格式字段含义：
    -   B：Batch表示输入样本批量大小，取值范围为0~65536。
    -   S：Seq-Length表示输入样本序列长度，取值范围为0~16。
    -   He：Head-Size表示隐藏层的大小，取值为7168、7680或6144。

    -   Hcq：q低秩矩阵维度，取值为1536。
    -   N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    -   Hckv：kv低秩矩阵维度，取值为512。
    -   D：qk不含位置编码维度，取值为128。
    -   Dr：qk位置编码维度，取值为64。
    -   Nkv：kv的head数，取值为1。
    -   BlockNum：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
    -   BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    -   T：BS合轴后的大小，取值范围：0~1048576。

-   shape约束：
    -   若token_x的维度采用BS合轴，即(T, He)，则rope_sin和rope_cos的shape为(T, Dr)，cache_index的shape为(T,)，dequant_scale_x的shape为(T, 1)，query的shape为(T, N, Hckv)，query_rope的shape为(T, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(T, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   若token_x的维度不采用BS合轴，即(B, S, He)，则rope_sin和rope_cos的shape为(B, S, Dr)，cache_index的shape为(B, S)，dequant_scale_x的shape为(B*S, 1)，query的shape为(B, S, N, Hckv)，query_rope的shape为(B, S, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(B*S, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        -   如果B、S、T取值为0，则query、query_rope、dequant_scale_q_nope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        -   如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。

支持的芯片型号:
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
B = 2
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 6144
S = 2
Nkv = 1
BlockSize = 128
BlockNum = math.ceil(B * Skv / BlockSize)
T = 8

token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
rmsnorm_epsilon_cq = 1.0e-5
rmsnorm_epsilon_ckv = 1.0e-5
cache_mode = "PA_BSND"
qc_qr_scale = 10.0
kc_scale = 10.0
dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

# 调用MlaPrologV3算子
query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch.ops.npu.mla_prolog_npu_v3(token_x, w_dq_cast, 
    w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
    cache_mode, qc_qr_scale, kc_scale, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
    quant_scale_ckv, smooth_scale_cq)

# 执行上述代码的输出类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

# 入图方式

import torch
import torch_npu
import math

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "2"
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
B = 2
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 6144
S = 1
Nkv = 1
BlockSize = 128
BlockNum = math.ceil(B * Skv / BlockSize)
T = 8

class Model_ds(torch.nn.Module):
    def init(self):
        super().init()
    def forward(self, token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, kv_cache, kr_cache, cache_index, dequant_scale_x,
        dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr,
        quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, epsilon_cq = 0.00001, epsilon_ckv = 0.00001,
        cache_mode = "PA_BSND", qc_qr_scale = 1.0, kc_scale = 1.0):
            return torch_npu.npu_mla_prolog_v3(token_x,
            w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
            sin, cos, kv_cache, kr_cache, cache_index=cache_index, dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None,
            smooth_scales_cq=None, epsilon_cq=epsilon_cq, epsilon_ckv=epsilon_ckv, cache_mode=cache_mode, 
            qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)


if name=="main":
    torch_npu.npu.set_device(0)

    token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
    w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
    kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
    kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
    kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
    kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    qc_qr_scale = 10.0
    kc_scale = 10.0
    dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
    dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
    dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
    dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
    quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
    smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    cpu_model = Model_ds().npu()
    # 图模式调用
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = model(token_x, w_dq, w_uq_qr, w_uk,
        w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, kv_cache, kr_cache, cache_index=cache_index, dequant_scale_x=dequant_scale_x, 
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None, 
        smooth_scales_cq=None, epsilon_cq=epsilon_cq, epsilon_ckv=epsilon_ckv, cache_mode=cache_mode, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

    # 单算子调用
    query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch.ops.npu.mla_prolog_npu_v3(token_x, w_dq_cast, 
        w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
        cache_mode, qc_qr_scale, kc_scale, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
        quant_scale_ckv, smooth_scale_cq)

# 执行上述代码的输出类似如下
single op output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

graph output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],        device='npu:0', dtype=torch.bfloat16)
"""
)

_add_torch_npu_docstr(
    "npu_mla_prolog_v3_functional",
    """
功能描述:
推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分五路，首先对输入x乘以WeightDq进行下采样和RmsNorm后分成两路，第一路乘以WeightUq和WeightUk经过两次上采样后得到query；第二路乘以WeightQr后经过旋转位置编码（ROPE)得到query_rope；第三路是输入x乘以WeightDkv进行下采样和RmsNorm后传入Cache中得到kvCache；第四路是输入x乘以Wkr后经过旋转位置编码后传入另一个Cache中得到krCache；第五路是输出query经过DynamicQuant后得到的量化参数。

接口原型:
torch_npu.npu_mla_prolog_v3_functional(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, Tensor kv_cache, Tensor kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, str cache_mode="PA_BSND", float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)

参数说明:
-   token_x（Tensor）：必选参数，对应公式中x。shape支持2维和3维，格式为(T, He)和(B, S, He)，dtype支持bfloat16和int8，数据格式支持ND。
-   weight_dq（Tensor）：必选参数，表示计算Query的下采样权重矩阵，即公式中W<sup>DQ</sup>。shape支持2维，格式为(He, Hcq)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ（可通过torch_npu.npu_format_cast将ND格式转为FRACTAL_NZ格式）。
-   weight_uq_qr（Tensor）：必选参数，表示计算Query的上采样权重矩阵和Query的位置编码权重矩阵，即公式中W<sup>UQ</sup>和W<sup>QR</sup>。shape支持2维，格式为(Hcq, N*(D+Dr))，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   weight_uk（Tensor）：必选参数，表示计算Key的上采样权重，即公式中W<sup>UK</sup>。shape支持3维，格式为(N, D, Hckv)，dtype支持bfloat16，数据格式支持ND。
-   weight_dkv_kr（Tensor）：必选参数，表示计算Key的下采样权重矩阵和Key的位置编码权重矩阵，即公式中W<sup>DKV</sup>和W<sup>KR</sup>。shape支持2维，格式为(He, Hckv+Dr)，dtype支持bfloat16和int8，数据格式支持FRACTAL_NZ。
-   rmsnorm_gamma_cq（Tensor）：必选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hcq,)，dtype支持bfloat16，数据格式支持ND。
-   rmsnorm_gamma_ckv（Tensor）：必选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的_γ_参数。shape支持1维，格式为(Hckv,)，dtype支持bfloat16，数据格式支持ND。
-   rope_sin（Tensor）：必选参数，表示用于计算旋转位置编码的正弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   rope_cos（Tensor）：必选参数，表示用于计算旋转位置编码的余弦参数矩阵。shape支持2维和3维，格式为(T, Dr)和(B, S, Dr)，dtype支持bfloat16，数据格式支持ND。
-   kv_cache（Tensor）：必选参数，表示用于cache索引的aclTensor。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache（Tensor）：必选参数，表示用于key位置编码的cache。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。
-   cache_index（Tensor）：可选参数，表示用于存储kv_cache和kr_cache的索引。shape支持1维和2维，格式为(T)和(B, S)，dtype支持int64，数据格式支持ND。
-   cache_index的取值范围为[0,BlockNum*BlockSize)，当前不会对cache_index传入值的合法性进行校验，需用户自行保证。
-   dequant_scale_x（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为pertoken。其shape支持2维，格式为(T, 1)和(BS, 1)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dq（Tensor）：可选参数，用于输入token_x为int8类型时，下采样后进行反量化操作时的参数，token_x量化方式为perchannel。其shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   dequant_scale_w_uq_qr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化参数维perchannel。shape支持2维，格式为(1, N*(D+Dr))，dtype支持float，数据格式支持ND。
-   dequant_scale_w_dkv_kr（Tensor）：可选参数，用于对MatmulQcQr矩阵乘后进行反量化操作时的参数，量化算法为perchannel。其shape支持2维，格式为(1, Hckv+Dr)，dtype支持float，数据格式支持ND。
-   quant_scale_ckv（Tensor）：可选参数，用于对输出到kv_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Hckv)，dtype支持float，数据格式支持ND。
-   quant_scale_ckr（Tensor）：可选参数，用于对输出到kr_cache_out中的数据做量化操作时的参数。shape支持2维，格式为(1, Dr)，dtype支持float，数据格式支持ND。
-   smooth_scales_cq（Tensor）：可选参数，用于对RmsNormCq输出做动态量化操作时的参数。shape支持2维，格式为(1, Hcq)，dtype支持float，数据格式支持ND。
-   rmsnorm_epsilon_cq（float）：可选参数，表示计算c<sup>Q</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   rmsnorm_epsilon_ckv（float）：可选参数，表示计算c<sup>KV</sup>的RmsNorm公式中的ε参数，用户不特意指定时可传入默认值1e-05。
-   cache_mode（str）：可选参数，表示kvCache的模式，支持"PA_BSND"、"PA_NZ"，其用户不特意指定时可传入默认值“PA_BSND”。
-   qc_qr_scale（float）：可选参数，表示Query的尺度矫正参数，不传入的时候默认值为1.0。
-   kc_scale（float）：可选参数，表示Key的尺度矫正参数，不传入的时候默认值为1.0。

输出说明：
-   query（Tensor）：表示Query的输出Tensor，即公式中q<sup>N</sup>。shape支持3维和4维，格式为(T, N, Hckv)和(B, S, N, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   query_rope（Tensor）：表示Query位置编码的输出Tensor，即公式中q<sup>R</sup>。shape支持3维和4维，格式为(T, N, Dr)和(B, S, N, Dr)，dtype支持bfloat16，数据格式支持ND。
-   dequant_scale_q_nope（Tensor）：表示Query的输出Tensor的反量化参数。其shape支持1维和3维，全量化kv_cache量化场景下，其shape为(T, N, 1)和(B*S, N, 1)；其他场景下，其shape为(1)，dtype支持float，数据格式支持ND。
-   query_norm（Tensor）：预留输出，默认生成shape为(1,)的零张量，dtype支持bfloat16和int8，数据格式支持ND。
-   dequant_scale_q_norm（Tensor）：预留输出，默认生成shape为(1,)的零张量，dtype支持float，数据格式支持ND。
-   kv_cache_out（Tensor）：表示Key输出到kv_cache中的Tensor（本质in-place更新），即公式中k<sup>C</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Hckv)，dtype支持bfloat16和int8，数据格式支持ND。
-   kr_cache_out（Tensor）：表示Key的位置编码输出到kr_cache中的Tensor（本质in-place更新），即公式中k<sup>R</sup>。shape支持4维，格式为(BlockNum, BlockSize, Nkv, Dr)，dtype支持bfloat16和int8，数据格式支持ND。

约束说明:
-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   接口参数中shape格式字段含义：
    -   B：Batch表示输入样本批量大小，取值范围为0~65536。
    -   S：Seq-Length表示输入样本序列长度，取值范围为0~16。
    -   He：Head-Size表示隐藏层的大小，取值为7168、7680或6144。

    -   Hcq：q低秩矩阵维度，取值为1536。
    -   N：Head-Num表示多头数，取值范围为1、2、4、8、16、32、64、128。

    -   Hckv：kv低秩矩阵维度，取值为512。
    -   D：qk不含位置编码维度，取值为128。
    -   Dr：qk位置编码维度，取值为64。
    -   Nkv：kv的head数，取值为1。
    -   BlockNum：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度，该值允许取0。
    -   BlockSize：PagedAttention场景下的块大小，取值范围为16、128。
    -   T：BS合轴后的大小，取值范围：0~1048576。

-   shape约束：
    -   若token_x的维度采用BS合轴，即(T, He)，则rope_sin和rope_cos的shape为(T, Dr)，cache_index的shape为(T,)，dequant_scale_x的shape为(T, 1)，query的shape为(T, N, Hckv)，query_rope的shape为(T, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(T, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   若token_x的维度不采用BS合轴，即(B, S, He)，则rope_sin和rope_cos的shape为(B, S, Dr)，cache_index的shape为(B, S)，dequant_scale_x的shape为(B*S, 1)，query的shape为(B, S, N, Hckv)，query_rope的shape为(B, S, N, Dr)。全量化kv_cache量化场景下，dequant_scale_q_nope的shape为(B*S, N, 1)，其他场景下dequant_scale_q_nope的shape为(1)。
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        -   如果B、S、T取值为0，则query、query_rope、dequant_scale_q_nope输出空Tensor，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新。
        -   如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache、kv_cache_out、kr_cache_out不更新，即输出空Tensor。

支持的芯片型号:
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例:
# 单算子调用方式
import torch
import torch_npu
import math

# 生成随机数据, 并发送到npu
B = 2
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 6144
S = 2
Nkv = 1
BlockSize = 128
BlockNum = math.ceil(B * Skv / BlockSize)
T = 8

token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
rmsnorm_epsilon_cq = 1.0e-5
rmsnorm_epsilon_ckv = 1.0e-5
cache_mode = "PA_BSND"
qc_qr_scale = 10.0
kc_scale = 10.0
dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

# 调用npu_mla_prolog_v3_functional算子
query_mla, query_rope_mla, dequant_scale_q_nope_mla, _, _, kv_cache_mla, kr_cache_mla = torch.ops.npu.npu_mla_prolog_v3_functional(token_x, w_dq_cast, 
    w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
    cache_mode, qc_qr_scale, kc_scale, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
    quant_scale_ckv, smooth_scale_cq)


# 执行上述代码的输出类似如下
tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

# 入图方式

import torch
import torch_npu
import math

import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.aoe_config.aoe_mode = "2"
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
B = 2
He = 7168
Hcq = 1536
Hckv = 512
N = 32
D = 128
Dr = 64
Skv = 6144
S = 1
Nkv = 1
BlockSize = 128
BlockNum = math.ceil(B * Skv / BlockSize)
T = 8

class Model_ds(torch.nn.Module):
    def init(self):
        super().init()
    def forward(self, token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, kv_cache, kr_cache, cache_index, dequant_scale_x,
        dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr,
        quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, epsilon_cq = 0.00001, epsilon_ckv = 0.00001,
        cache_mode = "PA_BSND", qc_qr_scale = 1.0, kc_scale = 1.0):
            return torch_npu.npu_mla_prolog_v3_functional(token_x,
            w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
            sin, cos, kv_cache, kr_cache, cache_index=cache_index, dequant_scale_x=dequant_scale_x,
            dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None,
            smooth_scales_cq=None, epsilon_cq=epsilon_cq, epsilon_ckv=epsilon_ckv, cache_mode=cache_mode, 
            qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)


if name=="main":
    torch_npu.npu.set_device(0)

    token_x = torch.randint(-100, 100, (B, S, He), dtype=torch.int8).npu()
    w_dq = torch.randint(-100, 100, (He, Hcq), dtype=torch.int8).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.randint(-100, 100, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.randint(-100, 100, (He, Hckv + Dr), dtype=torch.int8).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.randint(0, B * S, (B, S), dtype=torch.int64).npu()
    kv_cache = torch.randint(-100, 100, (1, BlockNum * BlockSize * Nkv * Hckv), dtype=torch.int8).npu()
    kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
    kr_cache = torch.rand(1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16).npu()
    kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    qc_qr_scale = 10.0
    kc_scale = 10.0
    dequant_scale_x = torch.rand(B * S, 1, dtype=torch.float32).npu()
    dequant_scale_w_dq = torch.rand(1, Hcq, dtype=torch.float32).npu()
    dequant_scale_w_uqqr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
    dequant_scale_w_dkvkr = torch.rand(1, Hckv + Dr, dtype=torch.float32).npu()
    quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32).npu()
    smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    cpu_model = Model_ds().npu()
    # 图模式调用
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    query_mla, query_rope_mla, dequant_scale_q_nope_mla, _, _, kv_cache_mla, kr_cache_mla = model(token_x, w_dq, w_uq_qr, w_uk,
        w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, kv_cache, kr_cache, cache_index=cache_index, dequant_scale_x=dequant_scale_x, 
        dequant_scale_w_dq=dequant_scale_w_dq, dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, 
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr, quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=None, 
        smooth_scales_cq=None, epsilon_cq=epsilon_cq, epsilon_ckv=epsilon_ckv, cache_mode=cache_mode, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale)

    # 单算子调用
    query_mla, query_rope_mla, dequant_scale_q_nope_mla, _, _, kv_cache_mla, kr_cache_mla = torch.ops.npu.npu_mla_prolog_v3_functional(token_x, w_dq_cast, 
        w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_copy, kr_cache_copy, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv,
        cache_mode, qc_qr_scale, kc_scale, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uqqr, dequant_scale_w_dkvkr,
        quant_scale_ckv, smooth_scale_cq)

# 执行上述代码的输出类似如下
single op output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        device='npu:0', dtype=torch.bfloat16)

graph output: tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],        device='npu:0', dtype=torch.bfloat16)
"""
)

_add_torch_npu_docstr(
    "npu_all_gather_base_mm",
    """
接口原型：
torch_npu.npu_all_gather_base_mm(input, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, gather_index=0, bool gather_output=True, comm_turn=0, output_dtype=None, comm_mode=None) -> (Tensor, Tensor)

功能描述
TP切分场景下, 实现allgather和matmul的融合, 实现通信和计算流水并行. 
使用该接口时, 请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本, 否则将会引发报错, 比如BUS ERROR等. 

参数说明
input: Tensor类型, 数据类型支持float16、bfloat16、int8, 数据格式支持ND, 输入shape支持2维, 形如(m, k)、(k, n), 轴满足matmul算子入参要求, k轴相等, 且k轴取值范围为[256, 65535). 
x2: Tensor类型, 数据类型、输入shape维度需要和input保持一致, 数据格式支持ND、NZ。NZ仅在comm_mode为aiv时支持。
hcom: String类型, 通信域handle名, 通过get_hccl_comm_name接口获取. 
world_size: int类型, 通信域内的rank总数, 仅支持为2、4、8. 
*: 代表其之前的变量是位置相关, 按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
bias: Tensor类型, 可选输入, 数据类型支持float16、bfloat16, 数据格式支持ND格式. 数据类型需要和input保持一致. bias仅支持一维, 且维度大小与output的第1维大小相同. 当前版本暂不支持bias输入为非0的场景. 
x1_scale: 可选Tensor类型，mm左矩阵反量化参数。数据类型支持float32，数据格式支持ND格式。数据维度为(m, 1), 支持pertoken量化。
x2_scale: 可选Tensor类型。mm右矩阵反量化参数。数据类型支持float32、int64，数据格式支持ND格式。数据维度为(1, n), 支持perchannel量化。如需传入int64数据类型的，需要提前调用torch_npu.npu_trans_quant_param来获取int64数据类型的x2_scale。
gather_index: int类型, 表示gather操作对象, 0: 对input做gather, 1: 对x2做gather. 默认值0. 当前版本仅支持输入0. 
gather_output: bool类型, 表示是否需要gather输出， 默认值true。
comm_turn: int类型, 表示rank间通信切分粒度, 默认值: 0, 表示默认的切分方式. 当前版本仅支持输入0. 
output_dtype ：可选dtype参数。表示第一个输出的数据类型。仅支持在量化场景且x1_scale和x2_scale均为float32时，可指定输出数据类型为bfloat16或float1，默认值为bfloat16。
comm_mode：可选str参数。表示通信模式，支持ai_cpu、aiv两种模式。ai_cpu模式仅支持基础场景。aiv模式支持基础场景和量化场景。


输出说明
两个输出, 均为Tensor类型: (Tensor, Tensor)
-   Tensor：第一个输出Tensor是allgather+matmul的结果。
基础场景时数据类型和input保持一致。
量化场景下，x2_scale为int64数据类型时，输出数据类型为float16。x1_scale和x2_scale均为float32时, 输出数据类型由output_dtype指定，默认为bfloat16。
-   Tensor：第二个输出Tensor是allgather的结果。

约束说明
该接口支持训练场景下使用. 
该接口支持图模式. 
Atlas A2 训练系列产品支持2、4、8卡,  支持hccs链路all mesh组网(每张卡和其它卡两两相连). 
Atlas A3 训练系列产品支持2、4、8、16卡,  支持hccs链路double ring组网(多张卡按顺序组成一个圈, 每张卡只和左右卡相连). 
input不支持输入转置后的tensor, x2转置后输入, 需要满足shape的第一维大小与x1的最后一维相同, 满足matmul的计算条件. 
Atlas A2 训练系列产品: 一个模型中的通算融合算子(AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce), 仅支持相同通信域. 

支持的PyTorch版本
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11.0

支持的型号
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例
单算子模式调用
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    tensor_allgather_shape = x1_shape
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]

    input_ = torch.randn(single_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    output, gather_out = torch_npu.npu_all_gather_base_mm(input_, weight, hcomm_info, world_size)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16

    mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
图模式调用
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, weight, hcomm_info, world_size, gather_output):
        output, gather_output = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size,
                                                                 gather_output=gather_output)
        return output, gather_output
def define_model(model, graph_type):
    import torchair
    if graph_type == 1:  # 传统入图模式, 静态shape+在线编译场景
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  # ACLNN入图模式, 动态shape+二进制
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    else:
        print("Error type")
    return model
def get_graph(input, weight, hcomm_info, world_size, gather_output):
    model = ALLGATHER_MM_GRAPH_Model()
    model = define_model(model, 2)
    model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
    output_npu = model_output[0]
    gather_output_npu = model_output[1]
    return output_npu, gather_output_npu
def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    input = torch.randn(single_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    is_gather_out = True
    output, gather_out = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
    print("output:", output)
if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
"""
)

_add_torch_npu_docstr(
    "npu_group_norm_silu",
    """
接口原型：
torch_npu.npu_group_norm_silu(Tensor input, Tensor weight, Tensor bias, int group, float eps) -> (Tensor, Tensor, Tensor)

功能描述
计算输入input的组归一化结果out、均值meanOut、标准差的倒数rstdOut、以及silu的输出. 

参数说明
input: Tensor类型, 必选输入, 源数据张量, 维度需大于一维, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 
weight: Tensor类型, 必选输入, 索引张量, 维度为1且元素数量需与输入input的第1维度保持相同, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 
bias: Tensor类型, 必选输入, 更新数据张量, 维度为1元素数量需与输入input的第1维度保持相同, 数据格式支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 
group: int类型, 必选输入, 表示将输入input的第1维度分为group组. 
eps: float类型, 可选参数, 数值稳定性而加到分母上的值, 若保持精度, 则eps需大于0. 

输出说明
out: Tensor类型, 数据类型和shape与input相同, 支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 
meanOut: Tensor类型, 数据类型与input相同, shape为(N, group)支持ND, 支持非连续的Tensor. 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 
rstdOut: Tensor类型, 数据类型与input相同, shape为(N, group). 
Atlas 推理系列产品: 数据类型支持float16、float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float16、float、bfloat16. 

约束说明
该接口支持图模式. 
input、weight、bias、out、meanOut、rstdOut数据类型必须支持的范围之内. 
out、meanOut、rstdOut的数据类型与input相同; weight、bias与input可以不同. 
input第1维度能整除group. 
out的shape与input相同. 
meanOut与rstdOut的shape为(N, group), 其中N为input第0维度值. 
weight与bias的数据类型必须保持一致, 且数据类型的精度不能低于input的数据类型. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列产品

调用示例
单算子调用: 
import torch
import numpy as np
import torch_npu
 
dtype = np.float32
shape_x = [24,320,48,48]
num_groups = 32
shape_c = [320]
eps = 0.00001
 
x_npu=torch.randn(shape_x,dtype=torch.float32).npu()
gamma_npu=torch.randn(shape_c,dtype=torch.float32).npu()
beta_npu=torch.randn(shape_c,dtype=torch.float32).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
 
 
x_npu=torch.randn(shape_x,dtype=torch.bfloat16).npu()
gamma_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
beta_npu=torch.randn(shape_c,dtype=torch.bfloat16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
 
x_npu=torch.randn(shape_x,dtype=torch.float16).npu()
gamma_npu=torch.randn(shape_c,dtype=torch.float16).npu()
beta_npu=torch.randn(shape_c,dtype=torch.float16).npu()
out_npu, mean_npu, rstd_out = torch_npu.npu_group_norm_silu(x_npu, gamma_npu, beta_npu, group=num_groups, eps=eps)
"""
)

_add_torch_npu_docstr(
    "npu_mm_reduce_scatter_base",
    """
接口原型：
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None) -> Tensor

功能描述
TP切分场景下, 实现matmul和reduce_scatter的融合, 融合算子内部实现计算和通信流水并行. 支持perchanel, pertoken量化。
使用该接口时, 请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本, 否则将会引发报错, 比如BUS ERROR等. 

参数说明
input: Tensor类型, 数据类型支持float16、bfloat16、int8, 数据格式支持ND, 输入shape支持2维. 
x2: Tensor类型, 数据类型支持float16、bfloat16、int8, 数据格式支持ND、NZ。NZ仅在comm_mode为aiv时支持。数据类型需要和input保持一致, 输入shape维度和input保持一致. 
hcom: String类型, 通信域handle名, 通过get_hccl_comm_name接口获取. 
world_size: int类型, 通信域内的rank总数, 仅支持为2、4、8. 
*: 代表其之前的变量是位置相关, 按照顺序输入, 必选; 之后的变量是键值对赋值的, 位置无关, 可选(不输入会使用默认值). 
reduce_op: String类型, reduce操作类型, 当前仅支持'sum', 默认值: 'sum'. 
bias: Tensor类型, 可选输入, 数据类型支持float16、bfloat16, 数据格式支持ND格式. 数据类型需要和input保持一致. bias仅支持一维, 且维度大小与output的第1维大小相同. 当前版本暂不支持bias输入为非0的场景. 
x1_scale: Tensor类型，可选参数。mm左矩阵反量化参数。数据类型支持float32，数据格式支持$ND$格式。数据维度为(m, 1), 支持pertoken量化。
x2_scale： Tensor类型，可选参数。mm左矩阵反量化参数。数据类型支持float32、int64，数据格式支持$ND$格式。数据维度为(1, n), 支持perchannel量化。如需传入int64数据类型的，需要提前调用torch_npu.npu_trans_quant_param来获取int64数据类型的x2_scale。
comm_turn：int类型, 可选参数。表示rank间通信切分粒度，默认值为0，表示默认的切分方式。当前版本仅支持输入0。
output_dtype: ScalarType, 可选参数。表示输出数据类型。仅支持在量化场景且x1_scale和x2_scale均为float32时，可指定输出数据类型为bfloat16或float16，默认值为bfloat16。
comm_mode：str类型，可选参数。表示通信模式，支持ai_cpu、aiv两种模式。ai_cpu模式仅支持基础场景。aiv模式支持基础场景和量化场景。

输出说明
shape维度和input保持一致。
基础场景时数据类型和input保持一致。
量化场景下，x2_scale为int64数据类型时，输出数据类型为float16。x1_scale和x2_scale均为float32时, 输出数据类型由output_dtype指定，默认为torch.bfloat16。

约束说明
comm_mode为ai_cpu时：
该接口仅在训练场景下使用. 
该接口支持图模式. 
输入input、x2必须是2维, 分别为(m, k)、(k, n), 轴满足matmul算子入参要求, k轴相等, 且k轴取值范围为[256, 65535), m轴约束如下: 
m轴需要整除world_size. 
Atlas A2 训练系列产品支持2、4、8卡,  支持hccs链路all mesh组网(每张卡和其它卡两两相连). 
Atlas A3 训练系列产品支持2、4、8、16卡,  支持hccs链路double ring组网(多张卡按顺序组成一个圈, 每张卡只和左右卡相连). 
input不支持输入转置后的tensor, x2转置后输入, 需要满足shape的第一维大小与input的最后一维相同, 满足matmul的计算条件. 
Atlas A2 训练系列产品: 一个模型中的通算融合算子(AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce), 仅支持相同通信域. 
comm_mode为aiv时：
- 支持Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- 支持Atlas A3 训练系列产品/Atlas A3 推理系列产品

支持的PyTorch版本
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11.0

支持的型号
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例
单算子模式调用
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    input_ = torch.randn(x1_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    output = torch_npu.npu_mm_reduce_scatter_base(input_, weight, hcomm_info, world_size)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16

    mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
图模式调用
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
class MM_REDUCESCATTER_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, weight, hcomm_info, world_size, reduce_op):
        output = torch_npu.npu_mm_reduce_scatter_base(input, weight, hcomm_info, world_size,
                                                      reduce_op=reduce_op)
        return output
def define_model(model, graph_type):
    import torchair
    if graph_type == 1:  # 传统入图模式, 静态shape+在线编译场景
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  # ACLNN入图模式, 动态shape+二进制
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    else:
        print("Error type")
    return model
def get_graph(input, weight, hcomm_info, world_size):
    model = MM_REDUCESCATTER_GRAPH_Model()
    model = define_model(model, 2)
    model_output = model(input, weight, hcomm_info, world_size, reduce_op="sum")
    return model_output
def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    input = torch.randn(x1_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    output = get_graph(input, weight, hcomm_info, world_size)
    print("output:", output)
if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
"""
)

_add_torch_npu_docstr(
    "npu_moe_compute_expert_tokens",
    """
接口原型：
torch_npu.npu_moe_compute_expert_tokens(Tensor sorted_expert_for_source_row, int num_expert) -> Tensor

功能描述
算子功能: MoE(Mixture of Experts, 混合专家模型)计算中, 通过二分查找的方式查找每个专家处理的最后一行的位置. 
计算公式: 
expertTokens_{i}=BinaerSearch(sortedExpertForSourceRow,numExpert)

参数说明
sorted_expert_for_source_row: Tensor类型, 必选参数, 经过专家处理过的结果, 要求是一个1D的Tensor, 数据类型支持int32, 数据格式要求为ND. shape大小需要小于2147483647. 
num_expert: int类型, 必选参数, 总专家数. 

输出说明
expertTokens: Tensor类型, 公式中的输出, 要求的是一个1D的Tensor, 数据类型与sorted_expert_for_source_row保持一致. 

约束说明
该接口支持推理场景下使用. 
该接口支持图模式. 

支持的PyTorch版本
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 2.0
PyTorch 1.11.0

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例
单算子模式调用
import torch
import torch_npu
sorted_experts = torch.tensor([3,3,4,5,6,7], dtype=torch.int32)
num_experts = 5
output = torch_npu.npu_moe_compute_expert_tokens(sorted_experts.npu(), num_experts)
图模式调用
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
class GMMModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, sorted_experts, num_experts):
        return torch_npu.npu_moe_compute_expert_tokens(sorted_experts, num_experts)
def main():
    sorted_experts = torch.tensor([3,3,4,5,6,7], dtype=torch.int32)
    num_experts = 5
    model = GMMModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    custom_output = model(sorted_experts, num_experts)
if __name__ == '__main__':
    main()
"""
)

_add_torch_npu_docstr(
    "npu_moe_finalize_routing",
    """
接口原型：
torch_npu.npu_moe_finalize_routing(Tensor expanded_permuted_rows, Tensor? skip1, Tensor? skip2, Tensor? bias, Tensor? scales, Tensor expanded_src_to_dst_row, Tensor? export_for_source_row, int? drop_pad_mode=0) -> Tensor

功能描述
算子功能: MoE计算中, 最后处理合并MoE FFN的输出结果. 
计算公式: 
expertid=exportForSourceRow[i,k]
out(i,j)=skip1_{i,j}+skip2Optional_{i,j}+$\sum_{k=0}^{K}$(sclaes_{i,k}*(expandPermutedRowx_{expandedSrcRowToDstRowx_{i}+k*num_rows_{j}}+bias_{expertid_{j}}))

参数说明
expanded_permuted_rows: Tensor类型, 必选参数, 经过专家处理过的结果, 要求是一个2D的Tensor, 数据类型支持float16、bfloat16、float32, 数据格式要求为ND. shape支持(NUM_ROWS * K, H), NUM_ROWS为行数, K为从总的专家E中选出K个专家, H为列数. 
skip1: Tensor类型, 可选参数, 求和的输入参数1, 要求是一个2D的Tensor, 数据类型要求与expanded_permuted_rows一致 , shape要求与输出out的shape一致. 
skip2: Tensor类型, 可选参数, 求和的输入参数2, 要求是一个2D的Tensor, 数据类型要求与expanded_permuted_rows一致 , shape要求与输出out的shape一致. skip2参数为None时, skip1参数必须也为None. 
bias: Tensor类型, 可选参数, 专家的偏差, 要求是一个2D的Tensor, 数据类型要求与expanded_permuted_rows一致. shape支持(E, H), E为总的专家个数, H为列数. 
scales: Tensor类型, 可选参数, 专家的权重, 要求是一个2D的Tensor, 数据类型要求与expanded_permuted_rows一致, shape支持(NUM_ROWS, K). 
expanded_src_to_dst_row: Tensor类型, 必选参数, 保存每个专家处理结果的索引, 要求是一个1D的Tensor, 数据类型支持int32. shape支持(NUM_ROWS * K), NUM_ROWS为行数, K为从总的专家E中选出K个专家, drop_pad_mode参数为0时, Tensor中的值取值范围是[0, NUM_ROWS * K-1]. 
export_for_source_row: Tensor类型, 可选参数, 每行处理的专家号, 要求是一个2D的Tensor, 数据类型支持int32. shape支持(NUM_ROWS, K), NUM_ROWS为行数, K为从总的专家E中选出K个专家. 
drop_pad_mode: int类型, 可选参数, 表示是否支持丢弃模式, 取值范围为0, 默认值为0. 

输出说明
out: Tensor类型, 最后处理合并MoE FFN的输出结果. 

约束说明
该接口支持推理场景下使用. 
该接口支持图模式. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例
单算子模式调用
import torch
import torch_npu

expert_num = 16
token_len = 10
top_k = 4
num_rows = 50
device =torch.device('npu')
dtype = torch.float32
expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
drop_pad_mode = 0
output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
图模式调用
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
class GMMModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode):
        return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
def main():
    expert_num = 16
    token_len = 10
    top_k = 4
    num_rows = 50
    device =torch.device('npu')
    dtype = torch.float32
    expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
    skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
    skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
    bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
    scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
    expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
    expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
    drop_pad_mode = 0
    model = GMMModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    custom_output = model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
if __name__ == '__main__':
    main()
"""
)

_add_torch_npu_docstr(
    "npu_moe_gating_top_k_softmax",
    """
接口原型：
torch_npu.npu_moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor, Tensor, Tensor)

功能描述
MoE计算中, 对输入x做Softmax计算, 再做topk操作.

参数说明
x: Tensor类型, 必选输入, 表示待计算的输入要求是一个2D/3D的Tensor, 数据类型支持float16、bfloat16、float32, 数据格式要求为ND. 
finished: Tensor类型, 可选输入, 表示输入中需要参与计算的行, 要求是一个1D/2D的Tensor, 数据类型支持bool, shape为x[:-1], 数据格式要求为ND. 
k: Host侧的int类型, 表示topk的k值, 大小为0<k<=x的-1轴大小, k<=1024. 

输出说明
y: Tensor类型, 对x做softmax后取的topk值, 要求是一个2D/3D的Tensor, 数据类型与x需要保持一致, 其非-1轴要求与x的对应轴大小一致, 其-1轴要求其大小同k值. 数据格式要求为ND. 
expert_idx: Tensor类型, 对x做softmax后取topk值的索引, 即专家的序号. shape要求与y一致, 数据类型支持int32, 数据格式要求为ND. 
row_idx: Tensor类型, 指示每个位置对应的原始行位置, shape要求与y一致, 数据类型支持int32, 数据格式要求为ND. 

约束说明
该接口支持推理场景下使用. 
该接口支持图模式. 

支持的PyTorch版本
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例
单算子模式调用
import torch
import torch_npu
x = torch.rand((3, 3), dtype=torch.float32).to("npu")
finished = torch.randint(2, size=(3,), dtype=torch.bool).to("npu")
y, expert_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x, finished, k=2)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
torch_npu.npu.set_compile_mode(jit_compile=True)
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
device=torch.device(f'npu:0')
torch_npu.npu.set_device(device)
class MoeGatingTopkSoftmaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, finish, k):
        res = torch_npu.npu_moe_gating_top_k_softmax(x, finish, k)
        return res
x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
moe_gating_topk_softmax_model = MoeGatingTopkSoftmaxModel().npu()
moe_gating_topk_softmax_model = torch.compile(moe_gating_topk_softmax_model, backend=npu_backend, dynamic=True)
res = moe_gating_topk_softmax_model(x, None, 2)
print(res)
"""
)

_add_torch_npu_docstr(
    "npu_moe_gating_top_k",
    """
函数原型
npu_moe_gating_top_k(x, k, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)

## 参数说明

-   **x**（`Tensor`）：必选参数，表示待计算的输入。要求是一个2D的Tensor，数据类型支持`float16`、`bfloat16`、`float32`，数据格式要求为ND。支持非连续Tensor。最后一维的大小（即专家数）要求不大于`2048`。

-   **k**（`int`）：必选参数，表示每个token最终筛选得到的专家个数，数据类型为`int64`。要求`1 <= k <= x_shape[-1] / group_count * k_group`。

-   <strong>*</strong>：代表其之前的变量是位置相关，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
 
-   **bias**（`Tensor`）：可选参数，表示与输入`x`进行计算的bias值。要求是1D的Tensor，要求shape值与`x`的最后一维相等。数据类型支持`float16`、`bfloat16`、`float32`，数据类型需要与`x`保持一致，数据格式要求为ND。支持非连续`Tensor`。

-   **k_group**（`int`）：可选参数，表示每个token组筛选过程中，选出的专家组个数，数据类型为`int64`，默认值为`1`。要求`1 <= k_group <= group_count`，并且`k_group * x_shape[-1] / group_count`的值要大于等于`k`。

-   **group_count**（`int`）：可选参数，表示将全部专家划分的组数，数据类型为`int64`，默认值为`1`。要求`group_count > 0，x_shape[-1]`能够被`group_count`整除且整除后的结果大于`2`，并且整除的结果按照`32`个数对齐后乘`group_count`的结果不大于`2048`。

-   **group_select_mode**（`int`）：可选参数，表示一个专家组的总得分计算方式。默认值为`0`，`0`表示组内取最大值，作为专家组得分；`1`表示取组内Top2的专家进行得分累加，作为专家组得分。

-   **renorm**（`int`）：可选参数，表示renorm标记，默认值为`0`，表示先进行norm再进行topk计算。当前仅支持`0`。
-   **norm_type**（`int`）：可选参数，表示norm函数类型，`1`表示使用Sigmoid函数，`0`表示Softmax函数。默认值为`0`。

-   **out_flag**（`bool`）：可选参数，是否输出norm函数中间结果。默认值为`False`。
-   **routed_scaling_factor**（`float`）：可选参数，表示计算`yOut`使用的`routed_scaling_factor`系数，默认值为`1.0`。
-   **eps**（`float`）：可选参数，表示计算`yOut`使用的`eps`系数，默认值为`1e-20`。

返回值说明
y_out（Tensor）：表示对x做norm操作和分组排序topk后计算的结果。是一个2D的Tensor，数据类型与`x`一致。
expert_idx_out（Tensor）：表示对`x`做norm操作和分组排序topk后的索引，即专家的序号。shape与y_out一致，数据类型为`int32`。
norm_out（Tensor）：表示norm计算的输出结果。shape与`x`一致，数据类型为`float32`。

约束说明

该接口支持推理场景下使用。
该接口支持图模式。

调用示例
单算子模式调用
import torch
import torch_npu
import numpy

k = 1
k_group = 4
group_count = 8
group_select_mode = 1
renorm = 0
norm_type = 1
out_flag = False
routed_scaling_factor = 1.0
eps = 1e-20

# 生成随机数据, 并发送到npu
x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
x_tensor = torch.tensor(x).npu()
bias_tensor = torch.tensor(bias).npu()

# 调用MoeGatingTopK算子
y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)

图模式调用
# 入图方式
import torch
import torch_npu
import torchair
import numpy

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_tensor, bias_tensor):
        return torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
# 实例化模型model
model = Model().npu()
# 从TorchAir获取NPU提供的默认backend
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
# 使用TorchAir的backend去调用compile接口编译模型
model = torch.compile(model, backend=npu_backend)

k = 1
k_group = 4
group_count = 8
group_select_mode = 1
renorm = 0
norm_type = 1
out_flag = False
routed_scaling_factor = 1.0
eps = 1e-20

# 生成随机数据, 并发送到npu
x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
x_tensor = torch.tensor(x).npu()
bias_tensor = torch.tensor(bias).npu()

# 调用MoeGatingTopK算子
y_npu, expert_idx_npu, out_npu = model(x_tensor, bias_tensor)
"""
)

_add_torch_npu_docstr(
    "npu_moe_init_routing",
    """
接口原型：
torch_npu.npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num) -> (Tensor, Tensor, Tensor)

功能描述
算子功能: MoE的routing计算, 根据torch_npu.npu_moe_gating_top_k_softmax的计算结果做routing处理. 
计算公式为: 
expanded_expert_idx, sorted_rowIdx=keyValueSort(expert_idx,row_idx)
expanded_row_idx[sorted_row_idx[i]]=i
expanded_x[i]=x[sorted_row_idx[i]%num_rows]

参数说明
x: Tensor类型, 必选输入, MOE的输入即token特征输入, 要求为一个2D的Tensor, shape为 (NUM_ROWS, H). 数据类型支持float16、bfloat16、float32, 数据格式要求为ND. shape大小需要小于2^24. 
row_idx: Tensor类型, 必选输入, 指示每个位置对应的原始行位置, shape要求与expert_idx一致. 数据类型支持int32, 数据格式要求为ND. 
expert_idx: Tensor类型, 必选输入, torch_npu.npu_moe_gating_top_k_softmax的输出每一行特征对应的K个处理专家, 要求是一个2D的shape (NUM_ROWS, K), 数据类型支持int32, 数据格式要求为ND. 
active_num: int类型, 表示总的最大处理row数, 输出expanded_x只有这么多行是有效的. 

输出说明
expanded_x: Tensor类型, 根据expert_idx进行扩展过的特征, 要求是一个2D的Tensor, shape (min(NUM_ROWS, activeNum) * k, H). 数据类型同x, 数据格式要求为ND. 
expanded_row_idx: Tensor类型, expanded_x和x的映射关系,  要求是一个1D的Tensor, Shape为(NUM_ROWS*K, ), 数据类型支持int32, 数据格式要求为ND. 
expanded_expert_idx: Tensor类型, 输出expert_idx排序后的结果. 

约束说明
该接口支持推理场景下使用. 
该接口支持图模式. 

支持的PyTorch版本
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例
单算子模式调用
import torch
import torch_npu
x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
active_num = 3
expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
torch_npu.npu.set_compile_mode(jit_compile=True)
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

device=torch.device(f'npu:0')

torch_npu.npu.set_device(device)

class MoeInitRoutingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, row_idx, expert_idx, active_num):
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num=active_num)
        return expanded_x, expanded_row_idx, expanded_expert_idx

x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
active_num = 3

moe_init_routing_model = MoeInitRoutingModel().npu()
moe_init_routing_model = torch.compile(moe_init_routing_model, backend=npu_backend, dynamic=True)
expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing_model(x, row_idx, expert_idx, active_num=active_num)
print(expanded_x)
print(expanded_row_idx)
print(expanded_expert_idx)
"""
)

_add_torch_npu_docstr(
    "npu_moe_init_routing_v2",
    """
算子功能：MoE（Mixture of Expert）的routing计算，根据3.28 torch_npu.npu_moe_gating_top_k_softmax的计算结果做routing处理，支持不量化和动态量化模式。

接口原型
torch_npu.npu_moe_init_routing_v2(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
参数说明
    x：Tensor类型，表示MoE的输入即token特征输入，要求为2D的Tensor，shape为(NUM_ROWS, H)，H代表每个Token的长度。数据类型支持float16、bfloat16、float32、int8，数据格式要求为ND。
    expert_idx：Tensor类型，表示torch_npu.npu_moe_gating_top_k_softmax输出每一行特征对应的K个处理专家，要求是2D的Tensor，shape为(NUM_ROWS, K)，且专家id不能超过专家数。数据类型支持int32，数据格式要求为ND。
    scale：Tensor类型，可选参数，用于计算量化结果的参数。数据类型支持float32，数据格式要求为ND。如果不输入表示计算时不使用scale，且输出expanded_scale中的值未定义。
        非量化场景下，如果输入则要求为1D的Tensor，shape为(NUM_ROWS,)。
        动态quant场景下，如果输入则要求为2D的Tensor，shape为(expert_end-expert_start, H)。
    offset：Tensor类型，可选参数，用于计算量化结果的偏移值。数据类型支持float32，数据格式要求为ND。
        在非量化场景下不输入。
        动态quant场景下不输入。
    active_num：int类型，表示总的最大处理row数，输出expanded_x只有这么多行是有效的，当前入参校验需大于等于0。当前未使用，校验需等于NUM_ROWS*K。
    expert_capacity：int类型，表示每个专家能够处理的tokens数，取值范围大于等于0。当前未使用，仅校验非空。
    expert_num：int类型，表示专家数。expert_tokens_num_type为key_value模式时，取值范围为[0, 5120]；其他模式取值范围为[0, 10240]。
    drop_pad_mode：int类型，表示是否为drop_pad场景，取值为0和1。0表示dropless场景，该场景下不校验expert_capacity。1表示drop_pad场景。当前仅支持0。
    expert_tokens_num_type：int类型，取值为0、1和2。0表示cumsum模式；1表示count模式，即输出的值为各个专家处理的token数量的累计值；2表示key_value模式，即输出的值为专家和对应专家处理token数量的累计值 。当前仅支持1和2。
    expert_tokens_num_flag：bool类型，表示是否输出expert_token_cumsum_or_count，默认False表示不输出。当前仅支持True。
    quant_mode：int类型，表示量化模式，支持取值为0、1、-1。0表示静态量化，-1表示不量化场景；1表示动态quant场景。当前仅支持-1和1。x数据类型为int8时仅支持-1，不可再量化。
    active_expert_range：int类型长度为2的数组，表示活跃expert的范围。数组内值为[expert_start, expert_end]，表示活跃的expert范围在expert_start到expert_end之间，左闭右开。要求值大于等于0，并且expert_end不大于expert_num。
    row_idx_type：int类型，表示输出expanded_row_idx使用的索引类型，支持取值0和1，默认值0。0表示gather类型的索引；1表示scatter类型的索引。性能模板下仅支持1。
输出说明
    expanded_x：Tensor类型，根据expert_idx进行扩展过的特征，要求是2D的Tensor，shape为(NUM_ROWS*K, H)。非量化场景下数据类型同x；量化场景下数据类型支持int8。数据格式要求为ND。前available_idx_num*H个元素为有效数据，其余由row_idx_type决定。其中available_idx_num为expert_idx中active_expert_range范围的元素的个数。量化场景下，当x的数据类型为int8时，输出值未定义。
    expanded_row_idx：Tensor类型，expanded_x和x的映射关系， 要求是1D的Tensor，shape为(NUM_ROWS*K, )，数据类型支持int32，数据格式要求为ND。前available_idx_num个元素为有效数据，其余无效数据由row_idx_type决定，其中available_idx_num为expert_idx中active_expert_range范围的元素的个数。row_idx_type为0时，无效数据由-1填充；row_idx_type为1时，无效数据未初始化。
    expert_token_cumsum_or_count：Tensor类型。在expert_tokens_num_type为1的场景下，要求是1D的Tensor，表示active_expert_range范围内expert对应的处理token的总数。shape为(expert_end-expert_start, )；在expert_tokens_num_type为2的场景下，要求是2D的Tensor，shape为(expert_num, 2)，表示active_expert_range范围内token总数为非0的expert，以及对应expert处理token的总数；expert id在active_expert_range范围且剔除对应expert处理token为0的元素对为有效元素对，存放于Tensor头部并保持原序。数据类型支持int64，数据格式要求为ND。
    expanded_scale：Tensor类型，数据类型支持float32，数据格式要求为ND。令available_idx_num为active_expert_range范围的元素的个数。
        非量化场景下，即quant_mode为-1，shape为(NUM_ROWS*K, )。当scale未输入时，输出值未定义；当scale输入时，输出表示一个1D的Tensor，前available_idx_num*H个元素为有效数据，其余为无效数据。
        动态quant场景下，即quant_mode为1，输出量化计算过程中scale的中间值，shape为(NUM_ROWS*K)。当scale未输入时，输出值未定义；当scale输入时，输出表示一个1D的Tensor，前available_idx_num个元素为有效数据，其余为无效数据，若x的输入类型为int8，输出值未定义。
约束说明
    该接口支持推理场景下使用。
    该接口支持图模式。
    不支持静态量化模式。
    该算子支持两种性能模板，进入两种性能模板需要分别额外满足以下条件，不满足条件则进入通用模板：
    进入低时延性能模板需要同时满足以下条件：
        x、expert_idx、scale输入Shape要求分别为：(1, 7168)、(1, 8)、(256, 7168)
        x数据类型要求：bfloat16
        属性要求：active_expert_range=[0,256]、 quant_mode=1、expert_tokens_num_type=2、expert_num=256
    进入大batch性能模板需要同时满足以下条件：
        NUM_ROWS范围为[384, 8192]
        K=8
        expert_num=256
        expert_end-expert_start<=32
        quant_mode=-1
        row_idx_type=1
        expert_tokens_num_type=1

支持的PyTorch版本
PyTorch 2.6
PyTorch 2.5
PyTorch 2.3
PyTorch 2.1
支持的型号
    Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
    Atlas A3 训练系列产品/Atlas A3 推理系列产品
调用示例
    单算子模式调用
import torch 
import torch_npu 
 
bs = 1 
h = 613 
k = 475 
active_num = 475 
expert_capacity = -1 
expert_num = 226 
drop_pad_mode = 0 
expert_tokens_num_type = 1 
expert_tokens_num_flag = True 
quant_mode = -1 
active_expert_range = [23, 35] 
row_idx_type = 0 
 
x = torch.randn((bs, h), dtype=torch.float32).npu() 
expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu() 
scale = torch.randn((bs,), dtype=torch.float32).npu() 
offset = None 
 
expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch_npu.npu_moe_init_routing_v2( 
                x, expert_idx, scale=scale, offset=offset, 
                active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode,  
                expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag, 
                active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    图模式调用
import torch 
import torch.nn as nn 
import torch_npu 
import torchair as tng 
from torchair.configs.compiler_config import CompilerConfig 
 
config = CompilerConfig() 
npu_backend = tng.get_npu_backend(compiler_config=config) 
 
class MoeInitRoutingV2Model(nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1, 
                expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False, 
                quant_mode=0, active_expert_range=0, row_idx_type=0): 
        return torch.ops.npu.npu_moe_init_routing_v2(x, expert_idx, scale=scale, offset=offset, 
                active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode,  
                expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag, 
                active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type) 
 
def main(): 
    bs = 1 
    h = 613 
    k = 475 
 
    active_num = 475 
    expert_capacity = -1 
    expert_num = 226 
    drop_pad_mode = 0 
    expert_tokens_num_type = 1 
    expert_tokens_num_flag = True 
    quant_mode = -1 
    active_expert_range = [23, 35] 
    row_idx_type = 0 
 
    x = torch.randn((bs, h), dtype=torch.float32).npu() 
    expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu() 
    scale = torch.randn((bs,), dtype=torch.float32).npu() 
    offset = None 
 
    moe_init_routing_v2_model = MoeInitRoutingV2Model().npu() 
    moe_init_routing_v2_model = torch.compile(moe_init_routing_v2_model, backend=npu_backend, dynamic=False) 
    expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = moe_init_routing_v2_model(x, 
                                    expert_idx, scale=scale, offset=offset, active_num=active_num, 
                                    expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode,  
                                    expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag, 
                                    active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type) 
 
if __name__ == '__main__': 
    main()
"""
)


_add_torch_npu_docstr(
    "npu_moe_token_permute_with_routing_map",
    """
接口原型：
torch_npu.npu_moe_token_permute_with_routing_map(Tensor tokens, Tensor routing_map, *, Tensor? probs=None, int? num_out_tokens=None, bool drop_and_pad=False) -> (Tensor, Tensor, Tensor)

算子功能：
MoE的permute计算，将token和expert的标签作为routingMap传入，根据routing_map将tokens和可选probs广播后排序

接口说明：
输入：
tokens（Tensor，计算输入）：输入toke，要求为一个维度为2D的Tensor，shape为 (tokens_num, hidden_size)，数据类型支持BFLOAT16，FLOAT16，FLOAT，数据格式要求为ND。支持非连续的Tensor。
routing_map（Tensor ，计算输入）：表token到expert的映射关系，要求shape为一个2D的（tokens_num，experts_num），数据类型支持INT8、BOOL。当数据类型为INT8，取值支持0、1，当数据类型为bool，取值支持true、false，数据格式要求为ND。支持非连续的Tensor。非droppad模式要求每行中包含topK个true 或 1。
probs（Tensor，计算输入）：可选输入probs，关键字参数，默认值为None，要求元素个数与routing_map相同,当probs为None时，可选输出permute_probs_out_optional为空，数据类型同tokens。数据格式要求为ND。支持非连续的Tensor。
num_out_tokens（int64_t，计算输入）：可选输入，默认值为token_num, 用于计算topK 和capacity 的有效输出token数。
drop_and_pad（bool，计算输入）：可选输入，默认值为False，表示是否开启drop_and_pad模式。
输出：
permuted_tokens_out（Tensor，第一个输出）：根据indices进行扩展并排序筛选过的tokens，要求是一个2D的Tensor，shape为(outToken , hidden_size)。数据类型同tokens，数据格式要求为ND。支持非连续的Tensor。
permute_probs_out_optional（Tensor，第二个输出）：根据indices进行排序并筛选过的probs，Shape为(outToken)，数据类型同probsOptional，数据格式要求为ND。支持非连续的Tensor。
sorted_indices_out（Tensor，第三个输出）：permute_tokens和tokens的映射关系， 要求是一个1D的Tensor，Shape为(outToken)，数据类型支持INT32，数据格式要求为ND。支持非连续的Tensor。

输入约束：
1. 由于float无损转int限制，tokens_num和experts_num要求小于16777215。
2. 由于UB限制，routing_map 中 每行为1或true的个数固定且小于512，num_out_tokens/num_tokens小于512。
3. drop_and_pad为False场景，num_out_tokens / num_tokens需和routing_map中每行1或True的个数一致。

调用示例：
import torch,torch_npu

x = torch.randn((3, 4), dtype=torch.float).npu()
rounting_map = torch.tensor(
    [[True, True], [True, True], [True, True]], dtype=torch.bool).npu()
numtoken = 6
pad_mode = False

permuted_tokens_out, _, sorted_indices_out = torch_npu.npu_moe_token_permute_with_routing_map(x, rounting_map, num_out_tokens=numtoken, drop_and_pad=pad_mode)
    """
)


_add_torch_npu_docstr(
    "npu_prefetch",
    """
接口原型：
torch_npu.npu_prefetch(Tensor input, Tensor? dependency, int max_size, int offset=0) -> ()

功能描述
提供网络weight预取功能, 将需要预取的权重搬到L2 Cache中. 尤其在做较大Tensor的MatMul计算且需要搬移到L2 Cache的操作时, 可通过该接口提前预取权重, 适当提高模型性能, 具体效果基于用户对并行的处理. 

参数说明
input: Tensor类型, 表示需要预取的权重, 不做数据处理, 与数据类型和数据格式无关; 输入不能含有为None. 
dependency: Tensor类型, 表示开始预取的节点, 单算子下不生效可为None, 图模式下不可为None; 不做数据处理, 与数据类型和数据格式无关. 
max_size: int类型, 取值需大于0, 表示权重预取的最大size, 超过预取权重的size时, 会设置为权重的最大size. 数据类型为int32、int64. 
offset: int类型, 默认值0, 取值大于等于0, 表示权重预取内存地址偏移, 不允许超过权重地址范围. 数据类型为int32、int64. 

输出说明
无

约束说明
该接口支持图模式. 

支持的PyTorch版本
Pytorch 2.5
PyTorch 2.4
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品

调用示例:
单算子多流并发调用
import torch
import torch_npu
s_cmo = torch.npu.Stream()
x = torch.randn(10000, 10000, dtype=torch.float32).npu()
y = torch.randn(10000, 1, dtype=torch.float32).npu()
add = torch.add(x, 1)
with torch.npu.stream(s_cmo):
    torch_npu.npu_prefetch(y, None, 10000000)
abs = torch.abs(add)
mul = torch.matmul(abs, abs)
out = torch.matmul(mul, y)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.debug.graph_dump.type = 'pbtxt'
npu_backend = tng.get_npu_backend(compiler_config=config)
x = torch.randn(10000, 10000, dtype=torch.float32).npu()
y = torch.randn(10000, 1, dtype=torch.float32).npu()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        add = torch.add(x, 1)
        torch_npu.npu_prefetch(y, add, 10000000)
        abs = torch.abs(add)
        mul = torch.matmul(abs, abs)
        out = torch.matmul(mul, y)
        return out

npu_model = Model().npu()
model = torch.compile(npu_model, backend=npu_backend, dynamic=False, fullgraph=True)
output = model(x,y)
"""
)

_add_torch_npu_docstr(
    "npu_quantize",
    """
接口原型：
torch_npu.npu_quantize(Tensor input, Tensor scales, Tensor? zero_points, ScalarType dtype, int axis=1, bool div_mode=True) -> Tensor

功能描述
算子功能: 对输入的张量进行量化处理. 
计算公式: 
如果div_mode为True: result=(input/scales)+zero_points
如果div_mode为False: result=(input*scales)+zero_points

参数说明
input: Tensor类型, 需要进行量化的源数据张量, 数据格式支持ND、NZ, 支持非连续的Tensor. div_mode为False且dtype为torch.quint4x2时, 最后一维需要能被8整除. 
Atlas 推理系列产品: 数据类型支持float、float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float、float16、bfloat16. 
scales: Tensor类型, 对input进行scales的张量, 必选输入: 
div_mode为True时
Atlas 推理系列产品: 数据类型支持float. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float、bfloat16. 
div_mode为False时, 数据格式支持ND, 支持非连续的Tensor. 支持1维或多维(1维时, 对应轴的大小需要与input中第axis维相等或等于1; 多维时, scales的shape需要与input的shape维度相等, 除axis指定的维度, 其他维度为1, axis指定的维度必须和input对应的维度相等或等于1). 
Atlas 推理系列产品: 数据类型支持float、float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float、float16、bfloat16. 
zero_points: Tensor类型, 对input进行offset的张量, 可选输入. 
div_mode为True时
Atlas 推理系列产品: 数据类型支持int8、uint8、int32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、uint8、int32、bfloat16. 
div_mode为False时, 数据格式支持ND, 支持非连续的Tensor. 支持1维或多维(1维时, 对应轴的大小需要与input中第axis维相等或等于1; 多维时, scales的shape需要与input维度相等, 除axis指定的维度, 其他维度为1, axis指定的维度必须和input对应的维度相等). zero_points的shape和dtype需要和scales一致. 
Atlas 推理系列产品: 数据类型支持float、float16. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持float、float16、bfloat16. 
dtype: ScalarType类型int类型, 指定输出参数的类型. 
div_mode为True时, 
Atlas 推理系列产品: 类型支持torch.qint8、torch.quint8、torch.int32. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 类型支持torch.qint8、torch.quint8、torch.int32. 
div_mode为False时, 类型支持torch.qint8、torch.quint4x2. 如果dtype为torch.quint4x2时, 输出tensor类型为int32, 由8个int4拼接. 
axis: int类型, 量化的elemwise轴,  其他的轴做broadcast, 默认值为1. 
div_mode为False时, axis取值范围是[-2, +∞)且指定的轴不能超过输入input的维度数. 如果axis=-2, 代表量化的elemwise轴是输入input的倒数第二根轴; 如果axis大于-2, 量化的elemwise轴是输入的最后一根轴. 
div_mode: 布尔类型, 表示计算scales模式. 当div_mode为True时, 表示用除法计算scales; div_mode为False时, 表示用乘法计算scales, 默认值为True. 

输出说明
y: Tensor类型, 公式中的输出, 输出大小与input一致. 数据类型由参数dtype指定, 如果参数dtype为torch.quint4x2, 输出的dtype是torch.int32, shape的最后一维是输入shape最后一维的1/8, shape其他维度和输入一致. 

约束说明
该接口支持推理场景下使用. 
该接口支持图模式. 
input数据格式为NZ时, input输入shape支持3维, 形如(e, k, n), scales输入shape支持1维, zero_points输入为None, dtype为quint4x2. 
div_mode为False时: 
支持Atlas A2 训练系列产品/Atlas 800I A2 推理产品. 
当dtype为torch.quint4x2或者axis为-2时, 不支持Atlas 推理系列产品. 

支持的PyTorch版本
PyTorch 2.4
PyTorch 2.3
PyTorch 2.1

支持的型号
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas 推理系列产品

调用示例:
单算子模式调用
import torch
import torch_npu
x = torch.randn((2, 3, 12), dtype=torch.float).npu()
scale = torch.tensor(([3] * 12),dtype=torch.float).npu()
out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
print(out)
图模式调用
import torch
import torch_npu
import torchair as tng
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
x = torch.randn((2, 3, 12), dtype=torch.float16).npu()
scale = torch.tensor(([3] * 12),dtype=torch.float16).npu()
axis =1
div_mode = False

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
    def forward(self, x, scale,zero_points, dst_type,div_mode):
        return torch_npu.npu_quantize(x, scale, zero_points=zero_points, dtype=dst_type, div_mode=div_mode)
model = Network()
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)
config.debug.graph_dump.type = 'pbtxt'
model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
output_data = model(x, scale,None,dst_type=torch.qint8, div_mode=div_mode)
print(output_data)
"""
)

_add_torch_npu_docstr(
    "npu_kronecker_quant",
    """
接口原型：
npu_kronecker_quant(Tensor x, Tensor kronecker_p1, Tensor kronecker_p2, float? clip_ratio=None, ScalarType? dst_dtype=None) -> (Tensor out, Tensor quant_scale)

功能描述
为矩阵x依次进行两次小矩阵乘法，然后针对矩阵乘的结果进行量化处理。

参数说明
x: Device侧的Tensor类型，表示输入；数据类型支持FLOAT16、BFLOAT16类型；shape为[K, M, N]，其中N必须为8的整数倍。
kronecker_p1: Device侧的Tensor类型，表示输入；数据类型支持FLOAT16、BFLOAT16类型，数据类型与x一致；shape为[M, M]，M与x第一维相同。
kronecker_p2: Device侧的Tensor类型，表示输入；数据类型支持FLOAT16、BFLOAT16类型，数据类型与x一致；shape为[N, N]，N与x第二维相同。
clip_ratio: float类型，可选参数，数据范围为(0, 1]，默认值为1。
dst_dtype：ScalarType类型，可选参数，输入值允许为torch.int32，默认值为torch.int32。

输出说明
out：Device侧的Tensor类型，表示量化输出；数据类型支持INT32；shape为[K, M, N/8]，第零维和第一维与x一致，第二维是x的1/8。
quant_scale: Device侧的Tensor类型，表示量化缩放系数；数据类型支持FLOAT32；shape为[K]，K与x第零维相同。

约束说明
输入数据类型仅支持float16和bfloat16，x、kronecker_p1和kronecker_p2数据类型要保持一致。

支持的型号
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu
x = torch.rand((16, 3, 64), dtype=torch.bfloat16).npu()
p1 = torch.rand((3, 3), dtype=torch.bfloat16).npu()
p2 = torch.rand((64, 64), dtype=torch.bfloat16).npu()
out, quant_scale = torch_npu.npu_kronecker_quant(x, p1, p2, 0.7848)
"""
)

_add_torch_npu_docstr(
    "scatter_update",
    """
接口原型：
torch_npu.scatter_update(Tensor data, Tensor indices, Tensor updates, int axis) -> Tensor

功能描述
将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值, 并将结果保存到输出tensor, data本身的数据不变. 

参数说明
data: Tensor类型, data只支持2-8维, 且维度大小需要与updates一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 
indices: Tensor类型, 数据类型支持int32、int64; 目前仅支持一维跟二维; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
updates: Tensor类型, updates的维度大小需要与data一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 
axis: 整型, 用来scatter的维度, 数据类型为int64. 

输出说明
out: Tensor类型, 计算输出, out只支持2-8维, 且维度大小需要与data一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor.
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32.
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32.
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 

约束说明
data与updates的秩一致. 
不支持索引越界, 索引越界不校验. 

支持的PyTorch版本
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号
Atlas 训练系列产品
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用: 
import torch
import torch_npu
import numpy as np
data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype=torch.float32).npu()
indices = torch.tensor ([1],dtype=torch.int64).npu()
updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]] , dtype=torch.float32).npu()
out = torch_npu.scatter_update(data, indices, updates, axis=-2)
"""
)

_add_torch_npu_docstr(
    "scatter_update_",
    """
接口原型：
torch_npu.scatter_update_(Tensor(a!) data, Tensor indices, Tensor updates, int axis) -> Tensor(a!)

功能描述
将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值, 并将结果保存到输出tensor, data本身的数据被改变. 

参数说明
data: Tensor类型, data只支持2-8维, 且维度大小需要与updates一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 
indices: Tensor类型, 数据类型支持int32、int64; 目前仅支持一维跟二维; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
updates: Tensor类型, updates的维度大小需要与data一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor. 
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32. 
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 
axis: 整型, 用来scatter的维度, 数据类型为int64. 

输出说明
out: Tensor类型, 计算输出, 复用输入地址; out只支持2-8维, 且维度大小需要与data一致; 支持非连续的tensor; 数据格式支持ND; 不支持空Tensor.
Atlas A2 训练系列产品/Atlas 800I A2 推理产品: 数据类型支持int8、float16、float32、bfloat16、int32.
Atlas A3 训练系列产品: 数据类型支持int8、float16、float32、bfloat16、int32.
Atlas 训练系列产品: 数据类型支持int8、float16、float32、int32. 

约束说明
data与updates的秩一致. 
不支持索引越界, 索引越界不校验. 

支持的PyTorch版本
PyTorch 2.3
PyTorch 2.2
PyTorch 2.1
PyTorch 1.11.0

支持的型号
Atlas 训练系列产品
Atlas A2 训练系列产品/Atlas 800I A2 推理产品
Atlas A3 训练系列产品

调用示例:
单算子模式调用: 
import torch
import torch_npu
import numpy as np
data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype=torch.float32).npu()
indices = torch.tensor ([1],dtype=torch.int64).npu()
updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]] , dtype=torch.float32).npu()
out = torch_npu.scatter_update_(data, indices, updates, axis=-2)
"""
)

_add_torch_npu_docstr(
    "npu_group_norm_swish",
    """
接口原型：
npu_group_norm_swish(Tensor input, int num_groups, Tensor weight, Tensor bias, float? eps=1e-5, float? swish_scale=1.0) -> (Tensor, Tensor, Tensor)

功能描述
计算输入input的组归一化结果，并进行swish计算。

参数说明
input: Device侧的Tensor类型，计算输入；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；input只支持2-8维；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
num_groups:int类型, 计算输入；表示将input的第1维分为num_groups组，inpu的第1维必须能被num_groups整除。
weight: Device侧的Tensor类型，计算输入；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，并且与input一致；input只支持1维，且第0维大小与input的第1维大小相同；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
bias: Device侧的Tensor类型，计算输入；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，并且与input一致；input只支持1维，且第0维大小与input的第1维大小相同；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
eps: Float类型，可选；用于防止产生除0操作；默认值为1e-5。
swish_scale: Float类型，可选; 用于计算swish；默认值为1.0。

输出说明
out：Device侧的Tensor类型，计算输出；表示将输入组归一化的结果；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型。
mean: Device侧的Tensor类型，计算输出；表述将输入分组后的均值；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型，。
rstd: Device侧的Tensor类型，计算输出；表述将输入分组后的标准差的倒数；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型。

约束说明
BFLOAT16数据类型仅支持如下产品型号：Atlas A2训练系列产品/Atlas 800I A2推理产品

支持的型号
Atlas A2训练系列产品/Atlas 800I A2推理产品

调用示例:
import torch
import torch_npu

input = torch.randn(3, 4, 6, dtype=torch.float32).npu()
weight = torch.randn(input.size(1), dtype=torch.float32).npu()
bias = torch.randn(input.size(1), dtype=torch.float32).npu()
num_groups = input.size(1)
swish_scale = 1.0
eps = 1e-5
out = torch_npu.npu_group_norm_swish(input, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
"""
)

_add_torch_npu_docstr(
    "npu_dequant_swiglu_quant",
    """
功能描述:
-   对输入张量 x 进行反量化、Swiglu 激活计算及量化，输出量化后的结果 y 和量化 scale。
-   支持静态量化和动态量化两种模式；支持 per-token 激活 scale、bias、平滑量化系数、分组计算和 Swiglu 变种计算。

接口原型:
torch_npu.npu_dequant_swiglu_quant(
    Tensor x,
    *,
    Tensor weight_scale=None,
    Tensor? activation_scale=None,
    Tensor? bias=None,
    Tensor? quant_scale=None,
    Tensor? quant_offset=None,
    Tensor? group_index=None,
    bool activate_left=False,
    int quant_mode=0,
    int swiglu_mode=0,
    float clamp_limit=7.0,
    float glu_alpha=1.702,
    float glu_bias=1.0
) -> (Tensor y, Tensor scale)

>Tensor中shape使用的变量说明：
>-   TokensNum：表示传输的Tokens数，取值≥0。
>-   H：表示嵌入向量的长度，取值\>0。
>-   groupNum：表示group\_index输入的长度，取值\>0。

-   x：Tensor类型，表示目标张量。要求是2D的Tensor，shape为\[TokensNum, 2H\]，尾轴为偶数。数据类型支持int32和bfloat16，数据格式为ND。
-   weight\_scale：Tensor类型，可选参数，表示权重量化对应的反量化系数。要求是2D的Tensor，shape为\[groupNum, 2H\]，数据类型支持float32，数据格式为ND。当x为int32时，要求该参数非None，表示需要做反量化。
-   activation\_scale：Tensor类型，可选参数，表示per-token权重量化对应的反量化系数。要求是1D的Tensor，shape为\[TokensNum\]，数据类型支持float32，数据格式为ND。当x为int32时，要求该参数非None，表示需要做反量化。
-   bias：Tensor类型，可选参数，表示x的偏置变量。数据类型支持int32，数据格式为ND。group\_index场景下（非None），该参数不生效为None。
-   quant\_scale：Tensor类型，可选参数，表示smooth量化系数。要求是2D的Tensor，shape为\[groupNum, H\]，数据类型支持float32、float16和bfloat16，数据格式为ND。
-   quant\_offset：Tensor类型，可选参数，表示量化中的偏移项。数据类型支持float32、float16和bfloat16，数据格式为ND。group\_index场景下（非None），该参数不生效为None。
-   group\_index：Tensor类型，可选参数，当前只支持count模式，表示该模式下指定分组的Tokens数（要求非负整数）。要求是1D的Tensor，数据类型支持int64，数据格式ND。
-   activate_left（bool）：可选参数，是否进行左激活，默认 False。
    -   取True时，out=swish\(split\[x, -1, 2\]\[0\]\)\*split\[x, -1, 2\]\[1\]
    -   取False时，out=swish\(split\[x, -1, 2\]\[1\]\)\*split\[x, -1, 2\]\[0\]

-   quant_mode（int）：可选参数，量化模式，0 表示静态量化，1 表示动态量化。group_index 场景下必须取 1。
-   swiglu_mode（int）：可选参数，swiglu 计算模式，0 表示传统 swiglu，1 表示变种 swiglu（支持 clamp、alpha、bias）。
-   clamp_limit（float）：可选参数，swiglu 输入门限，默认 7.0。
-   glu_alpha（float）：可选参数，glu 激活函数系数，默认 1.702。
-   glu_bias（float）：可选参数，swiglu 计算中的偏差，默认 1.0。

输出说明:
-   y：Tensor类型，表示量化后的输出tensor。要求是2D的Tensor，shape=\[TokensNum, H\]，数据类型支持int8，数据格式为ND。
-   scale：Tensor类型，表示量化的scale参数。要求是1D的Tensor，shape=\[TokensNum\]，数据类型支持float32，数据格式为ND。

约束说明:
-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   group\_index场景下（非None）约束说明：
    -   group\_index只支持count模式，需要网络保证group\_index输入的求和不超过x的TokensNum维度，否则会出现越界访问。
    -   H轴有维度大小限制：H≤10496同时64对齐场景；规格不满足场景会进行校验。
    -   输出y和scale超过group\_index总和的部分未进行清理处理，该部分内存为垃圾数据，可能会存在inf/nan异常值，网络使用的时候需要注意影响。
-   当 x 为 int32 时，必须提供 weight_scale。
-   当 x 为 float16 或 bfloat16 时，weight_scale、activation_scale、bias 必须为 None。
-   x 的最后一维长度必须为偶数。
-   当激活维度不是 x 的最后一维时，group_index 必须为 None。
-   当 group_index 非 None 时，仅支持动态量化（quant_mode=1），且 bias、quant_offset 必须为 None。
-   y 的类型仅支持 int8。
-   clamp_limit、glu_alpha、glu_bias 仅在 swiglu_mode=1 时生效。

支持的芯片型号:
-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

调用示例:

-   单算子模式调用
    import os
    import shutil
    import unittest
    
    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices

    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            y, scale = torch_npu.npu_dequant_swiglu_quant(
                x.npu(),
                weight_scale=weight_scale.npu(),
                activation_scale=activation_scale.npu(),
                bias=None,
                quant_scale=quant_scale.npu(),
                quant_offset=None,
                group_index=group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )

    if __name__ == "__main__":
        run_tests()

-   图模式调用

    ```python
    import os
    import shutil
    import unittest
    
    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices
    from torchair.configs.compiler_config import CompilerConfig
    import torchair as tng
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            y, scale = torch_npu.npu_dequant_swiglu_quant(
                x.npu(),
                weight_scale=weight_scale.npu(),
                activation_scale=activation_scale.npu(),
                bias=None,
                quant_scale=quant_scale.npu(),
                quant_offset=None,
                group_index=group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )
    
    if __name__ == "__main__":
        run_tests()

"""
)

_add_torch_npu_docstr(
    "npu_clipped_swiglu",
    """
接口原型:
torch_npu.npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True) -> Tensor

功能描述:
新增带截断的Swish门控线性单元激活函数，实现x的变体SwiGlu计算。

计算公式:
(1)将x基于输入参数dim进行合轴，合轴后维度为[pre, cut, after]。其中cut轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；pre，after 可以等于1。此外，由于after轴的元素为连续存放，且计算操作为逐元素的，因此将cut轴与after轴合并，得到x的维度为[pre, cut * after]。
(2)根据输入参数group_index, 对x的pre轴进行过滤处理，公式如下：sum = Sum(group_index), x = x[ : sum, : ]。其中sum表示group_index的所有元素之和。当不输入 group_index 时，跳过该步骤。
(3)根据输入参数interleaved，对x进行切分，公式如下：当 interleaved 为 true 时，表示奇偶切分：A = x[ : , : : 2], B = x[ : , 1 : : 2]
                                                  当 interleaved 为 false 时，表示前后切分：h = x.shape[1] // 2, A = x[ : , : h], B = x[ : , h : ]
(4)根据输入参数 alpha、limit、bias 进行变体SwiGlu计算，公式如下：A = A.clamp(min=None, max=limit), B = B.clamp(min=-limit, max=limit)
                                                                y_glu = A * sigmoid(alpha * A)
                                                                y = y_glu * (B + bias)
(5)重塑输出张量y的维度数量与合轴前的x的维度数量一致，dim轴上的大小为x的一半，其他维度与x相同。

参数说明:
x：Tensor类型，必选参数，表示目标张量。数据类型支持float16、bfloat16、float32，不支持非连续的Tensor，数据格式为ND，x的维数必须大于1维，dim轴为偶数。
group_index：Tensor类型，可选参数，表示对x进行分组的情况。要求为1维张量，第i个元素代表第i组需要处理的x合轴后的token数量，数据类型支持int64，数据格式ND。
dim： int类型，可选参数，表示需要对x进行切分的维度序号，取值范围为[-x.dim(), x.dim()-1]，默认 -1。
alpha：float类型，可选参数，表示glu激活函数系数，默认 1.702。
limit：float类型，可选参数，表示变体swiglu输入门限，默认 7.0。
bias：float类型，可选参数，表示变体swiglu计算中的偏差，默认 1.0。
interleaved： bool类型，可选参数，表示输入x是否按奇偶方式切分，true表示为奇偶方式切分，false表示为前后方式切分，默认为true。

输出说明:
y：Tensor类型，表示激活函数的输出，数据类型同输入x，在维度上，第dim维是输入x的1/2，其余维度与输入x相同，数据格式为ND。

约束说明:
无

支持的型号:
Atlas A2 推理系列产品
Atlas A3 推理系列产品

调用示例:
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestNPUClippedSwiglu(TestCase):
    def test_npu_clipped_swiglu(self, device="npu"):
        tokens_num = 4608
        hidden_size = 2048
        x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
        group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
        y = torch_npu.npu_clipped_swiglu(
            x.npu(),
            group_index=group_index.npu(),
            dim=-1,
            alpha=1.702,
            limit=7.0,
            bias=1.0,
            interleaved=True
        )

if __name__ == "__main__":
    run_tests()

"""
)

_add_torch_npu_docstr(
    "npu_cross_entropy_loss",
    """
接口原型:
torch_npu.npu_cross_entropy_loss(Tensor input, Tensor target, Tensor? weight=None, str reduction="mean", int ignore_index=-100, float label_smoothing=0.0, float lse_square_scale_for_zloss=0.0, bool return_zloss=False) -> (Tensor, Tensor, Tensor, Tensor)

功能描述:
将原生CrossEntropyLoss中的log_softmax和nll_loss融合，降低计算时使用的内存。接口允许计算zloss。

参数说明:
input: Device侧的Tensor类型，表示输入；数据类型支持FLOAT16、FLOAT32、BFLOAT16类型；shape为[N, C]，N为批处理大小，C为标签数，必须大于0。
target: Device侧的Tensor类型，表示标签；数据类型支持INT64类型；shape为[N]，与input第零维相同，取值范围[0, C)。
weight: Device侧的Tensor类型，表示每个类别指定的缩放权重，可选；数据类型支持FLOAT32类型；shape为[C]，与input第一维相同，取值范围(0, 1]，不指定值时默认全一。
reduction: str类型，表示loss的归约方式；支持范围["mean", "sum", "none"]，默认为"mean"。
ignore_index: int类型，指定忽略的标签；数值必须小于C，当小于0时视为无忽略标签；默认值为-100。
label_smoothing: float类型，表示计算loss时的平滑量；取值范围[0.0, 1.0)；默认值为0.0。
lse_square_scale_for_zloss: float类型，表示计算zloss所需要的scale；取值范围[0.0, 1.0)；默认值为0.0；当前暂不支持。
return_zloss: bool类型，控制是否返回zloss；设置为True将返回zloss，设置为False时不返回zloss；默认值为False；当前暂不支持。

输出说明:
loss：Device侧的Tensor类型，表示输出损失；数据类型与input相同；reduction为"none"时shape为[N]，与input第零维一致，否则shape为[1]。
log_prob: Device侧的Tensor类型，输出给反向计算的输出；数据类型与input相同；shape为[N, C]，与input一致。
zloss: Device侧的Tensor类型，表示辅助损失；数据类型与input相同；shape与loss一致；当return_zloss为True时输出zloss，否则将返回空tensor；当前暂不支持。
lse_for_zloss: Device侧的Tensor类型，zloss场景输出给反向计算的输出；数据类型与input相同；shape为[N]，与input第零维一致；lse_square_scale_for_zloss不为0.0时将返回该输出，否则将返回空tensor；当前暂不支持。

约束说明:
输入shape中N取值范围(0, 200000]。
当input.requires_grad=True时，sum/none模式下不支持修改label_smoothing的默认值；mean模式下不支持修改所有含默认值的入参的值，包括weight，reduction，ignor_index，label_smoothing，lse_square_scale_for_zloss和return_zloss。
属性lse_square_scale_for_zloss与return_zloss暂未使能。
输出zloss与lse_for_zloss暂未使能。
输出中仅loss和zloss支持梯度计算。

支持的型号:
Atlas A2 训练系列产品
Atlas A3 训练系列产品

调用示例:
import torch
import torch_npu
    
N = 4096
C = 8080
input = torch.randn(N, C).npu()
target = torch.arange(0, N).npu()
    
loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(input, target)
"""
)

_add_torch_npu_docstr(
    "npu_gemma_rms_norm",
    """
接口原型：
npu_gemma_rms_norm(Tensor input, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor)

功能描述
通过对数据的root mean square（RMS）进行归一化，避免均值的使用

参数说明
input: Device侧的Tensor类型，表示输入的需要归一化的数据。shape支持1-8维度，数据格式支持ND。数据类型支持FLOAT32、FLOAT16、BFLOAT16。
gamma: Device侧的Tensor类型，表示数据缩放因子；shape支持1-8维度，数据格式支持ND。shape需要满足gamma_shape = input_shape\[n:\], n < input_shape.dims()。数据类型支持FLOAT32、FLOAT16、BFLOAT16，与input数据类型保持一致。
epsilon: float数据类型，用于防止除0错误。默认值1e-06。

输出说明
y：Device侧的Tensor类型，表示归一化后的输出数据。shape支持1-8维度，数据格式支持ND。数据类型支持FLOAT32、FLOAT16、BFLOAT16，与输入input数据类型保持一致。
rstd: Device侧的Tensor类型，输入input数据的标准差；shape支持1-8维度，数据格式支持ND。数据类型支持FLOAT32、FLOAT16、BFLOAT16，与输入input数据类型保持一致。shape与输入input的shape前几维保持一致，前几维指输入input的维度减去输入gamma的维度，表示不需要norm的维度。

约束说明
不支持空进空出
不支持非连续tensor

支持的型号
Atlas A2训练系列产品/Atlas 800I A2中的推理产品
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu

input_x = torch.randn([20, 10, 64], dtype=torch.float32).npu()
input_gamma = torch.randn([64], dtype=torch.float32).npu()

y, rstd = torch_npu.npu_gemma_rms_norm(input_x, input_gamma)
"""
)

_add_torch_npu_docstr(
    "npu_add_rms_norm_dynamic_quant",
    """
接口原型：
npu_add_rms_norm_dynamic_quant(Tensor x1, Tensor x2, Tensor gamma, *, Tensor? smooth_scale1=None, Tensor? smooth_scale2=None, Tensor? beta=None, float epsilon=1e-6, bool[2] output_mask=[]) -> (Tensor, Tensor, Tensor, Tensor, Tensor)

功能描述
将RmsNorm前的Add操作、RmsNorm归一化和最多两路DynamicQuant量化进行融合，减少数据搬入搬出操作。

参数说明
x1: Tensor类型，标准化输入张量。shape支持2-8维，数据类型支持FLOAT16、BFLOAT16，格式支持ND。不支持空Tensor。
x2: Tensor类型，标准化输入张量。shape支持2-8维，数据类型支持FLOAT16、BFLOAT16，格式支持ND。不支持空Tensor。
gamma: Tensor类型，归一化权重张量。shape为1维，需与x1最后一维一致，数据类型与x1一致。不支持空Tensor。
smooth_scale1: Tensor类型，第一路量化的smooth_scale张量。可选，shape和数据类型与gamma一致。不支持空Tensor。
smooth_scale2: Tensor类型，第二路量化的smooth_scale张量。可选，shape和数据类型与gamma一致，需与smooth_scale1配合使用。不支持空Tensor。
beta: Tensor类型，归一化偏置项。可选，shape和数据类型与gamma一致。不支持空Tensor
epsilon: double类型，防止除0错误，默认值为1e-6.
output_mask: 数组，表示输出的掩码，数据类型支持BOOL，支持空指针，或长度为2的数组

输出说明
y1: Tensor类型，第一路量化输出。数据类型支持INT8。shape、数据格式需要与入参x1保持一致。支持非连续的Tensor，不支持空Tensor。
y2: Tensor类型，第二路量化输出。数据类型支持INT8。shape、数据格式需要与入参x1保持一致。支持非连续的Tensor，不支持空Tensor。若未输入smooth_scale2，此输出无实际意义。
x_out: Tensor类型，x1与x2之和。shape、数据类型与x1一致。
scale1: Tensor类型，第一路量化scale输出。shape为x1除最后一维后的shape，数据类型为float32，数据格式支持ND，支持非连续的Tensor，不支持空Tensor
scale2: Tensor类型，第二路量化scale输出，shape同scale1，数据类型为float32， 数据类型支持ND，支持非连续的Tensor，不支持空Tensor。若未输入smooth_scale2，此输出无实际意义。

约束说明
所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。
当outputMaskOptional不为空时，参数smoothScale1Optional有值时，则outputMaskOptional[0]必须为True。参数smoothScale2Optional有值时，则outputMaskOptional[1]必须为True。
当outputMaskOptional不为空时，outputMaskOptional[0]与outputMaskOptional[1]不能同时为False。
当outputMaskOptional为空时，参数smoothScale2Optional有值时，参数smoothScale1Optional也必须有值。

支持的型号
Atlas A3训练系列产品/Atlas A3推理系列产品
Atlas A2训练系列产品/Atlas 800I A2推理产品/A200I A2 Box异构组件

调用示例:
import torch
import torch_npu

x1 = torch.randn(3, 4, 8, dtype=torch.float16, device='npu')
x2 = torch.randn(3, 4, 8, dtype=torch.float16, device='npu')
gamma = torch.ones(8, dtype=torch.float16, device='npu')
beta = torch.zeros(8, dtype=torch.float16, device='npu')
smooth_scale1 = torch.ones(8, dtype=torch.float16, device='npu')
smooth_scale2 = torch.ones(8, dtype=torch.float16, device='npu')
epsilon = 1e-6
output_mask = [True, True]

y1_npu, y2_npu, x_out_npu, s1_npu, s2_npu = torch_npu.npu_add_rms_norm_dynamic_quant(
    x1_n, x2_n, gamma_n,
    smooth_scale1=s1_n,
    smooth_scale2=s2_n,
    beta=beta_n,
    epsilon=eps_f,
    output_mask=output_mask,
)
"""
)

_add_torch_npu_docstr(
    "npu_add_rms_norm_cast",
    """
接口原型：
npu_add_rms_norm_cast(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-06) -> (Tensor, Tensor, Tensor, Tensor)

功能描述
add_rms_norm和cast的融合算子，对add_rms_norm计算后的输出做指定类型的cast操作，减少搬入搬出。

参数说明
x1：Device侧的Tensor类型，需要归一化的原始数据输入。shape支持1-8维。数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。不支持空tensor。
x2：Device侧的Tensor类型，需要归一化的原始数据输入。shape支持1-8维，数据格式支持ND，数据类型支持BFLOAT16、FLOAT16。shape、数据格式、数据类型均需要与入参x1保持一致。不支持空tensor。
gamma：Device侧的Tensor类型，数据缩放因子。shape支持1-8维，数据格式支持ND，数据类型支持FLOAT16、BFLOAT16。shape需要满足gamma_shape = x_shape\[n:\], n < x_shape.dims()。数据类型、数据格式需要与入参x1保持一致。不支持空tensor。
epsilon：float数据类型，用于防止除0错误，数据类型为DOUBLE，默认值为1e-6。

输出说明
y1：Device侧的Tensor类型，归一化后经过类型转换的输出数据。shape支持1-8维，数据格式支持ND，数据类型支持FLOAT32。shape、数据格式需要与入参x1保持一致。不支持空tensor。
y2：Device侧的Tensor类型，归一化后的输出数据。shape支持1-8维，数据格式支持ND，数据类型支持BFLOAT16、FLOAT16。shape、数据格式、数据类型均需要与入参x1保持一致。不支持空tensor。
rstd：Device侧的Tensor类型，x的标准差。数据类型支持FLOAT32，shape支持1-8维。shape与入参x1的shape前几维保持一致，前几维指x1的维度减去gamma的维度，表示不需要norm的维度。数据格式支持ND，需要与入参x1的数据格式保持一致。不支持空tensor。
x：Device侧的Tensor类型，归一化的数据和。shape支持1-8维，数据格式支持ND，数据类型支持BFLOAT16、FLOAT16。shape、数据格式、数据类型均需要与入参x1保持一致。不支持空tensor。

支持的型号
Atlas A2训练系列产品/Atlas 800I A2中的推理产品
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu

input_x1 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
input_x2 = torch.randn([20, 10, 64], dtype=torch.float16).npu()
input_gamma = torch.randn([64], dtype=torch.float16).npu()

y1, y2, rstd, x = torch_npu.npu_add_rms_norm_cast(input_x1, input_x2, input_gamma)
"""
)

_add_torch_npu_docstr(
    "npu_add_rms_norm_quant",
    """
接口原型：
func: npu_add_rms_norm_quant(Tensor x1, Tensor x2, Tensor gamma, Tensor scales1, 
Tensor? zero_points1, Tensor? beta, Tensor? scales2=None, Tensor? zero_points2=None, *, int axis=-1, 
float epsilon=1e-06, bool div_mode=True) -> (Tensor, Tensor, Tensor)

功能描述
add_rms_norm_quant算子将rms_norm前的add算子以及之后的quantize算子融合，减少搬入搬出。新增偏置项beta参数

参数说明
x1：Device侧的Tensor类型，表示标准化过程中的源数据张量。shape支持1-8维。数据类型支持BFLOAT16、FLOAT16，数据格式支持ND，支持非连续的Tensor。不支持空tensor。
x2：Device侧的Tensor类型，表示标准化过程中的源数据张量。shape支持1-8维。数据类型支持BFLOAT16、FLOAT16，数据格式支持ND，支持非连续的Tensor。数据格式、数据类型均需要与入参x1保持一致。不支持空tensor。
gamma：Device侧的Tensor类型，表示标准化过程中的权重张量。shape支持1-8维，shape需要与x1需要Norm的维度一致。数据类型支持BFLOAT16、FLOAT16，数据格式支持ND，支持非连续的Tensor。数据类型需要与入参x1保持一致。不支持空tensor。
scales1：Device侧的Tensor类型，表示量化过程中得到y1进行的scales张量。shape需要与gamma保持一致。数据类型支持FLOAT32、BFLOAT16，数据格式支持ND，支持非连续的Tensor。当参数divMode的值为True时，该参数的值不能为0。
zero_points1：Device侧的Tensor类型，表示量化过程中得到y1进行的offset张量。可选参数。shape需要与gamma保持一致。数据类型支持INT32、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
beta：Device侧的Tensor类型，表示标准化过程中的偏置项。可选参数。shape支持1-8维，shape需要与gamma的shape保持一致。数据类型支持BFLOAT16、FLOAT16，数据类型需要与gamma保持一致，数据格式支持ND，支持非连续的Tensor。
scales2：Device侧的Tensor类型，表示量化过程中得到y2进行的scales张量。可选参数。shape需要与gamma保持一致。数据类型支持FLOAT32、BFLOAT16，数据类型需要与scales1保持一致。数据格式支持ND，支持非连续的Tensor。当参数divMode的值为True时，该参数的值不能为0。
zero_points2：Device侧的Tensor类型，表示量化过程中得到y2进行的offset张量。可选参数。shape需要与gamma保持一致。数据类型支持INT32、BFLOAT16，数据类型需要与zero_points1保持一致。数据格式支持ND，支持非连续的Tensor。
axis：Host侧的整型，表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入x的维度数。数据类型为int64_t，当前仅支持-1，传其他值均不生效。
epsilon：用于防止除0错误，数据类型为double。建议传较小的正数。默认值为1e-6。

输出说明
y1：Device侧的Tensor类型，表示量化后的输出数据。shape支持1-8维度，shape需要与输入x1/x2一致。数据类型支持INT8，数据格式支持ND，支持非连续的Tensor。
y2：Device侧的Tensor类型，表示量化后的输出数据。shape支持1-8维度，shape需要与输入x1/x2一致，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor。
x_out：Device侧的Tensor类型，表示x1和x2的和。shape支持1-8维度，shape需要与输入x1/x2一致。数据类型支持BFLOAT16、FLOAT16，需要与输入x1、x2一致。数据格式支持ND，支持非连续的Tensor。

支持的型号
Atlas 推理系列产品
Atlas A2训练系列产品/Atlas 800I A2中的推理产品/A200I A2 Box异构组件/Atlas A3训练系列产品/Atlas A3推理系列产品

调用示例:
import torch
import torch_npu

x_shape = [16, 32]
quant_shape = [32, ]
x1 = torch.randn(x_shape, dtype=torch.float16).npu()
x2 = torch.randn(x_shape, dtype=torch.float16).npu()
gamma = torch.randn(quant_shape, dtype=torch.float16).npu()
beta = torch.randn(quant_shape, dtype=torch.float16).npu()
scales1 = torch.randn(quant_shape, dtype=torch.float32).npu()
zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32).npu()

y1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, beta)
"""
)

_add_torch_npu_docstr(
    "npu_advance_step_flashattn",
    """
接口原型：
npu_advance_step_flashattn(Tensor(a!) input_tokens, Tensor sampled_token_ids, Tensor(b!) input_positions, Tensor(c!) seq_lens, Tensor(d!) slot_mapping, Tensor block_tables, int num_seqs, int num_queries, int block_size, *, Tensor? spec_token=None, Tensor? accepted_num=None) -> ()

功能描述
在npu上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新input_tokens，input_positions，seq_lens和slot_mapping，增加可选入参，用于支持投机推理的计算。

参数说明
input_tokens: Device侧的Tensor类型，输入/输出张量，用于更新vLLM模型中的token值；数据类型支持int64类型；如果是非投机场景，shape为[num_seqs,]，如果是投机场景，shape为[num_seqs, 1 + spec_num]；Shape第一维长度与num_seqs相同，不支持空tensor，必须为大于0的正整数；
sampled_token_ids: Device侧的Tensor类型，输入张量，用于储存token_id；数据类型支持INT64类型；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；Shape第一维长度与num_queries相同，不支持空tensor，必须为大于0的正整数。
input_positions: Device侧的Tensor类型，输入/输出张量，用于记录token的index；数据类型支持INT64类型；如果是非投机场景，shape为[num_queries, 1]，第二维长度是1；如果是投机场景，shape为[num_seqs, 1 + spec_num]；第一维长度与num_seqs相同，不支持空tensor，必须为大于0的正整数。
seq_lens: Device侧的Tensor类型，输入/输出张量，用于记录不同block_idx下seq的长度；数据类型支持INT64类型；非投机场景与投机场景shape均为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，必须为大于0的正整数。
slot_mapping: Device侧的Tensor类型，输入/输出张量，用于将token值在序列中的位置映射到物理位置；数据类型支持INT64类型；非投机场景与投机场景shape均为[num_seqs,]，第一维长度与num_seqs相同，不支持空tensor，必须为大于0的正整数。
block_tables: Device侧的Tensor类型，输入/输出张量，用于记录不同block_idx下block的大小；数据类型支持INT64类型；非投机场景与投机场景shape均为二维，第一维长度与num_seqs相同，第二维长度需要大于seq_lens_cpu中最大值除以block_size的整数部分，不支持空tensor，必须为大于0的正整数。
num_seqs: int类型，记录输入的seq数量；非投机场景与投机场景相同，必须为大于0的正整数。
num_queries: int类型，记录输入的query数量；投机场景下与num_seqs相同，必须为大于0的正整数。
block_size: int类型，记录每个block的大小；非投机场景与投机场景相同，必须为大于0的正整数。
spec_token: 可选参数，Device侧的Tensor类型，输入张量，用于记录投机场景下当前的token的idx。数据类型支持INT64类型；spec_token为空时，则为非投机场景，默认为None；spec_token不为空时，则为投机场景，shape为[num_seqs, spec_num]；不支持空tensor，必须为大于0的正整数。
accepted_num: 可选参数，Device侧的Tensor类型，输入张量，用于记录投机场景下每个request接受的投机的数量。数据类型支持INT64类型；accepted_num为空时，则为非投机场景，默认为None；accepted_num不为空时，则为投机场景，shape为[num_seqs,]；不支持空tensor，必须为大于0的正整数。

输出说明
此接口将原地更新input_tokens，input_positions，seq_lens和slot_mapping的值，无返回值。

约束说明
1. 输入input_tokens，input_positions，seq_lens，slot_mapping和block_tables的第一维长度与num_seqs相同
2. 投机场景下，输入input_tokens的第二维长度为1 + spec_num
3. 输入sampled_token_ids的第一维长度与num_queries相同，非投机场景下第二维长度为1，投机场景下第二维长度为1 + spec_num
4. 输入block_tables的shape的第二维长度大于seq_lens_cpu中最大值除以block_size的整数部分
5. 非投机场景下，输入num_seqs必须大于输入num_queries；投机场景下，num_queries下与num_seqs相同

支持的型号
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例:
非投机场景：
import numpy as np

import torch
import torch_npu

num_seqs = 16
num_queries = 8
block_size = 8
spec_num = 2

input_token = np.random.randint(10, size=(num_seqs,))
sampled_token_id = np.random.randint(10, size=(num_queries, 1))
input_position = np.random.randint(10, size=(num_seqs,))
seq_len = np.random.randint(10, size=(num_seqs,))
slot_mapping = np.random.randint(10, size=(num_seqs,))

input_tokens = torch.tensor(input_token, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_id, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_position, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_len, dtype=torch.int64, device="npu")
slot_mappings = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")

block_table = np.random.randint(10, size=(num_seqs, torch.max(seq_lens.cpu()) // block_size + 1))
block_tables = torch.tensor(block_table, dtype=torch.int64, device="npu")


torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                     seq_lens, slot_mappings, block_tables, num_seqs,
                                     num_queries, block_size, spec_token=None, accepted_num=None)

投机场景：
import numpy as np

import torch
import torch_npu

num_seqs = 16
block_size = 8
spec_num = 2

input_token = np.random.randint(10, size=(num_seqs, 1 + spec_num))
sampled_token_id = np.random.randint(10, size=(num_seqs, 1 + spec_num))
input_position = np.random.randint(10, size=(num_seqs,))
seq_len = np.random.randint(10, size=(num_seqs,))
slot_mapping = np.random.randint(10, size=(num_seqs,))
spec_token = np.random.randint(10, size=(num_seqs, spec_num))
accepted_num = np.random.randint(10, size=(num_seqs,))

input_tokens = torch.tensor(input_token, dtype=torch.int64, device="npu")
sampled_token_ids = torch.tensor(sampled_token_id, dtype=torch.int64, device="npu")
input_positions = torch.tensor(input_position, dtype=torch.int64, device="npu")
seq_lens = torch.tensor(seq_len, dtype=torch.int64, device="npu")
slot_mappings = torch.tensor(slot_mapping, dtype=torch.int64, device="npu")
spec_tokens = torch.tensor(spec_token, dtype=torch.int64, device="npu")
accepted_nums = torch.tensor(accepted_num, dtype=torch.int64, device="npu")

block_table = np.random.randint(10, size=(num_seqs, torch.max(seq_lens.cpu()) // block_size + 1))
block_tables = torch.tensor(block_table, dtype=torch.int64, device="npu")


torch_npu.npu_advance_step_flashattn(input_tokens, sampled_token_ids, input_positions,
                                     seq_lens, slot_mappings, block_tables, num_seqs,
                                     num_seqs, block_size, spec_token=spec_tokens, accepted_num=accepted_nums)
"""
)

_add_torch_npu_docstr(
    "empty_with_swapped_memory",
    """
接口原型：
torch_npu.empty_with_swapped_memory(size, dtype, device) -> Tensor

功能描述
申请一个device信息为NPU、实际内存在host侧的特殊tensor。

参数说明
size (ListInt) - 定义输出张量shape的整数序列。可以是参数数量(可变值)，也可以是列表或元组等集合。
dtype (torch.dtype, 可选，默认值为None) - 返回张量所需数据类型。如果值为None，请使用全局默认值(请参见torch.set_default_tensor_type()).
device (torch.device, 可选，默认值为None) - 返回张量的所需设备。

输出说明
此接口将返回一个device信息为NPU、实际内存在host侧的特殊tensor。

约束说明
1. 当前申请出来的特殊tensor仅支持如下算子：
torch.fill_
torch.zero_
torch_npu.npu_apply_adam_w
torch_npu.npu_hans_encode
torch_npu.npu_hans_decode
2. 支持版本
PyTorch 2.1，PyTorch 2.5及更高版本

支持的型号
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu

swapped_tensor = torch_npu.empty_with_swapped_memory([12, 12], dtype=torch.float32, device=torch.device("npu:0"))
swapped_tensor.zero_()
"""
)

_add_torch_npu_docstr(
    "npu_alltoallv_gmm",
    """
接口原型：
npu_alltoallv_gmm(Tensor gmm_x, Tensor gmm_weight, str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, *, Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False, bool permute_out_flag=False) -> (Tensor, Tensor, Tensor)

功能描述
alltoallv和grouped matmul的融合算子，对alltoallv通信后的输出做grouped matmul操作，通信时间和计算时间进行掩盖。

参数说明
    gmmX: device侧Tensor，表示输入，数据类型支持float16，bfloat16。该输入进行AllToAllv通信，仅支持二维, 数据格式支持ND，通信后结果作为GrouedMatMul计算的左矩阵
    gmmWeight：device侧Tensor，表示输入，数据类型支持float16, bfloat16，类型需与gmmX保持一致，仅支持三维, 数据格式支持ND，GrouedMatMul计算的右矩阵
    hcom：char*类型,计算输入，专家并行的通信域名。字符串长度需大于0，小于128。
    ep_world_size：int类型，计算输入，ep通信域size，支持8/16/32/64。
    sendCounts：int[]，计算输入，支持int数据类型，通信发送的数据量。
    recvCounts：int[]，计算输入，支持int数据类型，通信接收的数据量。
    send_counts_tensor：device侧Tensor，表示输入，暂不支持。
    recv_counts_tensor：device侧Tensor，表示输入，暂不支持。
    mm_x：device侧Tensor，表示输入，数据类型支持float16，bfloat16，共享专家的左矩阵。
    mm_weight：device侧Tensor，表示输入，数据类型支持float16，bfloat16，共享专家的右矩阵。
    transGmmWeight：为True：表明gmm的右矩阵要转置，为False时表明gmm右矩阵不转置，默认为false
    transMmWeight：为True：表明mm的右矩阵要转置，为False时表明mm右矩阵不转置，默认为false
    permute_out_flag：为True：表明permute结果输出，为False时表明permute结果不输出，默认为false

输出说明
    gmmY：device侧Tensor, 计算输出,数据类型支持float16, bfloat16。最终计算结果，数据类型与输入gmmX保持一致
    mmY：device侧Tensor, 数据类型支持float16, bfloat16，共享专家matmul的输出，仅当传入mmX与mmWeight才输出，数据类型与mmX保持一致。
    permute_out：device侧Tensor, 数据类型支持float16, bfloat16，alltoallv输出的中间结果，permute_out_flag为True表明permute结果输出，为False时表明permute结果不输出。

支持的型号
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

def run_npu_alltoallv_gmm(rank, world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    input = torch.randn(gmm_x, dtype=dtype).npu()
    weight = torch.randn(gmm_w, dtype=dtype).npu()
    gmmYOut, _, _ = torch_npu.npu_alltoallv_gmm(gmm_x=input,
                                                gmm_weight=weight,
                                                send_counts_tensor=None,
                                                recv_counts_tensor=None,
                                                mm_x=None,
                                                mm_weight=None,
                                                group=hcom_info,
                                                ep_world_size=world_size,
                                                send_counts=send_counts,
                                                recv_counts=recv_counts,
                                                trans_gmm_weight=False,
                                                trans_mm_weight=False,
                                                permute_out_flag=True)

def generate_matrix(self, e, ep_world_size, bsk, name="alltoallv_gmm", max_iter=10000):
    import hashlib
    hash_bytes = hashlib.sha256(name.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    np.random.seed(seed)
    row_size = ep_world_size
    col_size = e * ep_world_size
    matrix = []
    avg = bsk // col_size
    tail_num = bsk % col_size
    matrix = np.full((row_size, col_size), avg)
    matrix[:, -1] += tail_num
    return matrix

if __name__ == "__main__":
    worksize = 8
    e = 4
    master_ip = '127.0.0.1'
    master_port = '50001'
    BS = 128
    K = 8
    x1_shape = [BS*K, 2048]
    x2_shape = [2048, 2048]
    send_counts = self.generate_matrix(e, worksize, BS*K)
    recv_counts = np.hstack(np.split(mc2_send_counts.reshape(-1, e), epWorldSize, axis=0))

    dtype = torch.float16

    mp.spawn(run_npu_alltoallv_gmm, args=(worksize, master_ip, master_port, gmm_x, gmm_weight, send_counts, recv_counts, dtype), nprocs=worksize)
"""
)

_add_torch_npu_docstr(
    "npu_swiglu_quant",
    """
torch_npu.npu_swiglu_quant(Tensor x, *, Tensor? smooth_scales=None, Tensor? offsets=None, Tensor? group_index=None, bool activate_left=False, int quant_mode=0, int group_list_type=0, ScalarType? dst_type=None) -> (Tensor, Tensor)
功能描述
在swiglu激活函数后添加quant操作。

参数说明
x (Tensor)：必选参数，表示目标张量。数据类型支持float16、bfloat16、float32，支持非连续的Tensor，数据格式为ND，x的维数必须大于1维，尾轴为偶数且长度不超过8192，当dst_type传入值为29(输出为int4量化)时，x的最后一维需要为4的倍数。
smooth_scales (Tensor)：可选参数，表示smooth量化系数。数据类型支持float32，支持非连续的Tensor，数据格式为ND。shape支持[G, N]，[G, ]。
offsets (Tensor)：可选参数，表示量化中的偏移项，该参数在动态量化场景下不生效，传入None即可。静态量化场景下：数据类型支持FLOAT，支持非连续的Tensor，数据格式为ND。per_channel模式下shape支持[G, N]，per_tensor模式下shape支持[G, ]，且数据类型和shape需要与smooth_scales保持一致。
group_index (Tensor)：可选参数，当前支持cumsum和count两种模式，要求为1维张量，数据类型支持int32，数据格式ND，shape支持[G, ]，group_index内元素要求为非递减，且最大值不得超过输入x的除最后一维之外的所有维度大小之积。
activate_left (bool)：可选参数，Swiglu流程中是否进行左激活，默认False。
quant_mode (int)：可选参数，表示量化类型，默认值为0。0表示静态量化，1表示动态量化。
group_list_type (int)：可选参数，表示group_index类型，默认值为0。0表示cumsum模式，1表示count模式。
dst_type (ScalarType): 可选参数，指定量化输出的类型, 传None时当做torch.int8处理。

输出说明
out (Tensor)：表示量化后的输出tensor。数据类型支持int8和int4，支持非连续的Tensor，数据格式为ND。
scale (Tensor)：表示量化的scale参数，计算输出scale的shape与计算输入x相比，无最后一维，其余维度与计算输入x保持一致，数据类型支持float32，数据格式为ND。

支持的型号
A2训练、推理系列产品
A3训练、推理系列产品

调用示例
import os
import shutil
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUSwigluQuant(TestCase):
    def test_npu_swiglu_quant(self, device="npu"):
        batch_size = 4608
        hidden_size = 2048
        x_shape = (batch_size, hidden_size)
        input_data = np.random.randn(*x_shape).astype(np.float32)

        quant_mode = 1
        group_list_type = 0
        dst_type = torch.int8
        activate_left = False
        offsets = None
        num_groups = 8
        group_sizes = batch_size // num_groups
        group_index = [(i + 1) * group_sizes for i in range(num_groups)]
        smooth_scales = np.random.randn(num_groups, hidden_size // 2).astype(np.float32)

        device = "npu"
        npu_x = torch.from_numpy(input_data).to(device)
        npu_group_index = torch.from_numpy(np.array(group_index)).to(device)
        npu_smooth_scales = torch.from_numpy(smooth_scales).to(device)
        result = torch_npu.npu_swiglu_quant(
            npu_x,
            smooth_scales=npu_smooth_scales,
            offsets=offsets,
            group_index=npu_group_index,
            activate_left=False,
            quant_mode=quant_mode,
            group_list_type=group_list_type,
            dst_type=dst_type
        )

if __name__ == "__main__":
    run_tests()

"""
)

_add_torch_npu_docstr(
    "npu_grouped_matmul_swiglu_quant",
    """
torch_npu.npu_grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor group_list, Tensor weight_scale, Tensor x_scale, *, Tensor? bias=None, Tensor? offset=None) -> (Tensor, Tensor, Tensor)
功能描述
aclnnGroupedMatmulV4、aclnnDynamicDequant、aclnnSwigluQuant融合, deepseek模型使用，对比小算子做性能优化。

参数说明
x（Tensor）：输入，左矩阵，公式中的X，Device侧的aclTensor。shape支持2维，数据类型支持INT8，数据格式支持ND，支持非连续的Tensor。
weight（Tensor）：输入，权重矩阵，公式中的W，Device侧的aclTensor。shape支持5维，数据类型支持INT8，数据格式支持FRACTAL_NZ，支持非连续的Tensor，需注意该接口会将weight的数据格式强制视为FRACTAL_NZ格式。
group_list （Tensor）：输入，指示每个分组参与计算的Token个数，公式中的grouplist，Device侧的aclTensor。shape支持1维，长度需与weight的首轴维度相等，数据类型支持INT64，数据格式支持ND，支持非连续的Tensor。
weight_scale （Tensor）：输入，右矩阵的量化因子，公式中的w_scale，Device侧的aclTensor。shape支持2维，首轴长度需与weight的首轴维度相等，尾轴长度需要与weight还原为ND格式的尾轴相同，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
x_scale （Tensor）：输入，左矩阵的量化因子，公式中的x_scale，Device侧的aclTensor。shape支持1维，长度需与x的首轴维度相等，数据类型支持FLOAT，数据格式支持ND，支持非连续的Tensor。
bias（可选，暂不支持，Tensor）：输入，矩阵乘计算的偏移值，公式中的bias，shape支持2维，数据类型支持INT32，预留输入，暂不支持。
offset（可选，暂不支持，Tensor）：输入，per-channel非对称反量化的偏移，公式中的offset，shape支持2维，数据类型支持Float，预留输入，暂不支持。

输出说明
group_list指导了输入和输出中的有效值范围，该数值由前置算子得到，动态变化。应根据group_list，对结果中脏数据做截断处理，即有效数据截至到group_list[-1]，即：output[:groupList[-1],:],output_scale[:groupList[-1]]
output（Tensor）：输出的量化结果，公式中的Q，Device侧的aclTensor。数据类型支持INT8，shape支持2维，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor。
output_scale（Tensor）：输出的量化因子，公式中的Q_scale，Device侧的aclTensor。数据类型支持FLOAT，shape支持1维，Device侧的aclTensor。数据格式支持ND，支持非连续的Tensor。
output_offset（预留输出，暂不支持，Tensor）：输出的非对称量化的偏移，公式中的Q_offset，Device侧的aclTensor，shape支持1维，数据类型支持FLOAT。

支持的型号
A2训练、推理系列产品
A3训练、推理系列产品

调用示例
import torch
import torch_npu
import numpy as np

def generate_non_decreasing_sequence(length, upper_limit):
    # 生成随机增量
    random_increments = torch.randint(1, 128, (length,), dtype=torch.int64)  # 避免零增量
    # 累加生成非递减序列
    sequence = torch.cumsum(random_increments, dim=0)
    # 确保最后一个元素不超过上限
    if sequence[-1] > upper_limit:
        # 线性缩放以确保总和不超过上限
        scale_factor = upper_limit / sequence[-1].item()
        sequence = (sequence * scale_factor).to(torch.int64)
        for i in range(1, length):
            if sequence[i] <= sequence[i-1]:
                sequence[i] = sequence[i-1] + 1
    return sequence

def gen_input_data(E=16, M=512, K=7168, N=4096):
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
    weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).npu()
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    weight_scale = torch.randn(E, N, dtype=torch.float32).npu()
    x_scale = torch.randn(M, dtype=torch.float32).npu()
    group_list = generate_non_decreasing_sequence(E, M).npu()
    output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(
        x, weight_npu, group_list, weight_scale, x_scale,
        bias=None,
        offset=None
    )
    return output, output_scale, output_offset

def main():
    output, output_scale, output_offset = gen_input_data()

if __name__ == "__main__":
    main()

"""
)

_add_torch_npu_docstr(
    "npu_grouped_matmul_swiglu_quant_v2",
    """
torch_npu.npu_grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale, Tensor group_list, *, Tensor? smooth_scale=None, Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, int? dequant_dtype=0, int? quant_mode=0, int? quant_dtype=0, int? group_list_type=0, int[]? tuning_config=None) -> (Tensor, Tensor)
功能描述
`npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、SwiGLu混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGlu激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

参数说明
x（Tensor）：必选输入，矩阵乘法的左矩阵。shape支持2维[m,k]，数据类型支持`int8`，数据格式支持ND，支持非连续的Tensor。
weight（TensorList）：：必选输入，权重矩阵(矩阵乘法右矩阵)，shape支持3维[e,k,n]，数据类型支持`int8`，数据格式支持FRACTAL_NZ(通过接口npu_format_cast，可实现格式转换)，支持非连续的Tensor。
weight_scale （TensorList）：必选输入，右矩阵的量化因子。shape支持2维[e,n],,,数据类型支持`float32`，数据格式支持ND，支持非连续的Tensor。
x_scale （Tensor）：必选输入，左矩阵的量化因子。shape支持1维[m]，数据类型支持`float32`，数据格式支持ND，支持非连续的Tensor。
group_list （Tensor）：必选输入，指示每个分组参与计算的Token个数。shape支持1维[e]，数据类型支持`int64`，数据格式支持ND，支持非连续的Tensor。
smooth\_scale（`Tensor`）：可选输入，量化的smooth_scales,数据类型为`float32`，当前仅支持传入默认值None。
weight_assist_matrix（`TensorList`）：可选输入，右矩阵的辅助矩阵，数据类型支持`float32`，当前仅支持传入默认值None。
bias（`Tensor`）：可选输入，矩阵乘计算的偏移值，公式中的bias，shape支持2维，数据类型支持`int32`，当前仅支持传入默认值None。
dequant_dtype（`int`）：可选输入，表示反量化类型，参数值对应0：pertoken 1：pergroup，数据类型为`int32`，当前仅支持传入默认值0。
dequant_mode（`int`）：可选输入，表示反量化模式，参数值对应0：左pertoken，右perchannel 1：左pertoken，右pergroup，数据类型为`int32`，当前仅支持传入默认值0。
quant_dtype（`int`）：可选输入，参数表示量化后低比特数据类型。0：`int8`；1：`float8_e8m0`；2：`float8_e5m2`；3：`float8_e4m3`，数据类型为`int32`，当前仅支持传入默认值0。
quant_mode（`int`）：可选输入，参数表示swiglu后的量化模式。0：pertoken 1：perchannel，数据类型为`int32`，当前仅支持传入默认值0。
group_list_type（`int`）：可选输入，参数表示grouplist的输入类型。0：cunsum 1：count，数据类型为`int32`，当前仅支持传入默认值0。
tuningConfig（`List[int]`）：可选输入，参数数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认设置为None。

输出说明
output（`Tensor`）：输出的量化结果，数据类型支持`int8`，shape支持2维[m,n]。数据格式支持ND，支持非连续的Tensor。
output_scale（`Tensor`）：输出的量化因子，数据类型支持`float`，shape支持1维[m]。数据格式支持ND，支持非连续的Tensor。

支持的型号
A2训练、推理系列产品
A3训练、推理系列产品

调用示例
import torch
import torch_npu
import numpy as np

def test_grouped_matmul_swiglu_quant_v2(E=16, M=512, K=7168, N=4096):
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
    weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).npu()
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    weight_scale = torch.randn(E, N, dtype=torch.float32).npu()
    x_scale = torch.randn(M, dtype=torch.float32).npu()
    groupList = torch.tensor([128, 128], dtype=torch.int64)
    output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x, [weight_npu], [weight_scale], x_scale, group_list, 
        bias=None,
    )
    return output, output_scale

def main():
    output, output_scale = test_grouped_matmul_swiglu_quant_v2()

if __name__ == "__main__":
    main()

"""
)

_add_torch_npu_docstr(
    "npu_gmm_alltoallv",
    """
接口原型：
npu_gmm_alltoallv(Tensor gmm_x, Tensor gmm_weight, str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, *, Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False) -> (Tensor, Tensor)

功能描述
grouped matmul和alltoallv的融合算子，对grouped matmul计算后的结果进行alltoallv通信的输出做操作，通信时间和计算时间进行掩盖。

参数说明
    gmm_x: device侧Tensor，表示输入，数据类型支持float16，bfloat16。该输入进行AllToAllv通信，仅支持二维, 数据格式支持ND，通信后结果作为GrouedMatMul计算的左矩阵
    gmm_weight：device侧Tensor，表示输入，数据类型支持float16, bfloat16，类型需与gmmX保持一致，仅支持三维, 数据格式支持ND，GrouedMatMul计算的右矩阵
    hcom：char*类型,计算输入，专家并行的通信域名。字符串长度需大于0，小于128。
    ep_world_size：int类型，计算输入，ep通信域size，支持8/16/32/64。
    send_counts：int[]，计算输入，支持int数据类型，通信发送的数据量。
    recv_counts：int[]，计算输入，支持int数据类型，通信接收的数据量。
    send_counts_tensor：device侧Tensor，表示输入，暂不支持。
    recv_counts_tensor：device侧Tensor，表示输入，暂不支持。
    mm_x：device侧Tensor，表示输入，数据类型支持float16，bfloat16，共享专家的左矩阵。
    mm_weight：device侧Tensor，表示输入，数据类型支持float16，bfloat16，共享专家的右矩阵。
    trans_gmm_weight：为True：表明gmm的右矩阵要转置，为False时表明gmm右矩阵不转置，默认为false。
    trans_mm_weight：为True：表明mm的右矩阵要转置，为False时表明mm右矩阵不转置，默认为false。

输出说明
    y：device侧Tensor, 计算输出,数据类型支持float16, bfloat16。最终计算结果，数据类型与输入gmm_X保持一致
    mm_y：device侧Tensor, 数据类型支持float16, bfloat16，共享专家matmul的输出，仅当传入mm_x与mm_weight才输出，数据类型与mm_x保持一致。


支持的型号
Atlas A3训练系列产品

调用示例:
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

def run_npu_gmm_alltoallv(rank, world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    input = torch.randn(gmm_x, dtype=dtype).npu()
    weight = torch.randn(gmm_w, dtype=dtype).npu()
    y, _= torch_npu.npu_gmm_alltoallv(gmm_x=input,
                                      gmm_weight=weight,
                                      send_counts_tensor=None,
                                      recv_counts_tensor=None,
                                      mm_x=None,
                                      mm_weight=None,
                                      group=hcom_info,
                                      ep_world_size=world_size,
                                      send_counts=send_counts,
                                      recv_counts=recv_counts,
                                      trans_gmm_weight=False,
                                      trans_mm_weight=False)

def generate_matrix(self, e, ep_world_size, bsk, name="alltoallv_gmm", max_iter=10000):
    import hashlib
    hash_bytes = hashlib.sha256(name.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    np.random.seed(seed)
    row_size = ep_world_size
    col_size = e * ep_world_size
    matrix = []
    avg = bsk // col_size
    tail_num = bsk % col_size
    matrix = np.full((row_size, col_size), avg)
    matrix[:, -1] += tail_num
    return matrix

if __name__ == "__main__":
    worksize = 8
    e = 4
    master_ip = '127.0.0.1'
    master_port = '50001'
    BS = 128
    K = 8
    x1_shape = [BS*K, 2048]
    x2_shape = [2048, 2048]
    send_counts = self.generate_matrix(e, worksize, BS*K)
    recv_counts = np.hstack(np.split(mc2_send_counts.reshape(-1, e), epWorldSize, axis=0))

    dtype = torch.float16

    mp.spawn(run_npu_gmm_alltoallv, args=(worksize, master_ip, master_port, gmm_x, gmm_weight, send_counts, recv_counts, dtype), nprocs=worksize)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_compress",
    """
torch_npu.npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len=None)
功能描述
实现Native Sparse Attention算法中训练场景下的压缩功能。

参数说明
input(Tensor)：必选参数，待压缩张量，shape支持[T,N,D]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
weight(Tensor)：必选参数，压缩的权重，shape支持[compress_block_size, N]，weight和input的shape满足broadcast关系，数据类型支持bfloat16、float16，数据类型与input保持一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
compress_block_size(int)：必选参数，压缩滑窗的大小。
compress_stride(int)：必选参数，两次压缩滑窗间隔大小。
actual_seq_len(list[int])：必选参数，长度表示query有多少个batch，值表示各batch的token长度的前缀和，例如，actual_seq_len[0]=s0,actual_seq_len[1]=s0+s1，...，actual_seq_len[-1]=T。

输出说明
代表压缩后的结果。

约束说明
input.shape[1] = weight.shape[1] = head_num
compress_block_size、compress_stride 必须是16的整数倍，且compress_block_size>=compress_stride
input.shape[0] = act_seq_len[-1]
input.shape[2] = head_dim必须是16的整数倍
目前仅支持head_num<=128，compress_block_size <= 128, head_dim <= 256

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> actual_seq_len = np.random.randint(0, 100, [48])
>>> actual_seq_len = np.cumsum(actual_seq_len).astype(np.int64)
>>> head_num = 4
>>> head_dim = 128
>>> compress_block_size = 16
>>> compress_stride = 16
>>> input = torch.randn(actual_seq_len[-1], head_num, head_dim, dtype=torch.float16).npu()
>>> weight = torch.randn(compress_block_size, head_num, dtype=torch.float16).npu()
>>> torch_npu.npu_nsa_compress(input, weight, compress_block_size, compress_stride, actual_seq_len=actual_seq_len)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_compress_infer",
    """
torch_npu.npu_nsa_compress_infer(input, weight, slot_mapping, compress_block_size, compress_stride, page_block_size, block_table=None, actual_seq_len=None, cache)
功能描述
Native Sparse Attention算法中推理场景下，实现对KV压缩的计算。

参数说明
input(Tensor)：必选输入，待压缩张量，shape支持[block_num,page_block_size,head_num,head_dim]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
weight(Tensor)：必选输入，压缩的权重，shape支持[compress_block_size, head_num]，weight和input的shape满足broadcast关系，数据类型支持bfloat16、float16，数据类型与input保持一致，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
slot_mapping(Tensor)：必选输入，表示每个batch尾部压缩数据存储的位置的索引，shape支持[batch_num]，数据类型支持int32，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
compress_block_size(int)：必选输入，压缩滑窗的大小。
compress_stride(int)：必选输入，两次压缩滑窗间隔大小。
page_block_size(int)：必选输入，page_attention场景下page的block_size大小。
block_table(Tensor)：可选输入，page_attention场景下kv缓存使用的block映射表，不支持非连续的Tensor。
actual_seq_len(list[int])：必选输入，表示每个batch对应的token的长度。
cache(Tensor)：必选输入，推理场景下的kv缓存，支持非连续的Tensor，不支持空Tensor。

输出说明
代表对KV压缩计算后的结果。

约束说明
input和weight满足broadcast关系，input的第三维大小与weight的第二维大小相等。
compress_block_size、compress_stride 必须是16的整数倍，且compress_block_size>=compress_stride，compress_block_size <= 64。
actual_seq_len目前仅支持取值1。
page_block_size只能是64或者128。
headDim是16的整数倍，且headDim <= 256。
需保证slotMapping的值无重复，否则会导致计算结果不稳定。
blockTable的值不应超过blockNum，否则会发生越界。
actual_seq_len的值不应该超过最大序列长度。
headNum <= 64，且headNum>50时headNum%2=0。

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> input = torch.randn(1, 128, 1, 192, dtype=torch.float16).npu()
>>> weight = torch.randn(32, 1, dtype=torch.float16).npu()
>>> slot_mapping = torch.randn([1]).int().npu()
>>> compress_block_size = 32
>>> compress_stride = 16
>>> page_block_size = 128
>>> act_seq_lens = [43]
>>> block_table = torch.randn([1, 1]).int().npu()
>>> cache = torch.zeros([1, 1, 192],dtype=torch.float16).npu()
>>> torch_npu.npu_nsa_compress_infer(input, weight,slot_mapping,compress_block_size,compress_stride,page_block_size,actual_seq_len=act_seq_lens,block_table=block_table,cache=cache)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_compress_attention",
    """
torch_npu.npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, topk_mask=None, atten_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None)
功能描述
实现Native Sparse Attention算法中训练场景下的压缩注意力功能。

参数说明
query(Tensor)：必选参数，shape支持[T,N,D]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
key(Tensor)：必选参数，shape支持[T,N2,D]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
value(Tensor)：必选参数，shape支持[T,N2,D2]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
scale_value(double)：必选参数，表示缩放系数，一般设置为D^-0.5。
head_num(int)：必选参数，表示query的head个数。
compress_block_size(int)：必选参数，压缩滑窗的大小。
compress_stride(int)：必选参数，两次压缩滑窗间隔大小。
select_block_size(int)：必选参数，表示select窗口的大小。
select_block_count(int)：必选参数，表示select窗口的数量。
topk_mask(Tensor)：可选参数，shape支持[S,S]，SS分别是max_sq和max_skv，数据类型支持bool。
atten_mask(Tensor)：可选参数，取值为1代表该位不参与计算（不生效），为0代表该位参与计算，数据类型支持bool，数据格式支持ND，输入shape类型支持[S,S]格式，SS分别是maxSq和maxSkv。
actual_seq_qlen(list[int])：必选参数，长度表示query有多少个batch，值表示各batch的token长度的前缀和，例如，actual_seq_qlen[0]=s0,actual_seq_qlen[1]=s0+s1，...，actual_seq_qlen[-1]=T。
actual_cmp_seq_kvlen(list[int])：必选参数，长度表示compress attention的key或value有多少个batch，值表示各batch的token长度的前缀和，例如，actual_cmp_seq_kvlen[0]=cmp_skv[0],actual_cmp_seq_kvlen[1]=cmp_skv[0]+cmp_skv[1]，...，actual_cmp_seq_kvlen[-1]=T。
actual_sel_seq_kvlen(list[int])：必选参数，长度表示select attention的key/value有多少个batch，值表示各batch的token长度的前缀和，例如，actual_sel_seq_kvlen[0]=sel_skv[0],actual_sel_seq_kvlen[1]=sel_skv[0]+sel_skv[1]，...，actual_sel_seq_kvlen[-1]=T。

输出说明
Tensor：代表压缩注意力attention的结果。
Tensor：代表选择出的topk。
Tensor：代表softmax计算的max中间结果，用于反向计算。
Tensor：代表softmax计算的sum中间结果，用于反向计算。

约束说明
compress_block_size、compress_stride、select_block_size必须是16的整数倍；且compress_block_size >= compress_stride，select_block_size >= compress_block_size，select_block_size % compress_stride == 0；selectBlockCount <= selKvLen。
目前仅支持compress_block_size=32, compress_stride=16, select_block_size=64, select_block_count=16。
cmp_skv[i] <= 14000。
sel_skv[i] = CeilDiv(cmp_skv[i], select_block_size // compress_stride)。
query、key、value的数据类型必须一致。
query、key、value的B：batchsize必须相等。
query、key、value的D：Head-Dim必须满足(qD == kD && kD >= vD)。
query、key、value的input_layout属性必须一致。
query、key、value的N：qN >= kN && kN == vN，qN与kN必须成比例关系，即qN / kN必须是非0整数。
G=qN / kN, G必须满足：G<128 && 128 % G == 0。
SparseMode：当前仅支持1；attenMask可传入[masS1, maxCmpS2]的下三角或none，topkMask可传入[maxS1, maxSelS2]的对角线或none（attenMask和topkMask数据填充也必须符合约束）。

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> query = torch.randn(65536, 64, 192, dtype=torch.bfloat16).npu()
>>> key = torch.randn(4096, 4, 192, dtype=torch.bfloat16).npu()
>>> value = torch.randn(4096, 4, 128, dtype=torch.bfloat16).npu()
>>> scale_value = 1 / (192**0.5)
>>> head_num = 64
>>> compress_block_size = 32
>>> compress_stride = 16
>>> select_block_size = 64
>>> select_block_count = 16
>>> actual_seq_qlen = [65536]
>>> actual_cmp_seq_kvlen = [4096]
>>> actual_sel_seq_kvlen = [1024]
>>> torch_npu.npu_nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=actual_sel_seq_kvlen)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_compress_attention_infer",
    """
torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, layout='TND', atten_mask=None, block_table=None, topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None)
功能描述
Native Sparse Attention算法中推理场景下，实现对KV压缩的计算。

参数说明
query(Tensor)：必选输入，layout为TND时，shape支持3维输入，为[batch, key_value_head_num * group_size, head_size_qk]，layout为BSND时，shape支持4维输入，为[batch, query_seq_len, key_value_head_num * group_size, head_size_qk]，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor，不支持inf，nan。
key(Tensor)：必选输入，shape支持3维输入，为[block_num, page_block_size, head_size_qk * key_value_head_num]，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor，不支持inf，nan。
value(Tensor)：必选输入，shape支持3维输入，为[block_num, page_block_size, head_size_v * key_value_head_num]，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor，不支持inf，nan。
scale_value(double)：必选输入，表示缩放系数。
head_num(int)：必选输入，表示query的head个数。
key_value_head_num(int)：必选输入，表示key或者value的head个数。
select_block_size(int)：必选输入，表示选择窗口的大小。
select_block_count(int)：必选输入，表示选择窗口的数量。
page_block_size**(int)：必选输入，page_attention场景下page的block_size大小。
compress_block_size**(int)：必选输入，压缩滑窗的大小。
compress_stride**(int)：必选输入，两次压缩滑窗间隔大小。
layout(str)：可选输入，表示输入的数据排布格式，支持TND、BSND，默认为TND。
atten_mask(Tensor)：可选输入，当前不支持。
block_table**(Tensor)：可选输入，shape支持2维输入，数据类型支持‘int32’，page_attention场景下kv缓存使用的block映射表，不支持非连续的Tensor，不支持空tensor，不支持inf，nan。
topk_mask**(Tensor)：可选输入，当前不支持。
actual_seq_qlen(list[int])：可选输入，当前不支持。
actual_cmp_seq_kvlen(list[int])：必选输入，表示压缩注意力的key/value的每个S的长度。
actual_sel_seq_kvlen(list[int])：可选输入，当前不支持。

输出说明
代表对KV压缩计算后的结果。

约束说明
- query的数据排布格式中，T代表B（Batch）与S（Seq-Length）合轴后的结果、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- key和value的数据排布格式当前（paged attention）支持（block_num, block_size, H），H（Head-Size）表示隐藏层的大小，H=N∗D。
- 参数query中的N和head_num值相等，key、value的N和key_value_head_num值相等，并且head_num是key_value_head_num的倍数关系。
- 参数query中的D和key的D(H/key_value_head_num)值相等。
- 参数query中的B、block_table的B、actual_cmp_seq_kvlen的shape值相等，B取值范围1-20。
- 参数key中的block_num和参数value中的block_num值相等。
- 参数key中的block_size、参数value中的block_size和page_block_size值相等。
- query，key，value输入，功能使用限制如下：
  -  支持query的N轴必须是key/value的N轴（H/D）的整数倍。
  -  支持query的N轴与key/value的N轴（H/D）的比值小于等于128，且128是group的整数倍。
  -  支持query与Key的D轴小于等于192，scale_value取值D^-0.5。
  -  支持value的D轴小于等于128。
  -  支持query与Key的D轴大于等于value的D轴。
  -  支持key与value的block_size小于等于128，且是16的整数倍。
  -  仅支持query的S轴等于1。
  -  仅支持paged attention。
  -  仅支持key/value的S轴小于等于8192。
  -  仅支持compress_block_size取值16、32、48、64、80、96、112、118。
  -  仅支持compress_stride取值16、32、48、64。
  -  仅支持select_block_size取值16、32、48、64、80、96、112、118。
  -  仅支持compress_block_size大于等于compress_stride , select_block_size大于等于compress_block_size , select_block_size是compress_stride的整数倍。
  -  压缩前的kv_seq_len的上限可以表示为：no_cmp_kv_seq_len_ceil = (cmp_kv_seq_len − 1) ∗ compress_block_stride + compress_block_size，需要满足no_cmp_kv_seq_len_ceil / select_block_size <= 4096，且需要满足select_block_count <= no_cmp_kv_seq_len_ceil / select_block_size。
  -  block_size第2维的取值需满足公式(max(cmp_kv_seq_len) + page_block_size - 1) // page_block_size。
  -  block_num的取值需满足公式B * (max(cmp_kv_seq_len) + page_block_size - 1) // page_block_size。
  -  block_table的取值范围需满足[0, block_num]。
  -  query，key，value的数据类型需保持一致。
  -  actual_cmp_seq_kvlen的取值范围为[128, 4096]。

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> query = torch.randn([1, 32, 65], dtype=torch.float16).npu()
>>> key = torch.randn([25, 48, 65], dtype=torch.float16).npu()
>>> value = torch.randn([25, 48, 18], dtype=torch.float16).npu()
>>> scale_value = 0.01
>>> head_num = 32
>>> key_value_head_num = 1
>>> select_block_size = 32
>>> select_block_count = 397
>>> page_block_size = 48
>>> compress_block_size = 32
>>> compress_stride = 16
>>> block_table = torch.tensor([[23, 2, 20, 22, 4, 21, 7, 12, 3, 20, 20, 0, 15, 0, 4, 8, 10, 20, 21, 18, 18, 18, 11, 12, 20]]).int().npu()
>>> actual_cmp_seq_kvlen = [1180]
>>> torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, compress_block_size, compress_stride, block_table=block_table, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_select_attention",
    """
torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None)
功能描述
实现Native Sparse Attention算法中训练场景下选择注意力的计算。

参数说明
query(Tensor)：必选参数，shape支持[T1,N1,D1]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
key(Tensor)：必选参数，shape支持[T2,N2,D1]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
value(Tensor)：必选参数，shape支持[T2,N2,D2]，数据类型支持bfloat16、float16，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
topk_indices(Tensor)：必选参数，shape为[T1, N2, select_block_count]，数据类型支持int32，数据格式支持ND，支持非连续的Tensor，不支持空Tensor。
scale_value(double)：必选参数，表示缩放系数，一般设置为D^-0.5。
head_num(int)：必选参数，表示单卡的head个数，即query的N1轴长度。
select_block_size(int)：必选参数，表示select窗口的大小。
select_block_count(int)：必选参数，表示select窗口的数量。
atten_mask(Tensor)：可选参数，当前暂不支持。
actual_seq_qlen(list[int])：必选参数，长度表示query有多少个batch，值表示各batch的token长度的前缀和，例如，actual_seq_qlen[0]=s0,actual_seq_qlen[1]=s0+s1，...，actual_seq_qlen[-1]=T1。
actual_seq_kvlen(list[int])：必选参数，，长度表示key或value有多少个batch，值表示各batch的token长度的前缀和，例如，actual_seq_kvlen[0]=s0,actual_seq_kvlen[1]=s0+s1，...，actual_seq_kvlen[-1]=T2。

输出说明
Tensor：代表经过选择后的注意力attention结果。
Tensor：代表softmax计算的max中间结果，用于反向计算。
Tensor：代表softmax计算的sum中间结果，用于反向计算。

约束说明
1. 输入query、key、value的batchsize必须相等，即要求传入的actual_seq_qlen和actual_seq_kvlen具有相同的长度。
2. 输入query、key、value的D（head_dim）必须满足D_q == D_k，D_k >= D_v。
3. 输入query、key、value的数据类型必须一致。
4. 输入query、key、value的input_layout必须一致，且只支持TND。
5. select_block_size目前仅支持64，与此对应的select_block_count为16。
6. topk_indices必须大于等于0且小于等于B对应的S2 / 64。
7. 支持输入query的N和key/value的N不相等，但必须成比例关系，即N_q / N_kv必须是非0整数，称为G（group），且需满足G <= 32
- B（batchsize）：取值范围为1\~65536。
- N（head_num）：取值范围为1\~128。
- G（group）：取值范围为1\~32。
- S（seq_length）：取值范围为1\~128K。且对于KV的S >= select_block_size * select_block_count,且为select_block_size的倍数。
- D（head_dim）：D_qk=192，D_v=128。

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> query = torch.randn(256, 16, 192, dtype=torch.float16).npu()
>>> key = torch.randn(3072, 4, 192, dtype=torch.float16).npu()
>>> value = torch.randn(3072, 4, 128, dtype=torch.float16).npu()
>>> topk_indices = torch.randn(256, 4, 16).int().npu()
>>> scale_value = 1.0
>>> head_num = 16
>>> select_block_size = 64
>>> select_block_count = 16
>>> atten_mask = torch.randn(512, 2048).bool().npu()
>>> actual_seq_qlen = [128, 256]
>>> actual_seq_kvlen = [2048, 3072]
>>> torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
"""
)


_add_torch_npu_docstr(
    "npu_nsa_select_attention_infer",
    """
torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout='BSND', atten_mask=None, block_table=None, actual_seq_qlen=None, actual_seq_kvlen=None)
功能描述
Native Sparse Attention算法中推理场景下，实现选择注意力的计算。

参数说明
query (Tensor)：必选输入，shape支持3维或者4维，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
key (Tensor)：必选输入，shape支持3维或者4维，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
value (Tensor)：必选输入，shape支持3维或者4维，数据类型支持bfloat16、float16，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
topk_indices (Tensor)：必选输入，shape为[batch_size, key_value_head_num, select_block_count]，数据类型支持int32，数据格式支持ND，不支持非连续的Tensor，不支持空Tensor。
scale_value (double)：必选输入，表示缩放系数。
head_num (int)：必选输入，表示query的head个数。
key_value_head_num (int)：必选输入，表示key或者value的head个数。
select_block_size (int)：必选输入，表示选择窗口的大小。
select_block_count (int)：必选输入，表示选择窗口的数量。
page_block_size(int)：必选输入，page_attention场景下page的block_size大小。
atten_mask (Tensor)：可选输入，当前暂不支持。
block_table(Tensor)：可选输入，page_attention场景下kv缓存使用的block映射表，数据类型支持int32，不支持非连续的Tensor，不支持空tensor。
layout(str)：可选输入，表示输入的数据排布格式，支持BSH、BSND、TND，默认为BSND。
actual_seq_qlen(list[int])：可选输入，当前暂不支持。
actual_seq_kvlen(list[int])：必选输入，表示key或value每个S的长度。

输出说明
代表经过选择后的注意力结果。

约束说明
query的数据排布格式中，B即Batch，S即Seq-Length，N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。key和value的数据排布格式当前（paged attention场景）支持(block_num, block_size, H)或(block_num, block_size, N, D)，H（Head-Size）表示隐藏层的大小，H = N * D。

参数query中的N和head_num值相等，key、value的N和key_value_head_num值相等，并且head_num是key_value_head_num的倍数关系。
参数query中的D和key的D(H/key_value_head_num)值相等。
query，key，value输入，功能使用限制如下：
  支持B轴小于等于3072；
  支持key/value的N轴（H/D）小于等于256；
  支持query的N轴与key/value的N轴（H/D）的比值小于等于16；
  支持query与key的D轴等于192；
  支持value的D轴等于128；
  支持query与key的block_size小于等于64或128；
  仅支持query的S轴等于1。
  仅支持paged attention。
  仅支持select_block_size取值为16的整数倍。
  selectBlockCount上限满足select_block_count * select_block_size <= MaxKvSeqlen，MaxKvSeqlen = Max(actual_seq_kvlen)。

支持的型号
Atlas A2训练系列产品

调用示例
>>> import torch
>>> import torch_npu
>>> query = torch.randn([1, 1, 768], dtype=torch.float16).npu()
>>> key = torch.randn([246, 64, 384], dtype=torch.float16).npu()
>>> value = torch.randn([246, 64, 256], dtype=torch.float16).npu()
>>> topk_indices = torch.tensor([[[0, -1], [0, -1]]], device="npu", dtype=torch.int32)
>>> block_table = torch.tensor([[1, 0]], device="npu", dtype=torch.int32)
>>> scale_value = 2.0
>>> head_num = 4
>>> key_value_head_num = 2
>>> select_block_size = 64
>>> select_block_count = 2
>>> page_block_size = 64
>>> layout = 'BSH'
>>> actual_seq_qlen = None
>>> actual_seq_kvlen = [82] * query.size(0)
>>> atten_mask = None
>>> torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, layout=layout, atten_mask=atten_mask, block_table=block_table, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
"""
)


_add_torch_npu_docstr(
    "npu_gather_sparse_index",
    """
接口原型: 
torch_npu.npu_gather_sparse_index(input, index) -> torch.Tensor

功能描述: 
从输入Tensor的指定维度dim，按照index中的下标序号提取元素，保存到out Tensor中。

参数说明: 
input(torch.Tensor): 输入张量，数据维度仅支持2维。
    在Atlas A2/Atlas A3上数据类型支持torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16,
    torch.int8, torch.uint8, torch.bool, torch.float64, torch.complex64, torch.complex128
index(torch.Tensor): 包含目标元素下标序号的张量。数据维度不超过8维。数据类型支持torch.int64, torch.int32。取值范围[0, input.shape[0] - 1], 不支持负数索引。

输出说明: 
out(torch.Tensor): 接口计算获得的结果，包含按照index中的下标序号提取的元素。数据类型与input一致，输出维度为index.dim + input.dim - 1。
    例如input.shape = [16, 32], index.shape = [2, 3]，则输出张量 out.shape = [2, 3, 32]

约束说明: 
1. input 的维度与 index 的维度之和减1不能超过8，即index.dim + input.dim - 1<=8。

支持版本: 
PyTorch 2.1
PyTorch 2.5及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
import torch
import torch_npu

inputs = torch.randn(16, 32).npu()
index = torch.randint(0, 16, [2, 3]).npu()
out = torch_npu.npu_gather_sparse_index(inputs, index)
"""
)


_add_torch_npu_docstr(
    "npu_moe_update_expert",
    """
torch_npu.npu_moe_update_expert(Tensor expert_ids, Tensor eplb_table, *, Tensor? expert_scales=None, Tensor? pruning_threshold=None, Tensor? active_mask=None, int local_rank_id=-1, int world_size=-1, int balance_mode=0) -> (Tensor, Tensor)

功能描述
完成冗余专家部署场景下每个token的topK个专家逻辑卡号到物理卡号的映射。
支持根据阈值对token发送的topK个专家进行剪枝

参数说明
    expert_ids：每个token的topK个专家索引，Device侧的Tensor，要求为一个2D的Tensor，shape为 (BS, K)。数据类型支持INT32，INT64，数据格式要求为ND，支持非连续的Tensor。
    eplb_table：逻辑专家到物理专家的映射表，外部调用者需保证输入Tensor的值正确：每行第一列为行号对应逻辑专家部署的实例数count，值需大于等于1，每行[1, count]列为对应实例的卡号，取值范围[0, moe_expert_num)，Device侧的Tensor，要求是一个2D的Tensor，shape为(moe_expert_num, F)。数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。其中F表示输入映射表的列数，第一列为各行号对应Moe专家部署的实例个数（值>0），后F-1列为该Moe专家部署的物理卡号，取值范围[2, world_size+1]。
    expert_scales：每个token的topK个专家的scale权重，用户需保证scale在token内部按照降序排列，可选择传入有效数据或空指针，该参数传入有效数据时，pruning_threshold也需要传入有效数据。Device侧的Tensor，要求是一个2D的Tensor，shape为 (BS, K)。数据类型支持FP16、BF16、FLOAT，数据格式要求为ND，支持非连续的Tensor。
    pruning_threshold：专家scale权重的最小阈值，当某个token对应的某个topK专家scale小于阈值时，该token将对该专家进行剪枝，即token不发送至该专家处理，可选择传入有效数据或空指针，该参数传入有效数据时，expert_scales也需要传入有效数据。Device侧的Tensor，要求是一个1D或2D的Tensor，shape为(K,)或(1, K)。数据类型支持FLOAT，数据格式要求为ND，支持非连续的Tensor。
    active_mask:表示token是否参与通信，可选择传入有效数据或空指针。传入有效数据时，expert_scales、pruning_threshold也必须传入有效数据，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true}为非法输入；传入空指针时是表示所有token都会参与通信。Device侧的Tensor，要求是一个1D的Tensor，shape为(BS,)。数据类型支持bool，数据格式要求为ND，支持非连续的Tensor。
    local_rank_id：本卡ID，数据类型支持INT64，当balance_mode设置0时，本属性取值范围为[0, world_ize)。
    world_size：通信域size，数据类型支持INT64，当balance_mode设置0时，本属性取值范围为[2, 768]
    balance_mode：均衡规则，数据类型支持INT64，取值支持0和1，0表示用local_rank_id进行负载均衡，1表示使用token_id进行负载均衡。当本属性取值为0时，local_rank_id和world_size必须传入有效值。
    
输出说明
    balanced_expert_ids：映射后每个token的topK个专家所在物理卡的卡号，Device侧的Tensor，要求是一个2D的Tensor，shape为(BS, K)，数据类型、数据格式与expert_ids保持一致。
    balanced_active_mask：剪枝后的active_mask，当expert_scales、pruning_threshold传入有效数据时该输出有效。Device侧的Tensor，要求是一个2的Tensor，shape为(BS, K)，数据类型支持BOOL，数据格式要求为ND，支持非连续的Tensor。

支持的型号
Atlas A3训练系列产品

调用示例
import os
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMoeUpdateExpert(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bs = 128
        self.k = 8
        self.log_ep_size = 256
        self.pyh_ep_size = 8
        self.F = 5
        self.is_pruning = True
        self.world_size = 8
        self.balance_mode = 0
        self.expert_ids = []
        self.eplb_table = []
        self.expert_scales = []
        self.pruning_threshold = []
        self.active_mask = []
        self.balanced_expert_ids = []
        self.balanced_active_mask = []
        self.gen_exp_result()

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_moe_update_expert(cls, rank_id, input_list):
        expert_ids, eplb_table, world_size, expert_scales, pruning_threshold, active_mask, balance_mode, init_pg, c2p, p2c = input_list
        _ = init_pg(rank_id, world_size)
        out_expert_idx, out_mask = torch_npu.npu_moe_update_expert(expert_ids=expert_ids.npu(),
                                                   eplb_table=eplb_table.npu(),
                                                   local_rank_id=rank_id,
                                                   world_size=world_size,
                                                   expert_scales=expert_scales.npu(),
                                                   pruning_threshold=pruning_threshold.npu(),
                                                   active_mask=active_mask.npu(),
                                                   balance_mode=balance_mode)
        c2p.put((rank_id, out_expert_idx.cpu(), out_mask.cpu()))
        p2c.get()
    
    def gen_exp_result(self):
        for rank_id in range(self.world_size):
            eplb_table = np.zeros((self.log_ep_size, self.F - 1))
            count_column = np.random.randint(1, self.F, size=(self.log_ep_size, 1))
            all_ranks = np.arange(self.pyh_ep_size)
            for i in range(self.log_ep_size):
                np.random.shuffle(all_ranks)
                for j in range(count_column[i][0]):
                    eplb_table[i][j] = all_ranks[j]
            eplb_table = np.hstack((count_column, eplb_table))

            expert_ids = np.random.randint(low=0, high=self.log_ep_size, size=(self.bs, self.k))
            if self.is_pruning:
                expert_scales = -np.sort(-np.random.uniform(low=0, high=0.25, size=(self.bs, self.k)), axis=1)
                pruning_threshold = np.random.uniform(low=0, high=0.15, size=(1, self.k))
                num_true = np.random.randint(0, self.bs + 1)
                active_mask = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(self.bs - num_true, dtype=bool)])
            eplb_table_tensor = torch.from_numpy(eplb_table).to(torch.int32)
            self.eplb_table.append(eplb_table_tensor)
            expert_ids_tensor = torch.from_numpy(expert_ids).to(torch.int32)
            self.expert_ids.append(expert_ids_tensor)
            if self.is_pruning:
                expert_scales_tensor = torch.from_numpy(expert_scales).to(torch.float32)
                self.expert_scales.append(expert_scales_tensor)
                pruning_threshold_tensor = torch.from_numpy(pruning_threshold).to(torch.float32)
                self.pruning_threshold.append(pruning_threshold_tensor)
                active_mask_tensor = torch.from_numpy(active_mask).to(torch.bool)
                self.active_mask.append(active_mask_tensor)

            balanced_expert_ids = np.zeros((self.bs, self.k))
            if self.is_pruning:
                balanced_active_mask = np.zeros((self.bs, self.k))

            for i in range(self.bs):
                for j in range(self.k):
                    log_ep_id = expert_ids_tensor[i][j]
                    if self.balance_mode == 0:
                        mod_val = math.ceil(self.world_size / eplb_table_tensor[log_ep_id][0].item())
                        phy_ep_id = eplb_table_tensor[log_ep_id][(rank_id // mod_val) + 1]
                        balanced_expert_ids[i][j] = phy_ep_id
                    if self.balance_mode == 1:
                        phy_ep_id = eplb_table_tensor[log_ep_id][(i % eplb_table_tensor[log_ep_id][0].item()) + 1]
                        balanced_expert_ids[i][j] = phy_ep_id
                    if self.is_pruning:
                        if not active_mask_tensor[i]:
                            balanced_active_mask[i][j] = 0
                        else:
                            if expert_scales_tensor[i][j] < pruning_threshold_tensor[0][j] * sum(expert_scales_tensor[i]):
                                balanced_active_mask[i][j] = 0
                            else:
                                balanced_active_mask[i][j] = 1
            self.balanced_expert_ids.append(torch.from_numpy(balanced_expert_ids).to(torch.int64))
            self.balanced_active_mask.append(torch.from_numpy(balanced_active_mask).to(torch.bool))        


    @SupportedDevices(['Ascend910_'])
    def test_npu_moe_update_expert(self):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(self.world_size)
        p2c = ctx.Queue(self.world_size)
        ps = []

        for rank_id in range(self.world_size):
            p = ctx.Process(
                target=self._test_npu_moe_update_expert,
                args=(rank_id, [self.expert_ids[rank_id], self.eplb_table[rank_id], self.world_size, 
                                self.expert_scales[rank_id], self.pruning_threshold[rank_id], self.active_mask[rank_id],
                                self.balance_mode, self._init_dist_hccl, c2p, p2c]))
            p.start()
            ps.append(p)

        for _ in range(self.world_size):
            rank_id, output_0, output_1 = c2p.get()
            self.assertEqual(output_0, self.balanced_expert_ids[rank_id],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, self.balanced_expert_ids[rank_id], output_0))
            self.assertEqual(output_1, self.balanced_active_mask[rank_id],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank_id, self.balanced_active_mask[rank_id], output_1))

        for _ in range(self.world_size):
            p2c.put(0)

        for p in ps:
            p.join()


if __name__ == '__main__':
    run_tests()
"""
)


_add_torch_npu_docstr(
    "npu_top_k_top_p",
    """
接口原型: 
torch_npu.npu_top_k_top_p(logits, p, k) -> torch.Tensor

功能描述: 
对原始输入logits进行top-k和top-p采样过滤

计算公式：
    1. 对输入logits按最后一轴进行升序排序，得到对应的排序结果sortedValue和sortedIndices。
    sortedValue, sortedIndices = sort(logits, dim=-1, descend=false, stable=true)

    2. 计算保留的阈值（第k大的值）。
    topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]

    3. 生成top-k需要过滤的mask。
    topKMask = sortedValue < topKValue

    4. 通过topKMask将小于阈值的部分置为-inf。
    sortedValue[b][v] = 
        -inf if topKMask[b][v] == true else sortedValue[b][v]

    5. 通过softmax将经过top-k过滤后的数据按最后一轴转换为概率分布。
    probsValue = softmax(sortedValue, dim=-1)

    6. 按最后一轴计算累计概率（从最小的概率开始累加）。
    probsSum = cumsum(probsValue, dim=-1)

    7. 生成top-p的mask，累计概率小于等于1-p的位置需要过滤掉，并保证每个batch至少保留一个元素。
    topPMask[b][v] = probsSum[b][v] <= 1-p[b]
    topPMask[b][-1] = false

    8. 通过topPMask将小于阈值的部分置为-inf。
    sortedValue[b][v] = 
        -inf if topPMask[b][v] == true else sortedValue[b][v]

    9. 将过滤后的结果按sortedIndices还原到原始顺序。
    out[b][v] = sortedValue[b][sortedIndices[b][v]]

    其中 0 <= b < logits.size(0), 0 <= v < logits.size(1)。

参数说明: 
logits(torch.Tensor): 输入张量，支持2维，数据类型支持torch.bfloat16, torch.float16, torch.float32。
p(torch.Tensor): 可选张量，默认值为None，不支持p和k同时传None。表示top-p的阈值，值域为[0, 1]，数据类型支持torch.bfloat16, torch.float16, torch.float32，数据类型需要与logits一致，shape支持1维且需要与logits的首轴相同，支持非连续Tensor，支持空tensor，支持ND
k(torch.Tensor): 可选张量，默认值为None，不支持p和k同时传None。表示top-k的阈值，值域为[1, 1024]，且最大值需要小于等于logits.size(1)，数据类型支持torch.int32，shape支持1维且需要与logits的首轴相同，支持非连续Tensor，支持空tensor，支持ND

输出说明: 
out(torch.Tensor): 表示过滤后的数据。数据类型支持torch.bfloat16, torch.float16, torch.float32，数据类型需要与logits一致，shape支持2维且需要与logits一致，支持非连续Tensor，数据格式支持ND

约束说明: 
无

支持版本: 
PyTorch 2.1
PyTorch 2.5及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
import torch
import torch_npu

logits = torch.randn(16, 2048).npu()
p = torch.rand(16).npu()
k = torch.randint(10, 1024, (16,)).npu().to(torch.int32)
out = torch_npu.npu_top_k_top_p(logits, p, k)
"""
)


_add_torch_npu_docstr(
    "ffn_worker_scheduler_",
    """
接口原型: 
ffn_worker_scheduler_(Tensor(a!) self, *, int sync_group_size=1, int execute_mode=0) -> Tensor(a!)

功能描述: 
Attention和Ffn分离部署场景下，Ffn侧数据扫描功能，扫描并原地完成数据整理。

参数说明: 
scheduler_context(torch.Tensor): 输入张量, scheduler_context的定义与生成可参照torch_npu._afd包；
sync_group_size(int): 可选，默认值为1；
execute_mode(int): 可选，默认值为0。

约束说明: 
无

支持版本: 
PyTorch 2.1及更高版本

支持的型号: 
Atlas A3训练系列产品
Atlas A3推理系列产品

调用示例: 
import torch
import torch_npu
import os

window_size = 209715200
ffn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 2
micro_batch_number = 3
batch_size = 6
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
ffn_window = ffn_window_tensor.data_ptr()

context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode = 0, session_num = attn_workers,
    micro_batch_num = micro_batch_number, micro_batch_size = batch_size, selected_expert_num = top_k + 1,
    expert_num = expert_num, attn_to_ffn_token_size = attn_to_ffn_token_size, ffn_to_attn_token_size = ffn_to_attn_token_size, 
    ffn_window = ffn_window, ffn_window_size = window_size)
    
schedule_context = context_holder.get_schedule_context_tensor()


def _set_all_flags():
    num_int8 = attn_workers * micro_batch_number * (8 + batch_size * top_k * 4)
    per_session_num = micro_batch_number * (8 + batch_size * top_k * 4)
    int32_view = ffn_window_tensor[:num_int8].view(torch.int32)
    int32_view[:] = 1


_set_all_flags()
torch_npu.ffn_worker_scheduler_(schedule_context, sync_group_size = 2)
"""
)


_add_torch_npu_docstr(
    "ffn_worker_scheduler",
    """
接口原型: 
ffn_worker_scheduler(Tensor self, *, int sync_group_size=1, int execute_mode=0) -> Tensor

功能描述: 
Attention和Ffn分离部署场景下，Ffn侧数据扫描功能，扫描并完成数据整理输出。

参数说明: 
scheduler_context(torch.Tensor): 输入张量, scheduler_context的定义与生成可参照torch_npu._afd包；
sync_group_size(int): 可选，默认值为1；
execute_mode(int): 可选，默认值为0。

输出说明: 
scheduler_context(torch.Tensor): 输出结果张量

约束说明: 
无

支持版本: 
PyTorch 2.1及更高版本

支持的型号: 
Atlas A3训练系列产品
Atlas A3推理系列产品

调用示例: 
import torch
import torch_npu
import os

window_size = 209715200
ffn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 2
micro_batch_number = 3
batch_size = 6
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
ffn_window = ffn_window_tensor.data_ptr()

context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode = 0, session_num = attn_workers,
    micro_batch_num = micro_batch_number, micro_batch_size = batch_size, selected_expert_num = top_k + 1,
    expert_num = expert_num, attn_to_ffn_token_size = attn_to_ffn_token_size, ffn_to_attn_token_size = ffn_to_attn_token_size, 
    ffn_window = ffn_window, ffn_window_size = window_size)
    
schedule_context = context_holder.get_schedule_context_tensor()


def _set_all_flags():
    num_int8 = attn_workers * micro_batch_number * (8 + batch_size * top_k * 4)
    per_session_num = micro_batch_number * (8 + batch_size * top_k * 4)
    int32_view = ffn_window_tensor[:num_int8].view(torch.int32)
    int32_view[:] = 1


_set_all_flags()
schedule_context_out = torch_npu.ffn_worker_scheduler(schedule_context, sync_group_size = 2)
"""
)


_add_torch_npu_docstr(
    "attention_worker_scheduler_",
    """
接口原型: 
attention_worker_scheduler_(Tensor(a!) self) -> Tensor(a!)

功能描述: 
Attention和Ffn分离部署场景下，Attention侧数据扫描功能，扫描并原地确保数据就绪。

参数说明: 
scheduler_context(torch.Tensor): 输入张量, scheduler_context的定义与生成可参照torch_npu._afd包。

约束说明: 
无

支持版本: 
PyTorch 2.1及更高版本

支持的型号: 
Atlas A3训练系列产品
Atlas A3推理系列产品

调用示例: 
import torch
import torch_npu
import os

window_size = 209715200
attn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 144
micro_batch_number = 3
batch_size = 30
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
attn_window = attn_window_tensor.data_ptr()

context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode = 1, session_num = attn_workers,
    micro_batch_num = micro_batch_number, micro_batch_size = batch_size, selected_expert_num = top_k + 1,
    expert_num = expert_num, attn_to_ffn_token_size = attn_to_ffn_token_size, ffn_to_attn_token_size = ffn_to_attn_token_size, 
    attention_window = attn_window, attention_window_size = window_size)
    
schedule_context = context_holder.get_schedule_context_tensor()

def _set_all_flags():
    num_int8 = batch_size * (top_k + 1) * 4 * micro_batch_number

    int32_view = attn_window_tensor[:num_int8].view(torch.int32)

    int32_view[:] = 1


_set_all_flags()
torch_npu.attention_worker_scheduler_(schedule_context)

"""
)


_add_torch_npu_docstr(
    "attention_worker_scheduler",
    """
接口原型: 
attention_worker_scheduler(Tensor self) -> Tensor

功能描述: 
Attention和Ffn分离部署场景下，Attention侧数据扫描功能，扫描并确保数据就绪。

参数说明: 
scheduler_context(torch.Tensor): 输入张量, scheduler_context的定义与生成可参照torch_npu._afd包。

输出说明: 
scheduler_context(torch.Tensor): 输出结果张量

约束说明: 
无

支持版本: 
PyTorch 2.1及更高版本

支持的型号: 
Atlas A3训练系列产品
Atlas A3推理系列产品

调用示例: 
import torch
import torch_npu
import os

window_size = 209715200
attn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 144
micro_batch_number = 3
batch_size = 30
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
attn_window = attn_window_tensor.data_ptr()

context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode = 1, session_num = attn_workers,
    micro_batch_num = micro_batch_number, micro_batch_size = batch_size, selected_expert_num = top_k + 1,
    expert_num = expert_num, attn_to_ffn_token_size = attn_to_ffn_token_size, ffn_to_attn_token_size = ffn_to_attn_token_size, 
    attention_window = attn_window, attention_window_size = window_size)
    
schedule_context = context_holder.get_schedule_context_tensor()

def _set_all_flags():
    num_int8 = batch_size * (top_k + 1) * 4 * micro_batch_number

    int32_view = attn_window_tensor[:num_int8].view(torch.int32)

    int32_view[:] = 1


_set_all_flags()
schedule_context_out = torch_npu.attention_worker_scheduler(schedule_context)

"""
)


_add_torch_npu_docstr(
    "npu_top_k_top_p_sample",
    """
接口原型: 
torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q=None, eps=1e-8, is_need_logits=False, top_k_guess=32) -> (torch.Tensor, torch.Tensor)

功能描述: 
根据输入词频logits、top_k/top_p采样参数、随机采样权重分布q，进行topK-topP-Sample采样计算，输出每个batch的最大词频logits_select_idx，以及topK-topP采样后的词频分布logits_top_kp_select。
算子包含三个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：topK采样、topP采样、指数采样（Sample）。它们可以构成八种计算场景。如下表所示：
  | 计算场景 | topK采样 | topP采样 | 指数分布采样 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|无|
  |topP采样|×|√|×|无|
  |Sample采样|×|×|√|无|
  |topK-topP采样|√|√|×|无|
  |topK-Sample采样|√|×|√|无|
  |topP-Sample采样|×|√|√|无|
  |topK-topP-Sample采样|√|√|√|无|

计算公式：
输入logits为大小是[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logtis中的每一行logtis[batch][:]根据相应的top_k[batch]、top_p[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。
    topK采样
    1.按分段长度v采用分段topK归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
    2.top_k[batch]对应当前batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch],1024)，如果top_k[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。
    具体计算流程如下所示：
    对当前batch分割为若干子段，滚动计算top_k_value[b]：
    top_k_value[b]={Max(top_k[b])}_{s=1}^{lceilfrac{S}{v}rceil}{top_k_value[b]{s-1}cup{logits[b][v]getop_k_min[b][s-1]}}
    Card(top_k_value[b])=top_k[b]
    其中：
    top_k_min[b][s]=Min(top_k_value[b]{s})
    v表示预设的滚动topK时固定的分段长度：
    v=8*1024
    *生成需要过滤的mask：
    sorted_value[b]=sort(top_k_value[b],descendant)
    top_k_mask[b]=sorted_value[b]<Min(top_k_value[b])

    *将小于阈值的部分通过mask置为-inf：
    sorted_value[b][v]=begin{cases} -inf&text{top_k_mask[b][v]=true}
    sorted_value[b][v]&text{top_k_mask[b][v]=false}&
    end{cases}

    通过softmax将经过topK过滤后的logits按最后一轴转换为概率分布：
    probs_value[b]=sorted_value[b].softmax(dim=-1)

    按最后一轴计算累积概率（从最小的概率开始累加）：
    probs_sum[b]=probs_value[b].cumcum(dim=-1)

    topP采样
    如果前序topK采样已有排序输出结果，则根据topK采样输出计算累积词频，并根据topP截断采样：
    top_p_mask[b]=probs_sum[b][*]<top_p[b]

    如果topK采样被跳过，则先对输入logits[b]进行softmax处理：
    logits_value[b]=logits[b].softmax(dim=-1)

    尝试使用top_k_guess，对logits进行滚动排序，获取计算topP采样的mask：
    top_p_value[b]={Max(top_k_guess)}_{s=1}^{lceilfrac{S}{v}rceil}{top_p_value[b]{s-1}cup{logits_value[b][v]getop_k_min[b][s-1]}}

    如果在访问到logits_value[b]的第1e4个元素之前，如下条件得到满足，则视为top_k_guess成功：
    sum^{top_k_guess}(top_p_value[b])getop_p[b]
    top_p_mask[b][Index(top_p_value[b])]=false

    如果top_k_guess失败，则对当前序logits_value[b]进行全排序和cumsum，按top_p[b]截断采样：
    sorted_logits[b]=sort(logits_value[b],descendant)
    probs_sum[b]=sorted_logits[b].cumcum(dim=-1)
    top_p_mask[b]=(probs_sum[b]-sorted_logits[b])>top_p[b]

    将需要过滤的位置设置为-inf，得到sorted_value[b][v]：
    sorted_value[b][v]=begin{cases}-inf&text{top_p_mask[b][v]=true}sorted_value[b][v]&text{top_p_mask[b][v]=false}end{cases}

    取过滤后sorted_value[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为logits_idx:
    logits_idx[b][v]=Index(sorted_value[b][v]inlogits)

    指数采样（Sample）
    如果is_need_logits设置为True，则根据logits_idx，选取采样后结果输出到logits_top_kp_select：
    logits_top_kp_select[b][logits_idx[b][v]]=sorted_value[b][v]

    对sorted_value进行指数分布采样：
    probs=softmax(sorted_value)
    probs_opt=frac{probs}{q+eps}

    从probs_opt中取出每个batch的最大元素，从logits_idx中gather相应元素的输入索引，作为输出logits_select_idx：
    logits_select_idx[b]=logits_idx[b][argmax(probs_opt[b][:])]
    其中0≤b<sorted_value.size(0)，0≤v<sorted_value.size(1)。

参数说明:
logits（Tensor）：必选参数，表示待采样的输入词频，目前支持2维，词频索引固定为最后一维。数据类型支持float16和bfloat16，数据格式支持ND，支持非连续Tensor。
top_k（Tensor）：必选参数，表示每个batch采样的k值，有效范围为1≤top_k[batch]≤min(voc_size[batch], 1024)，目前支持1维。数据类型支持int32,数据格式支持ND，支持非连续Tensor。
top_p（Tensor）：必选参数，表示每个batch采样的p值，有效范围为0<top_p[batch]<1，目前支持1维。数据类型和数据格式与logits保持一致，支持非连续Tensor。
q（Tensor）：可选参数，topK-topP采样输出的随机采样权重分布矩阵，数据类型支持float32，数据格式支持ND，支持非连续Tensor，默认值为None。
eps（float）：可选参数，在softmax和权重采样中防止除零，默认值1e-8。
is_need_logits（bool）：可选参数，控制logits_top_kp_select的输出条件，默认为False。
top_k_guess（int）：可选参数，表示每个batch在尝试topP部分遍历采样时的候选logits大小，必须为正整数，默认值为32。

输出说明: 
logits_select_idx（Tensor）：表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probs_opt[batch, :])在输入logits中的位置索引。数据类型支持int64，数据格式支持ND。
logits_top_kp_select（Tensor）：表示经过topK-topP计算流程后，输入logtis中剩余未被过滤的logits。数据类型支持float32，数据格式支持ND。

约束说明:
该接口支持推理场景下使用。
该接口目前不支持图模式。
logits、q、logits_top_kp_select的尺寸和维度必须完全一致。
logits、top_k、top_p、logits_select_idx除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，top_k、top_p、logits_select_idx必须是1维非空Tensor。logits、top_k、top_p不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
如果1024<top_k[batch]<voc_size[batch]，则视为选择当前batch的全部有效元素并跳过topK环节。
如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
如果需要单独跳过sample模块，使用其默认值或设置q为None；如需使用sample模块，则必须传入尺寸为[batch, voc_size]的Tensor。

支持的型号: 
Atlas A3 训练系列产品
Atlas A3 推理系列产品
Atlas A2 训练系列产品
Atlas 800I A2 推理产品
A200I A2 Box 异构组件

调用示例: 
>>> logits = torch.from_numpy(np.random.uniform(-2, 2, size=[2, 4])).type(torch.float16).npu()
>>> top_ks = torch.from_numpy(np.random.uniform(1, 2, size=[2, ])).type(torch.int32).npu()
>>> top_ps = torch.from_numpy(np.random.uniform(0.4, 0.5, size=[2, ])).type(torch.float16).npu()
>>> q = torch.from_numpy(np.random.uniform(0.1, 0.5, size=[2, 4])).type(torch.float32).npu()

>>> npu_out_index, logits_top_kp_select = torch_npu.npu_top_k_top_p_sample(logits, top_ks, top_ps, q, 1E-8, True)
>>> print(npu_out_index)
>>> print(logits_top_kp_select)
"""
)


_add_torch_npu_docstr(
    "npu_moe_token_permute",
    """
接口原型: 
torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False) -> (Tensor, Tensor)

功能描述
MoE的permute计算，根据索引indices将tokens广播并排序。


参数说明: 
tokens(torch.Tensor)：必选输入，2维Tensor, shape为(num_tokens，hidden_size），数据类型torch.bfloat16，支持非连续Tensor，支持ND
indices(torch.Tensor): 必选输入，2维Tensor，shape为（num_tokens，topK），数据类型torch.int64，支持非连续Tensor，支持ND
num_out_tokens(int, optional)：可选输入，默认为None，数据类型int64，表示有效输出token数。设置为0时，表示不会删除任何token。不为0时，会按照num_tokens进行切片丢弃按照indices排序好的token中超过num_tokens的部分，为负数时按照切片索引为负数时处理。
padded_mode(bool, optional): 可选输入，默认为False，如果为True，表示indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序，目前仅支持为False

输出说明: 
permuted_tokens(torch.Tensor)：2维Tensor，数据类型torch.bfloat16(当前版本permuted_tokens仅支持bfloat16)
sorted_indices(torch.Tensor)：1维Tensor，数据类型torch.int32(当前版本sorted_indices仅支持int32)

约束说明: 
indices 要求元素个数小于16777215，值大于等于0小于16777215(单点支持int32或int64的最大或最小值，其余值不在范围内排序结果不正确)
topK小于等于512

支持版本: 
PyTorch 2.1
PyTorch 2.5及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
import torch
import torch_npu

dtype = torch.bfloat16
tokens = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).npu().to(dtype)
indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
num_out_tokens = indices.numel()
probs = torch.ones_like(indices) / 2
probs = probs.npu().to(dtype)
permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens)
"""
)


_add_torch_npu_docstr(
    "npu_moe_token_unpermute",
    """
接口原型: 
torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None) -> Tensor


功能描述
根据sorted_indices存储的下标，获取permuted_tokens中存储的输入数据；如果存在probs数据，permuted_tokens会与probs相乘；最后进行累加求和，并输出计算结果

参数说明: 
permuted_tokens(torch.Tensor)：必选输入，2维Tensor, shape为(num_tokens*topK，hidden_size），数据类型torch.bfloat16，支持非连续Tensor，支持ND
sorted_indices(torch.Tensor): 必选输入，1维Tensor，shape为（num_tokens*topK），数据类型torch.int64，支持非连续Tensor，支持ND
probs(torch.Tensor, optional)：可选输入，默认为None，当probs传时，topK等于probs的第二维；当probs不传时，topK=1。shape为（num_tokens，topK），支持的数据类型BFLOAT16。数据格式支持ND，支持非连续输入
padded_mode(bool, optional): 可选输入，默认为False，数据类型int64，目前仅支持为False
restore_shape(torch.size, optional): 可选输入，默认为None，表示permute前输入的shape，只在padded_mode为True时生效。数据类型torch.size

输出说明: 
unpermuted_tokens(torch.Tensor)：2维Tensor，数据类型torch.bfloat16，padded_mode=False时，shape为(num_tokens，hidden_size)

约束说明:
目前仅支持padded_mode为False

支持版本: 
PyTorch 2.1
PyTorch 2.5及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
import torch
import torch_npu

dtype = torch.bfloat16
permuted_tokens = torch.tensor([[1., 1., 1.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [3., 3., 3.],
                                    [2., 2., 2.],
                                    [1., 1., 1.],
                                    [2., 2., 2.],
                                    [3., 3., 3.]]).npu().to(dtype)
sorted_indices = torch.tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=torch.int32).npu()
indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
probs = torch.ones_like(indices) / 2

unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)
"""
)


_add_torch_npu_docstr(
    "npu_dynamic_block_quant",
    """
接口原型: 
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128) -> (Tensor, Tensor)


功能描述
对输入张量，通过给定的`row_block_size`和`col_block_size`将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数`scale`，并根据`scale`对输入进行量化。输出最终的量化结果，以及每个块的量化参数`scale`。

参数说明: 
x (Tensor)：必选参数，输入张量，数据类型支持float16、bfloat16，支持非连续的Tensor，数据格式支持ND。当前shape支持2维和3维。
min_scale (float)：可选参数，参与scale计算的最小scale值。当前仅支持取值0。
round_mode (str)：可选参数，指定cast到输出的转换方式。当前仅支持取值rint。
dst_type (int)：可选参数，指定输出y的数据类型。当前仅支持取值1，表示代码输出y的数据类型为int8。
row_block_size (int)：可选参数，指定一个block的行大小。当前仅支持取值1。
col_block_size (int)：可选参数，指定一个block的列大小，当前仅支持取值128。

输出说明: 
y (Tensor)：量化结果。
scale (Tensor)：量化时使用的量化参数。

支持版本: 
PyTorch 2.1
PyTorch 2.5及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
>>> import torch
>>> import torch_npu

>>> x = torch.rand(3, 4).to("npu").to(torch.float16)
>>> min_scale = 0
>>> dst_type = 1
>>> row_block_size = 1
>>> col_block_size = 128

>>> y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)
>>> y
tensor([[ 92,  65,  15, 127],
      [100, 127, 116,  64],
      [ 95,  15,  87, 127]], device='npu:0', dtype=torch.int8)
>>> scale
tensor([[0.0063],
      [0.0076],
      [0.0073]], device='npu:0')
"""
)


_add_torch_npu_docstr(
    "obfuscation_initialize",
    """
功能描述: 
该接口用于完成PMCC（Privacy&Model Confidential Computing）模型混淆引擎的资源初始化，即与PMCC混淆引擎CA（普通OS中的Client Application）建立socket连接、对CA、TA（TEE OS中的Trusted Application）进行初始化，并返回socket连接符。

接口原型: 
torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type, model_obf_seed_id, data_obf_seed_id, thread_num, obf_coefficient) -> Tensor

参数说明
- hidden_size（`int`）：必选参数，隐藏层的维度，数据类型为`int32`，支持输入范围为1-10000，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- tp_rank（`int`）：必选参数， 张量并行TP Rank，数据类型为`int32`，支持输入范围为0-1024，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- cmd（`int`）：必选参数，资源初始化的指令编号，数据类型为`int32`，取值范围为{1, 2, 3}。
    * 1：进行浮点推理模式资源初始化。
    * 2：进行量化推理模式资源初始化。
    * 3：进行资源释放。
- data_type（`int`）：可选参数， 代表Tensor数据类型的编号，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
    * <term>Atlas 推理系列产品</term>: Tensor数据类型支持`float16` 、`float32`、`int8`。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: Tensor数据类型支持`float16`、`float32`、`bfloat16`、`int8`。
- model_obf_seed_id（`int`）：可选参数， 模型混淆因子id，用于`TA`从`TEE KMC`查询模型混淆因子，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写已注册的有效混淆因子id，否则填0。
- data_obf_seed_id（`int`）：必选参数， 数据混淆因子id，用于`TA`从`TEE KMC`查询数据混淆因子，数据类型为`int32`，仅在`cmd`设置为1或2时需要填写已注册的有效混淆因子id，否则填0。
- thread_num（`int`）：可选参数， `CA`/`TA`进行混淆处理使用的线程数，数据类型为`int32`，取值范围为{1, 2, 3, 4, 5, 6}，仅在`cmd`设置为1或2时需要填写有效值，否则填0。
- obf_coefficient（`float`）：可选参数，混淆系数，支持输入范围为0-1，默认值1.0。

输出说明:
`Tensor`
代表socket连接符，1D，shape为(1)，数据类型为`int32`。

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
Atlas 推理系列产品

调用示例:
import torch
import torch_npu

device = "npu:0"
hidden_size = int(3584)
cmd = 1
data_type = torch.bfloat16
model_obf_seed = 0
data_obf_seed = 0
thread_num = 4
tp_rank = 0
i = 0
hidden_states = torch.randn((1024,3584), dtype=torch.bfloat16, device=device)
obf_cft = 1.0
fd = torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
"""
)


_add_torch_npu_docstr(
    "obfuscation_finalize",
    """
功能描述: 
该接口用于完成PMCC（Privacy&Model Confidential Computing）模型混淆引擎的资源释放，即与PMCC混淆引擎CA（普通OS中的Client Application）断开socket连接。

接口原型: 
torch_npu.npu.obfuscation_finalize(fd_to_close) -> Tensor

参数说明: 
fd_to_close（`Tensor`）：填写调用[obfuscation_initialize](./torch_npu-npu-obfuscation_initialize.md)接口的返回值，数据类型为`int32`。

输出说明:
`Tensor`
代表socket连接符，1D，shape为(1)，数据类型为`int32`。

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
Atlas 推理系列产品

调用示例:
import torch
import torch_npu

device = "npu:0"
hidden_size = int(3584)
cmd = 1
data_type = torch.bfloat16
model_obf_seed = 0
data_obf_seed = 0
thread_num = 4
tp_rank = 0
i = 0
hidden_states = torch.randn((1024,3584), dtype=torch.bfloat16, device=device)
obf_cft = 1.0
fd = torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
torch_npu.npu.obfuscation_finalize(fd)
"""
)


_add_torch_npu_docstr(
    "obfuscation_calculate",
    """
功能描述: 
该接口用于将张量x和配置参数（如param）发送至PMCC（Privacy&Model Confidential Computing）混淆引擎。引擎的CA（普通OS中的Client Application）模块调用TA（TEE OS中的Trusted Application）模块，进行张量混淆处理，最终返回混淆结果。

接口原型: 
torch_npu.npu.obfuscation_calculate(fd, x, param, obf_coefficient) -> Tensor

参数说明: 
- fd（`Tensor`）：必选参数，socket连接符，数据类型为`int32`，填写调用[obfuscation_initialize](./torch_npu-npu-obfuscation_initialize.md)接口的返回值。
- x（`Tensor`）：必选参数，待混淆处理的`Tensor`输入，对`Tensor`维度不作限制，shape为( , *, ... , hiddenSize)，即最后一维的size是[obfuscation_initialize](./torch_npu-npu-obfuscation_initialize.md)的入参`hiddenSize`。数据格式支持ND。
    * <term>Atlas 推理系列产品</term>: `Tensor`数据类型支持`float16` 、`float32`、`int8`。
    * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: `Tensor`数据类型支持`float16`、`float32`、`bfloat16`、`int8`。
- param（`Tensor`）：必选参数，张量`x`的最后一维的维度，数据类型为`int32`。
- obf_coefficient（`float`）：可选参数，混淆系数，支持输入范围为(0.0，1.0]，默认值1.0。

输出说明:
`Tensor`
代表`obfuscation_calculate`的计算结果，输出数据类型及shape与`x`相同。

支持的芯片型号:
Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
Atlas 推理系列产品

调用示例:
import torch
import torch_npu

device = "npu:0"
hidden_size = int(3584)
cmd = 1
data_type = torch.bfloat16
model_obf_seed = 0
data_obf_seed = 0
thread_num = 4
tp_rank = 0
i = 0
hidden_states = torch.randn((1024,3584), dtype=torch.bfloat16, device=device)
obf_cft = 1.0
fd = torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
param = torch.tensor([3584], device=device)
x_obf_out = torch_npu.npu.obfuscation_calculate(fd, hidden_states, param, obf_coefficient=obf_cft)
"""
)


_add_torch_npu_docstr(
    "npu_gelu_mul",
    """
接口原型: 
torch_npu.npu_gelu_mul(input, *, approximate='none') -> Tensor


功能描述:
将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Gelu计算，将计算结果与x2相乘。

计算公式：
给定输入张量 input，最后一维的长度为 2d，函数 GeluMul 进行以下计算：
(1)将 input 分割为两部分：x₁ = input[...,:d], x₂ = input[...,d:]
(2)对 x1 应用 GELU 激活函数，"tanh"模式公式如下：GELU(x) = 0.5 * x * [1 + tanh( √(2/π) * (x + 0.044715 * x³) )]
                            “none”对应的erf模式公式如下：GELU(x)= 0.5 * x * [1 + erf( x / √2 )]
                            因此，计算：x₁ = GELU(x₁)
(3)最终输出是 x₁ 和 x₂ 的逐元素乘积：out = x₁ * x₂

参数说明: 
input (Tensor类型)：必选参数，输入张量，数据类型支持BFLOAT16、FLOAT16、FLOAT。支持非连续的Tensor，数据格式支持ND，shape维度2至8维，
                    且shape满足如下要求：(1)最后一维值为偶数且小于等于1024。(2)其他维度的乘积小于等于200000。
approximate(String类型)：可选参数，计算输入, Gelu计算的模式，只支持“none”和“tanh”，分别对应Gelu的erf模式和tanh模式，默认值为“none”。

输出说明: 
out (Tensor)：输出张量，数据类型支持BFLOAT16、FLOAT16、FLOAT。shape维度2至8维。支持非连续的Tensor，数据格式支持ND，输出的数据类型与输入保持一致，输出shape和输入shape其他维度一致，最后一维的值为输入shape最后一维值的二分之一。

约束说明
典型场景尾轴为16的倍数，当尾轴为非32B对齐时，建议走小算子拼接逻辑。

支持版本: 
PyTorch 2.6及更高版本

支持的型号: 
Atlas A2训练系列产品
Atlas A3训练系列产品

调用示例: 
import torch
import torch_npu

shape = [100, 400]
mode = "none"
input = torch.rand(shape, dtype=torch.float16).npu()
output = torch_npu.npu_gelu_mul(input, approximate=mode)

"""
)
