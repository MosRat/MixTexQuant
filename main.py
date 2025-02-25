import onnx
import onnxruntime
from onnx import helper, numpy_helper

model_path = "models/outputs/decoder_model_q8.onnx"

model= onnx.load_model(model_path)

# 获取模型的graph
graph = model.graph

# 创建一个新的输入 (use_cache_branch)
use_cache_branch_input = helper.make_tensor_value_info(
    'use_cache_branch',  # 输入名称
    onnx.TensorProto.BOOL,  # 数据类型 (布尔类型)
    [1]  # 输入形状 (1维标量)
)

# 将新的输入添加到模型的输入列表中
graph.input.append(use_cache_branch_input)

# 保存修改后的模型
new_model_path = 'models/outputs/decoder_model_q8_cache.onnx'
onnx.save(model, new_model_path)


# print(*(i.name for i in model.graph.input))
#
# session =  onnxruntime.InferenceSession(model_path)
# print(*(i.name for i in session.get_inputs()))