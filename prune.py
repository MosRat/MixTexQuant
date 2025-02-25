import onnx
from onnx import helper, shape_inference
from onnx import version_converter


def prune_past_node(onnx_model: str):
    # 加载ONNX模型
    model = onnx.load(onnx_model)

    # 打印 IR 版本
    print(f"IR Version: {model.ir_version}")

    # 打印算子集版本
    for opset in model.opset_import:
        print(f"Operator Set: {opset.domain}, Version: {opset.version}")

    model = version_converter.convert_version(model, target_version=21)

    # 打印 IR 版本
    print(f"IR Version: {model.ir_version}")

    # 打印算子集版本
    for opset in model.opset_import:
        print(f"Operator Set: {opset.domain}, Version: {opset.version}")

    # 获取计算图
    graph = model.graph

    # 找到if算子
    if_node = None
    for node in graph.node:
        if node.op_type == 'If':
            if_node = node
            break

    if if_node is None:
        raise ValueError("No If node found in the model")

    # 提取then分支
    then_branch = if_node.attribute[1].g

    # 创建一个新的计算图，只包含then分支
    new_graph = helper.make_graph(
        then_branch.node,
        "extracted_then_branch",
        model.graph.input,
        then_branch.output,
        graph.initializer
    )

    # 创建一个新的模型
    new_model = helper.make_model(new_graph, producer_name='onnx-extract-then-branch')

    # 进行形状推断
    # new_model = shape_inference.infer_shapes(new_model)

    # 找到并移除use_cache_branch输入
    new_inputs = [input for input in new_model.graph.input if input.name != 'use_cache_branch']

    # 创建一个新的计算图，替换输入
    new_graph = helper.make_graph(
        new_model.graph.node,  # 保留所有节点
        new_model.graph.name,
        new_inputs,  # 使用新的输入列表
        new_model.graph.output,
        new_model.graph.initializer,

    )

    # 创建一个新的模型
    new_model = helper.make_model(new_graph,
                                  producer_name='onnx-extract-then-branch',
                                  opset_imports=[helper.make_opsetid("", 21)],  # 指定算子集版本
                                  )

    # # 进行形状推断
    new_model = shape_inference.infer_shapes(new_model)

    # 打印 IR 版本
    print(f"IR Version: {new_model.ir_version}")

    # 打印算子集版本
    for opset in new_model.opset_import:
        print(f"Operator Set: {opset.domain}, Version: {opset.version}")

    onnx.checker.check_model(new_model)

    # print(new_model.graph.input)

    return new_model


if __name__ == '__main__':
    model = prune_past_node('models/origin/decoder_model_merged.onnx')
    onnx.save(model, 'models/outputs/decoder_model_past.onnx')
