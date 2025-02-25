from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
from pathlib import Path
from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
    quantize
)


def quant_4_dynamic(model_input, model_output):
    model = onnx.load(model_input)
    quant_pre_process(model, model_output, skip_symbolic_shape=True)
    quantize_dynamic(model_output, model_output, weight_type=QuantType.QInt4)


def quant_8_dynamic(model_input, model_output):
    model = onnx.load(model_input)
    quant_pre_process(model, model_output, skip_symbolic_shape=True)
    quantize_dynamic(
        model_output, model_output, op_types_to_quantize=['MatMul'], weight_type=QuantType.QInt8,
        nodes_to_exclude=['/embeddings/patch_embeddings/projection/Conv'])


def quant_4(model_input, model_output):
    quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128,  # 2's exponential and >= 16
        is_symmetric=True,  # if true, quantize to Int4. otherwsie, quantize to uint4.
        accuracy_level=4,
        # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),  # specify which op types to quantize
        quant_axes=(("MatMul", 0), ("Gather", 1),)  # specify which axis to quantize for an op type.
    )
    model = quant_utils.load_model_with_shape_infer(Path(model_input))
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model,
        nodes_to_exclude=None,  # specify a list of nodes to exclude from quantizaiton
        nodes_to_include=None,  # specify a list of nodes to force include from quantization
        algo_config=quant_config, )
    quant.process()
    quant.model.save_model_to_file(
        model_output,
        True)  # save data to external file


if __name__ == '__main__':
    # quant_4_dynamic("models/outputs/decoder_model_past.onnx","models/outputs/decoder_model_q4.onnx")
    # quant_4_dynamic("models/origin/encoder_model.onnx ","models/outputs/encoder_q4.onnx")
    quant_8_dynamic("models/origin/encoder_model.onnx ", "models/outputs/encoder_q8.onnx")
    quant_8_dynamic("models/outputs/decoder_model_past.onnx", "models/outputs/decoder_model_q8.onnx")
    # quant_4("models/outputs/decoder_model_past_sim.onnx","models/outputs/decoder_model_q4.onnx")
