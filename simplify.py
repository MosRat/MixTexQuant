import onnx
from onnxsim import simplify

def simplify_onnx(onnx_model):
    # load your predefined ONNX model
    model = onnx.load(onnx_model)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    # use model_simp as a standard ONNX model object
    return  model_simp

if __name__ == '__main__':
    onnx_model = simplify_onnx("models/origin/encoder_model.onnx")
    onnx.save(onnx_model, "models/outputs/encoder_sim.onnx")