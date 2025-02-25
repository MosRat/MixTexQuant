import numpy as np
import onnx
import onnxruntime as ort
print(ort.__version__)
print(ort.get_device())

from pathlib import Path

org_enc_path = Path("models/origin/encoder_model.onnx")
org_dec_path = Path("models/origin/decoder_model_merged.onnx")

quant_enc_path = Path("models/outputs/encoder_q8.onnx")
quant_dec_path = Path("models/outputs/decoder_model_q8.onnx")

org_encoder = ort.InferenceSession(org_enc_path)
org_decoder = ort.InferenceSession(org_dec_path)

fake_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

org_hidden_state: np.ndarray = org_encoder.run(None, {org_encoder.get_inputs()[0].name: fake_input})[0]
print(org_hidden_state.shape)

kv = {
    f'past_key_values.{i}.{n}': np.random.randn(1, 12, 0, 64).astype(np.float32) for i in range(6) for n in
    ['key', 'value']
}

org_output: np.ndarray = org_decoder.run(None, {
    "input_ids": np.array([[300, 300, 123]], dtype=np.int64),
    "use_cache_branch": np.array([True], dtype=np.bool),
    "encoder_hidden_states": org_hidden_state,
    **kv
})[0]

quant_encoder = ort.InferenceSession(quant_enc_path)
quant_decoder = ort.InferenceSession(quant_dec_path,providers=["CUDAExecutionProvider"])

quant_hidden_state: np.ndarray = quant_encoder.run(None, {quant_encoder.get_inputs()[0].name: fake_input})[0]

quant_output: np.ndarray = quant_decoder.run(None, {
    "input_ids": np.array([[300, 300, 123]], dtype=np.int64),
    "encoder_hidden_states": quant_hidden_state,
    **kv
})[0]

print(np.allclose(org_hidden_state, quant_hidden_state), np.abs(org_hidden_state - quant_hidden_state).max())
print(np.allclose(org_output, quant_output), np.abs(org_output - quant_output).max())
