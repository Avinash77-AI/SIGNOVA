import os
import joblib
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"snapdragon_model\job_jpxlzw78p_optimized_onnx\model.onnx"
DATA_PATH = "sign_data_normalized.csv"
CLASS_MODEL = "signova_nn.pkl"

print("Loading Snapdragon optimized model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("✅ Model loaded")
print("Input:", input_name)
print("Output:", output_name)

# Load one real SIGNOVA sample
data = np.loadtxt(
    DATA_PATH,
    delimiter=",",
    dtype=str
)

label = data[0, 0]
features = data[0, 1:].astype(np.float32)

x = features.reshape(1, 63)

result = session.run(
    [output_name],
    {input_name: x}
)

output = np.asarray(result[0])

print("\nExpected class:", label)
print("Model output shape:", output.shape)
print("Model output:", output)

print("\n🎯 SNAPDRAGON OPTIMIZED MODEL TEST COMPLETE")