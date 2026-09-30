import onnxruntime as ort
import numpy as np

MODEL_FILE = "signova_model.onnx"

print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_FILE,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully!")

print("Inputs:")

for inp in session.get_inputs():
    print(
        "Name:",
        inp.name,
        "Shape:",
        inp.shape
    )

print("Outputs:")

for out in session.get_outputs():
    print(
        "Name:",
        out.name,
        "Shape:",
        out.shape
    )

# Test with one dummy 63-feature input
dummy_input = np.zeros(
    (1, 63),
    dtype=np.float32
)

input_name = session.get_inputs()[0].name

result = session.run(
    None,
    {
        input_name: dummy_input
    }
)

print()
print("Inference test successful!")
print("Number of outputs:", len(result))