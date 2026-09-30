import onnx
import onnxruntime as ort

MODEL_FILE = "signova_nn_qcom.onnx"

print("Loading Qualcomm-friendly ONNX...")

model = onnx.load(MODEL_FILE)
onnx.checker.check_model(model)

print("ONNX model loaded successfully!")

print()
print("Outputs:")

for output in model.graph.output:
    print(
        "Name:", output.name,
        "| Type:", output.type.tensor_type.elem_type,
        "| Shape:",
        [
            d.dim_value if d.dim_value else "dynamic"
            for d in output.type.tensor_type.shape.dim
        ]
    )

print()
print("Testing ONNX Runtime...")

session = ort.InferenceSession(
    MODEL_FILE,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime loaded successfully!")

print()
print("Input:")
print("Name:", session.get_inputs()[0].name)
print("Shape:", session.get_inputs()[0].shape)

print()
print("Outputs:")

for output in session.get_outputs():
    print("Name:", output.name)
    print("Shape:", output.shape)

print()
print("================================")
print("QUALCOMM-FRIENDLY ONNX VERIFIED!")
print("================================")