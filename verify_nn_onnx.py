import onnx
import onnxruntime as ort

MODEL_FILE = "signova_nn.onnx"

print("Loading Neural Network ONNX...")

model = onnx.load(MODEL_FILE)
onnx.checker.check_model(model)

print("ONNX model loaded successfully!")

print()
print("Inputs:")
for inp in model.graph.input:
    print("Name:", inp.name)
    print("Shape:", [d.dim_value if d.dim_value else "dynamic"
                     for d in inp.type.tensor_type.shape.dim])

print()
print("Outputs:")
for output in model.graph.output:
    print(
        "Name:", output.name,
        "Type:", output.type.tensor_type.elem_type,
        "Shape:",
        [d.dim_value if d.dim_value else "dynamic"
         for d in output.type.tensor_type.shape.dim]
    )

print()
print("Testing ONNX Runtime...")

session = ort.InferenceSession(
    MODEL_FILE,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime loaded successfully!")

print()
print("Input name:", session.get_inputs()[0].name)
print("Input shape:", session.get_inputs()[0].shape)

for output in session.get_outputs():
    print("Output:", output.name, output.shape)

print()
print("================================")
print("ONNX verification successful!")
print("================================")