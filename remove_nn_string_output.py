import onnx

INPUT_FILE = "signova_nn.onnx"
OUTPUT_FILE = "signova_nn_qcom.onnx"

print("Loading Neural Network ONNX...")

model = onnx.load(INPUT_FILE)

print()
print("Original outputs:")

for output in model.graph.output:
    print(
        "Name:", output.name,
        "| Type:", output.type.tensor_type.elem_type
    )

# Keep only the numeric probability output
kept_outputs = [
    output
    for output in model.graph.output
    if output.name == "probabilities"
]

if not kept_outputs:
    print()
    print("ERROR: 'probabilities' output was not found.")
    print("Do not continue. Send me the output above.")
    exit()

model.graph.ClearField("output")
model.graph.output.extend(kept_outputs)

onnx.save(model, OUTPUT_FILE)

print()
print("================================")
print("Object/string output removed!")
print("Output:", OUTPUT_FILE)
print("================================")