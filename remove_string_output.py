import onnx

INPUT_FILE = "signova_model_qcom.onnx"
OUTPUT_FILE = "signova_model_qcom_final.onnx"

print("Loading ONNX model...")

model = onnx.load(INPUT_FILE)

print("Original outputs:")
for output in model.graph.output:
    print(output.name)

# Keep only the numeric probability output
kept_outputs = [
    output for output in model.graph.output
    if output.name == "probabilities"
]

model.graph.ClearField("output")
model.graph.output.extend(kept_outputs)

onnx.save(model, OUTPUT_FILE)

print()
print("================================")
print("String output removed!")
print("Output:", OUTPUT_FILE)
print("================================")