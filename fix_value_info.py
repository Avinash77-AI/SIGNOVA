import onnx

INPUT_FILE = "signova_nn_qcom_final.onnx"
OUTPUT_FILE = "signova_nn_qcom_clean.onnx"

print("Loading ONNX model...")

model = onnx.load(INPUT_FILE)

io_names = set()

for tensor in model.graph.input:
    io_names.add(tensor.name)

for tensor in model.graph.output:
    io_names.add(tensor.name)

print("Model IO names:")
for name in io_names:
    print(" -", name)

print()
print("Original value_info entries:", len(model.graph.value_info))

kept = []

for value in model.graph.value_info:
    if value.name not in io_names:
        kept.append(value)

model.graph.ClearField("value_info")
model.graph.value_info.extend(kept)

print("Remaining value_info entries:", len(model.graph.value_info))

onnx.checker.check_model(model)

onnx.save(model, OUTPUT_FILE)

print()
print("================================")
print("CLEAN ONNX MODEL CREATED!")
print("Output:", OUTPUT_FILE)
print("================================")