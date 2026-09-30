import onnx
from onnx.utils import extract_model

INPUT_FILE = "signova_nn.onnx"
OUTPUT_FILE = "signova_nn_qcom_final.onnx"

print("Pruning unused ONNX graph...")

model = onnx.load(INPUT_FILE)

print("Original outputs:")
for output in model.graph.output:
    print(" -", output.name)

print()
print("Extracting only:")
print(" input        -> probabilities")

extract_model(
    INPUT_FILE,
    OUTPUT_FILE,
    input_names=["input"],
    output_names=["probabilities"]
)

print()
print("================================")
print("ONNX graph successfully pruned!")
print("Output:", OUTPUT_FILE)
print("================================")