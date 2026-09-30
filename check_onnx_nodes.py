import onnx

MODEL_FILE = "signova_nn_qcom_final.onnx"

print("Checking ONNX operators...")

model = onnx.load(MODEL_FILE)

domains = {}

for node in model.graph.node:
    domain = node.domain if node.domain else "ai.onnx"
    domains[domain] = domains.get(domain, 0) + 1
    print("Node:", node.op_type, "| Domain:", domain)

print()
print("================================")
print("Domains found:")
for domain, count in domains.items():
    print(domain, ":", count)

print("================================")

if "ai.onnx.ml" in domains:
    print("❌ ai.onnx.ml STILL EXISTS")
else:
    print("✅ No ai.onnx.ml operators found!")

print("================================")