import numpy as np
import onnxruntime as ort

MODEL_PATH = r"snapdragon_model\job_jpxlzw78p_optimized_onnx\model.onnx"

CLASSES = ["No", "Thank_You", "Yes", "hello"]

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("✅ Snapdragon model loaded")
print("Input :", input_name)
print("Output:", output_name)

# Use the same real SIGNOVA sample from the previous test
data = np.loadtxt(
    "sign_data_normalized.csv",
    delimiter=",",
    dtype=str
)

features = data[0, 1:].astype(np.float32)
expected = data[0, 0]

x = features.reshape(1, 63)

result = session.run(
    [output_name],
    {input_name: x}
)

probabilities = np.asarray(result[0])[0]

predicted_index = int(np.argmax(probabilities))
predicted_class = CLASSES[predicted_index]
confidence = float(probabilities[predicted_index])

print("\nExpected :", expected)
print("Predicted:", predicted_class)
print("Confidence:", f"{confidence * 100:.2f}%")

print("\nAll probabilities:")
for name, probability in zip(CLASSES, probabilities):
    print(f"{name:10s}: {probability * 100:.4f}%")

if predicted_class.lower() == expected.lower():
    print("\n🎯 TEST PASSED — prediction is correct!")
else:
    print("\n❌ TEST FAILED — prediction mismatch.")