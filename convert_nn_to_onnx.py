import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_FILE = "signova_nn.pkl"
OUTPUT_FILE = "signova_nn.onnx"

print("Loading SIGNOVA Neural Network...")

model = joblib.load(MODEL_FILE)

print("Model loaded!")
print("Features:", model.n_features_in_)
print("Classes:", model.classes_)

initial_type = [
    ("input", FloatTensorType([None, 63]))
]

print("Converting Neural Network to ONNX...")

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={
        id(model): {
            "zipmap": False,
            "output_class_labels": False
        }
    }
)

with open(OUTPUT_FILE, "wb") as f:
    f.write(onnx_model.SerializeToString())

print()
print("================================")
print("Neural Network ONNX conversion successful!")
print("Output:", OUTPUT_FILE)
print("================================")