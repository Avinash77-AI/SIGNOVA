import joblib

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_FILE = "signova_model.pkl"
OUTPUT_FILE = "signova_model_qcom.onnx"

print("Loading Random Forest model...")
model = joblib.load(MODEL_FILE)

print("Model loaded!")
print("Features:", model.n_features_in_)
print("Classes:", model.classes_)

initial_type = [
    ("input", FloatTensorType([None, 63]))
]

print("Converting model to ONNX...")

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
print("Qualcomm ONNX created!")
print("Output:", OUTPUT_FILE)
print("================================")