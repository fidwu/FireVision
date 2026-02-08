import io
import numpy as np
from typing import List
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smoke and Fire Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://firevision.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model
session = ort.InferenceSession(
    "smoke_fire_classifier.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = [
    {"id": "smoke", "display": "Smoke"},
    {"id": "fire", "display": "Fire"},
    {"id": "non_fire", "display": "Non-Fire"},
]

MAX_SIZE = 500 * 1024  # 500 KB


def preprocess_pytorch(X):
    X = np.expand_dims(X, axis=0)

    X = X / 255.0

    X = X.transpose(0, 3, 1, 2)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    mean_tensor = mean.reshape(1, 3, 1, 1)
    std_tensor = std.reshape(1, 3, 1, 1)

    X = (X - mean_tensor) / std_tensor

    return X.astype(np.float32)


preprocessor = create_preprocessor(preprocess_pytorch, target_size=(224, 224))


@app.get("/")
async def root():
    return {"status": "ok", "message": "Smoke and Fire Classifier is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        return {
            "error": f"{file.filename} exceeds max size of 500 KB"
        }

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    x = preprocessor.preprocess(img)

    outputs = session.run([output_name], {input_name: x})
    logits = outputs[0][0]

    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    class_probabilities = {
        item["display"]: float(p) for item, p in zip(CLASSES, probs)
    }

    class_id = int(np.argmax(probs))
    class_info = CLASSES[class_id]

    return {
        "predicted_class": class_info["id"],
        "predicted_class_display": class_info["display"],
        "confidence": float(probs[class_id]),
        "probabilities": class_probabilities
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 100:
        return {"error": "Max 100 images per batch"}

    batch_inputs = []
    filenames = []
    for file in files:
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            return {
                "error": f"{file.filename} exceeds max size of 500 KB"
            }

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        x = preprocessor.preprocess(img)
        batch_inputs.append(x)
        filenames.append(file.filename)

    batch_array = np.vstack(batch_inputs)

    outputs = session.run([output_name], {input_name: batch_array})
    logits_batch = outputs[0]

    results = []
    for i, logits in enumerate(logits_batch):
        # Softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        class_probabilities = {
            item["display"]: float(p) for item, p in zip(CLASSES, probs)
        }

        class_id = int(np.argmax(probs))
        class_info = CLASSES[class_id]

        results.append({
            "filename": filenames[i],
            "predicted_class": class_info["id"],
            "predicted_class_display": class_info["display"],
            "confidence": float(probs[class_id]),
            "probabilities": class_probabilities
        })

    return {"results": results}
