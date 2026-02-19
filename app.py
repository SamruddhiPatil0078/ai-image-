from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import torch

from face_detection import detect_faces_from_array
from face_alignment import align_face
from embedding_facenet import (
    extract_embedding_facenet,
    extract_embedding_tensor,
    get_facenet_model
)
from distance import cosine_distance
from target_identity import generate_target_embedding
from perturbation_optimizer import optimize_perturbation

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load FaceNet once
facenet_model = get_facenet_model()


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def cv2_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)


def tensor_to_cv2(tensor):
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("image")
    if file is None:
        return jsonify([])

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify([])

    faces = detect_faces_from_array(image)
    results = []

    for i, f in enumerate(faces):

        bbox = (f["xmin"], f["ymin"], f["width"], f["height"])

        # --- Safe alignment ---
        result = align_face(image, bbox)

        if result is None:
            continue

        aligned, mask = result

        if aligned is None or not isinstance(aligned, np.ndarray):
            continue

        # ---------------- Tensor embedding ----------------
        face_tensor = cv2_to_tensor(aligned)
        orig_embedding = extract_embedding_tensor(face_tensor)

        # ---------------- Fake identity ----------------
        target_embedding = generate_target_embedding(orig_embedding)

        # ---------------- Optimize perturbation ----------------
        protected_tensor = optimize_perturbation(
            model=facenet_model,
            image_tensor=face_tensor,
            orig_emb=orig_embedding,
            target_emb=target_embedding,
            device=device
        )

        protected_face = tensor_to_cv2(protected_tensor)

        out_path = f"static/protected_{i}.jpg"
        cv2.imwrite(out_path, protected_face)

        # ---------------- Verification ----------------
        orig_np = extract_embedding_facenet(aligned)
        new_np = extract_embedding_facenet(protected_face)

        if new_np is not None:
            dist_orig = cosine_distance(orig_np, new_np)
            dist_target = cosine_distance(
                target_embedding.detach().cpu().numpy()[0],
                new_np
            )

            print(f"[Face {i}] distance from original: {dist_orig:.4f}")
            print(f"[Face {i}] distance from target:   {dist_target:.4f}")

        results.append({
            "face_id": i,
            "protected_image": out_path
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
