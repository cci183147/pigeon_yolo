from flask import Flask, request, jsonify, send_from_directory, abort, send_file
import os, uuid, torch
import numpy as np

from utils.yolo_crop import IrisCropper
from utils.embedding import embed_image
from utils.faiss_search import IrisSearcher
from models.siamese import load_siamese

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_THRESHOLD = 0.8

ASSETS = "assets"
YOLO_MODEL = f"{ASSETS}/best.pt"
SIAMESE_MODEL = f"{ASSETS}/best.pth"
FAISS_INDEX = f"{ASSETS}/idx.faiss"
META_CSV = f"{ASSETS}/meta.csv"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ---------------------------------------

app = Flask(__name__)

# ---- load once ----
cropper = IrisCropper(YOLO_MODEL)
siamese = load_siamese(SIAMESE_MODEL, DEVICE)
searcher = IrisSearcher(FAISS_INDEX, META_CSV)


# -------- frontend --------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# -------- api --------
def save_upload(file):
    fn = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, fn)
    file.save(path)
    return path


@app.route("/compare", methods=["POST"])
def compare():
    f1 = request.files.get("img1")
    f2 = request.files.get("img2")
    if not f1 or not f2:
        return jsonify({"error": "need two images"}), 400

    p1 = save_upload(f1)
    p2 = save_upload(f2)

    c1 = cropper.crop(p1)
    if c1 is None:
        return {"error": "Image 1 iris not detected or invalid image"}

    c2 = cropper.crop(p2)
    if c2 is None:
        return {"error": "Image 2 iris not detected or invalid image"}

    if c1 is None or c2 is None:
        return jsonify({"error": "iris not detected"})

    e1 = embed_image(siamese, c1, DEVICE)
    e2 = embed_image(siamese, c2, DEVICE)

    sim = float((e1 * e2).sum())
    same = sim >= SIM_THRESHOLD

    return {
        "similarity": sim,
        "same_blood": same,
        "crop1": c1,
        "crop2": c2
    }


@app.route("/search", methods=["POST"])
def search():
    f = request.files.get("image")
    k = int(request.form.get("k", 5))

    if not f:
        return jsonify({"error": "need image"}), 400

    p = save_upload(f)


    crop_path = cropper.crop(p)
    if crop_path is None:
        return jsonify({"error": "iris not detected"}), 400

    emb = embed_image(siamese, crop_path, DEVICE)


    results = searcher.search(emb, k)


    for r in results:
        r["image"] = "/image?path=" + r["path"]

    return jsonify({
        "query_crop": "/image?path=" + crop_path,
        "results": results
    })

@app.route("/image")
def image():
    path = request.args.get("path")
    if path is None:
        abort(400)

    if not os.path.exists(path):
        abort(404)

    return send_file(path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
