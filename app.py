import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
from io import BytesIO
import time
import os
import traceback

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Drone Detector (Single Image)",
    page_icon="🛸",
    layout="wide"
)

st.title("🛸 Drone Detection – Single Image (YOLOv11)")
st.markdown("""
Load a local YOLO model (.pt or .onnx). (TensorRT .engine is not supported on Apple Silicon.)
Choose or upload an image, run inference, and download the annotated result + JSON.
""")

# ---------------------- Utility Functions ----------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    """
    Cache the loaded YOLO model. Automatically adds '.pt' suffix if you pass a stem by accident.
    """
    p = Path(model_path)
    if p.is_dir():
        raise ValueError("Model path is a directory, expected a file.")
    # Ultralytics can infer .pt if stem only; but we keep explicit.
    return YOLO(str(p))

def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

def read_image_any(source_file, typed_path, picked_file):
    """
    1) If user uploaded a file (source_file), use it.
    2) Else if typed_path is provided and exists, load from disk.
    3) Else if picked_file (from dropdown) is selected, load that.
    Returns (image_array_bgr, origin_path_or_label).
    """
    # Uploaded file
    if source_file is not None:
        data = source_file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image.")
        return img, source_file.name

    # Typed local path
    if typed_path:
        p = Path(typed_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Typed image path not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            # Try Pillow fallback for formats like webp/tiff
            try:
                pil = Image.open(p).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError("Failed to load typed image via OpenCV/Pillow.")
        return img, str(p)

    # Picked from list
    if picked_file:
        p = Path(picked_file)
        img = cv2.imread(str(p))
        if img is None:
            pil = Image.open(p).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img, str(p)

    raise ValueError("No image source selected. Upload, type a path, or choose from folder list.")

def resize_if_needed(img, max_side: int):
    """
    If the largest side exceeds max_side, resize proportionally.
    """
    if max_side is None or max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def run_inference(model, img_bgr, conf, iou):
    """
    Runs YOLO inference and returns (annotated_bgr, detections_list, raw_result, inference_time).
    """
    start = time.time()
    results = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)
    end = time.time()
    result = results[0]
    names = result.names if hasattr(result, "names") else {0: "drone"}

    detections = []
    if hasattr(result, "boxes") and result.boxes is not None:
        for b in result.boxes:
            cls_id = int(b.cls[0])
            conf_score = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(conf_score, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2)
            })

    # Annotate
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, thickness=LINE_THICKNESS)
        if DRAW_LABELS:
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 2, y1), BOX_COLOR, -1)
            cv2.putText(annotated, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return annotated, detections, result, end - start

def list_local_models():
    """
    Finds model files in current directory: .pt, .onnx, .engine
    Returns sorted list of filenames.
    """
    exts = (".pt", ".onnx", ".engine")
    files = [p.name for p in Path(".").glob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def list_local_images(folder="images"):
    """
    Lists images in a folder (if exists) for quick selection.
    """
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    p = Path(folder)
    if not p.exists():
        return []
    imgs = [str(pp) for pp in p.rglob("*") if pp.suffix.lower() in exts]
    imgs.sort()
    return imgs

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.subheader("Model Selection")
    available_models = list_local_models()
    if available_models:
        selected_model = st.selectbox("Pick a local model file", available_models, index=0)
    else:
        st.error("No model files (.pt / .onnx / .engine) found in this folder.")
        selected_model = None

    st.markdown("---")
    st.subheader("Image Source")
    uploaded_image = st.file_uploader("Upload image (drag & drop)", type=["jpg","jpeg","png","webp","bmp","tif","tiff"])
    image_path_input = st.text_input("Or type local image path", value="")
    local_images = list_local_images("images")
    picked_local_image = st.selectbox("Or pick from ./images (optional)", [""] + local_images)

    st.markdown("---")
    st.subheader("Inference Settings")
    CONF_THRES = st.slider("Confidence Threshold", 0.01, 0.95, 0.25, 0.01)
    IOU_THRES = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.5, 0.05)

    DRAW_LABELS = st.checkbox("Draw labels", value=True)
    BOX_COLOR_HEX = st.color_picker("Box color", "#00FF00")
    BOX_COLOR = hex_to_bgr(BOX_COLOR_HEX)
    LINE_THICKNESS = st.slider("Line thickness", 1, 8, 2)

    MAX_SIDE = st.number_input("Max image side (resize if larger) - 0 = no resize",
                               min_value=0, max_value=4000, value=1600, step=100)
    RUN_BUTTON = st.button("Run Inference")

# ---------------------- Main Inference Logic ----------------------a
if RUN_BUTTON:
    if not selected_model:
        st.error("No model selected.")
        st.stop()

    # 1. Load model
    try:
        model = load_model(selected_model)
        st.success(f"Loaded model: {selected_model}")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    # 2. Acquire image
    try:
        img_bgr, src_label = read_image_any(
            source_file=uploaded_image,
            typed_path=image_path_input.strip(),
            picked_file=picked_local_image if picked_local_image else None
        )
    except Exception as e:
        st.error(f"Image load failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    original_shape = img_bgr.shape[:2]  # (h, w)
    resized_flag = False
    if MAX_SIDE > 0:
        before = img_bgr.shape[:2]
        img_bgr = resize_if_needed(img_bgr, MAX_SIDE)
        after = img_bgr.shape[:2]
        if before != after:
            resized_flag = True

    st.info(f"Source: {src_label} | Original: {original_shape[1]}x{original_shape[0]} "
            f"| Current: {img_bgr.shape[1]}x{img_bgr.shape[0]} (resized: {resized_flag})")

    # 3. Inference
    with st.spinner("Running inference..."):
        try:
            annotated, detections, raw_result, infer_time = run_inference(
                model, img_bgr, CONF_THRES, IOU_THRES
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # 4. Results
    st.subheader("Inference Stats")
    stats = {
        "model_file": selected_model,
        "source_image": src_label,
        "original_width": original_shape[1],
        "original_height": original_shape[0],
        "processed_width": img_bgr.shape[1],
        "processed_height": img_bgr.shape[0],
        "resized": resized_flag,
        "num_detections": len(detections),
        "confidence_threshold": CONF_THRES,
        "iou_threshold": IOU_THRES,
        "inference_time_sec": round(infer_time, 4),
        "approx_fps": round(1.0 / infer_time, 2) if infer_time > 0 else None
    }
    st.json(stats)

    st.subheader("Detections")
    if detections:
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("No detections above threshold.")

    st.subheader("Annotated Image")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 5. Download buttons
    # Annotated image
    pil_annotated = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    img_buf = BytesIO()
    pil_annotated.save(img_buf, format="PNG")
    st.download_button(
        label="⬇️ Download Annotated PNG",
        data=img_buf.getvalue(),
        file_name="annotated_image.png",
        mime="image/png"
    )

    # JSON
    json_buf = BytesIO()
    json_buf.write(json.dumps({"stats": stats, "detections": detections}, indent=2).encode("utf-8"))
    st.download_button(
        label="⬇️ Download Detections JSON",
        data=json_buf.getvalue(),
        file_name="detections.json",
        mime="application/json"
    )

# --------------- Help / Troubleshooting Section ---------------
st.markdown("---")
with st.expander("Troubleshooting Upload / Model Issues"):
    st.markdown("""
**Can't upload image?**  
- Ensure `maxUploadSize` is sufficient (see `.streamlit/config.toml`).  
- Try smaller image or use the local path method.

**Model not loading?**  
- On M1, use `.pt` or `.onnx`. A `.engine` built on Linux/NVIDIA will fail.  
- Make sure you installed a native arm64 Python (NOT Rosetta) and reinstalled dependencies.

**Slow inference?**  
- Reduce image size (set Max image side).  
- Use a smaller YOLO model variant (`yolo11n.pt`, `yolo11s.pt`).

**ONNX slower than .pt?**  
- On M1, PyTorch Metal backend can be faster; ONNX runtime is CPU-only.

**Need multiple classes?**  
- The app automatically uses class names from the model's `result.names`.
""")

st.caption("© Drone Detection Single Image App – YOLOv11")