from ultralytics import YOLO

model = YOLO("best (1).pt")

print("Exporting ONNX (fallback)...")
model.export(
    format="onnx",
    imgsz=160,
    opset=12,
    half=False,
    simplify=True,
    dynamic=False,
)

print("Exporting OpenVINO (FASTEST on CPU)...")
model.export(
    format="openvino",
    imgsz=160,
    half=False,
)

print("Done -> best (1).onnx and best (1)_openvino_model/")