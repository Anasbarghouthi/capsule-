"""Serve the image upload interface and run YOLO inference with best.pt."""

from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_UI_DIR = ROOT / "web_interface"


def find_latest_best_pt(root: Path = ROOT) -> Path:
    """Find the newest YOLO best.pt file inside runs/."""
    candidates = list((root / "runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No best.pt was found under runs/. Pass --weights with the trained model path."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(weights: Path):
    """Load Ultralytics YOLO only when the server starts."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: "
            "python -m pip install ultralytics"
        ) from exc
    return YOLO(str(weights))


def response_json(handler: SimpleHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def content_type_suffix(content_type: str) -> str:
    clean_type = content_type.split(";", 1)[0].strip().lower()
    if clean_type == "image/png":
        return ".png"
    if clean_type == "image/webp":
        return ".webp"
    return ".jpg"


class ModelRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler with a /predict endpoint."""

    model = None
    imgsz = 640
    conf = 0.25
    device = ""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            response_json(self, {"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
            return
        self.handle_predict()

    def handle_predict(self) -> None:
        if self.model is None:
            response_json(self, {"error": "Model is not loaded"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            response_json(self, {"error": "No image body was received"}, HTTPStatus.BAD_REQUEST)
            return

        image_bytes = self.rfile.read(length)
        suffix = content_type_suffix(self.headers.get("Content-Type", ""))

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(image_bytes)
                temp_path = Path(temp_file.name)

            predict_args: dict[str, Any] = {
                "source": str(temp_path),
                "imgsz": self.imgsz,
                "conf": self.conf,
                "verbose": False,
            }
            if self.device:
                predict_args["device"] = self.device

            results = self.model.predict(**predict_args)
            response_json(self, format_prediction(results[0], self.model.names))
        except Exception as exc:  # noqa: BLE001 - return the error to the local UI.
            response_json(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)


def format_prediction(result: Any, names: dict[int, str] | list[str]) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x_center, y_center, width, height = [float(value) for value in box.xywhn[0].tolist()]
            label = names[class_id] if isinstance(names, list) else names.get(class_id, f"class_{class_id}")
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence * 100, 2),
                    "box": {
                        "x": max(0.0, x_center - width / 2),
                        "y": max(0.0, y_center - height / 2),
                        "width": min(1.0, width),
                        "height": min(1.0, height),
                    },
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    max_confidence = round(detections[0]["confidence"]) if detections else 0

    return {
        "mode": "model",
        "verdict": "Polyp / tumor suspected" if detections else "No clear suspected polyp",
        "level": "danger" if detections else "clear",
        "confidence": max_confidence,
        "detections": detections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained YOLO model with the HTML interface.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained best.pt.")
    parser.add_argument("--ui-dir", type=Path, default=DEFAULT_UI_DIR, help="Directory that contains index.html.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8765, help="Server port.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--device", default="", help="Optional device, for example cpu or 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = (args.weights or find_latest_best_pt()).resolve()
    ui_dir = args.ui_dir.resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights}")
    if not (ui_dir / "index.html").exists():
        raise FileNotFoundError(f"UI index.html does not exist in: {ui_dir}")

    ModelRequestHandler.model = load_model(weights)
    ModelRequestHandler.imgsz = args.imgsz
    ModelRequestHandler.conf = args.conf
    ModelRequestHandler.device = args.device
    handler_class = partial(ModelRequestHandler, directory=str(ui_dir))

    mimetypes.add_type("application/javascript", ".js")
    server = ThreadingHTTPServer((args.host, args.port), handler_class)

    print("Model web app is running")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Weights: {weights}")
    print(f"UI directory: {ui_dir}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
