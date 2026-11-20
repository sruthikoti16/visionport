
import argparse
import os
from collections import Counter
from pathlib import Path


# ---- PART 1: Options and the role table --------------------------------------

parser = argparse.ArgumentParser(
    description="Inspect VisionPort model artefacts"
)

parser.add_argument(
    "--weights",
    default="yolov8n.pt",
    help="PyTorch checkpoint to inspect"
)

parser.add_argument(
    "--onnx",
    default=None,
    help="path of an existing .onnx file"
)

parser.add_argument(
    "--export-onnx",
    action="store_true",
    help="export the checkpoint to ONNX first (needs internet once)"
)

args = parser.parse_args()

ROLE = {
    ".pt": "PyTorch checkpoint: trained weights (+ architecture). Used for training and export.",
    ".onnx": "ONNX graph: framework-neutral model file. Used for export and ONNX Runtime.",
    ".engine": "TensorRT engine: compiled for one GPU and precision. Used for fast serving.",
    ".plan": "TensorRT engine (Triton naming). Used for fast serving.",
}


# ---- PART 2: Inventory of artefacts in the project ---------------------------

print("=== 1. ARTEFACT INVENTORY ===")

found = []

for folder, subfolders, files in os.walk("."):
    subfolders[:] = [
        d for d in subfolders
        if d not in (".venv", ".git", "__pycache__", ".idea")
    ]

    for name in files:
        if Path(name).suffix in ROLE:
            found.append(Path(folder) / name)

if not found:
    print("No .pt, .onnx, .engine or .plan files found in the project folder yet.")

for path in found:
    print(
        f"{str(path):<40} "
        f"{path.stat().st_size / 1e6:>8.1f} MB   "
        f"{ROLE[path.suffix]}"
    )


# ---- PART 3: The PyTorch checkpoint ------------------------------------------

print("\n=== 2. PYTORCH CHECKPOINT:", args.weights, "===")

yolo = None

try:
    import torch
    from ultralytics import YOLO

    yolo = YOLO(args.weights)
    weights_path = Path(args.weights)

    # 3a. What is stored inside the file?

    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False
    )

    print("Top-level contents of the checkpoint file:")

    for key, value in checkpoint.items():
        kind = type(value).__name__
        detail = f"{len(value)} entries" if isinstance(value, dict) else ""

        print(f" {key:<14} {kind:<16} {detail}")

    print(
        "Training state (optimizer) present:",
        "yes" if checkpoint.get("optimizer")
        else "no (stripped, for inference)"
    )

    stored_model = checkpoint.get("ema") or checkpoint.get("model")

    if stored_model is not None:
        print(
            "Weights stored in the file as:",
            next(stored_model.parameters()).dtype
        )

    # 3b. The architecture: a list of layers

    model = yolo.model

    print(
        f"\nArchitecture: {type(model).__name__}, "
        f"{len(model.model)} top-level layers"
    )

    for i in (0, 1, 2, len(model.model) - 1):
        layer = model.model[i]
        n = sum(p.numel() for p in layer.parameters())

        print(
            f" layer {i:>2}: {type(layer).__name__:<10} "
            f"{n:>10,} parameters"
        )

    # 3c. The weights: a dictionary of named tensors

    state = model.state_dict()

    n_params = sum(p.numel() for p in model.parameters())
    first = next(model.parameters())

    print(
        f"\nWeights: {len(state)} named tensors, "
        f"{n_params:,} parameters, loaded as {first.dtype}"
    )

    print(
        f"Approx. weight memory: "
        f"{n_params * first.element_size() / 1e6:.1f} MB"
    )

    for name in list(state)[:3]:
        print(f" {name:<36} shape {tuple(state[name].shape)}")

    print(
        f"Classes the model predicts: {len(model.names)} "
        f"e.g. {list(model.names.values())[:5]}"
    )

except Exception as error:
    print(
        "Could not inspect the checkpoint:",
        type(error).__name__,
        error
    )

    print(
        "Check that torch and ultralytics are installed "
        "and that the weights file exists."
    )


# ---- PART 4: The ONNX graph ---------------------------------------------------

onnx_path = args.onnx

if args.export_onnx:
    print("\nExporting to ONNX (this installs a few packages the first time)...")

    if yolo is None:
        print("Cannot export ONNX because the PyTorch checkpoint was not loaded.")

    else:
        try:
            onnx_path = str(yolo.export(format="onnx", imgsz=640))

        except Exception as error:
            print("Export failed:", type(error).__name__, error)

if onnx_path is None:
    guess = Path(args.weights).with_suffix(".onnx")
    onnx_path = str(guess) if guess.exists() else None

print("\n=== 3. ONNX GRAPH ===")

if onnx_path is None:
    print(
        "No .onnx file yet. Create one with: "
        "python inspect_artefacts.py --export-onnx"
    )

else:
    try:
        import onnx

        model_proto = onnx.load(onnx_path)
        onnx.checker.check_model(model_proto)

        print(
            f"File           : {onnx_path} "
            f"({Path(onnx_path).stat().st_size / 1e6:.1f} MB)"
        )

        print("Format check   : passed")
        print(f"Opset version : {model_proto.opset_import[0].version}")

        def describe(tensor_infos):
            rows = []

            for t in tensor_infos:
                dims = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.type.tensor_type.shape.dim
                ]

                rows.append(f"{t.name} {dims}")

            return rows

        weights = {
            w.name for w in model_proto.graph.initializer
        }

        print(
            "Inputs          :",
            describe(
                i for i in model_proto.graph.input
                if i.name not in weights
            )
        )

        print(
            "Outputs         :",
            describe(model_proto.graph.output)
        )

        ops = Counter(
            node.op_type for node in model_proto.graph.node
        )

        print(
            f"Graph           : {len(model_proto.graph.node)} operations, "
            f"{len(model_proto.graph.initializer)} weight tensors"
        )

        print(
            "Most common operations:",
            ", ".join(
                f"{k} x{v}" for k, v in ops.most_common(6)
            )
        )

    except Exception as error:
        print(
            "Could not inspect the ONNX file:",
            type(error).__name__,
            error
        )

        print("Install the reader with: pip install onnx")


# ---- PART 5: The TensorRT engine ---------------------------------------------

print("\n=== 4. TENSORRT ENGINE ===")

engines = [
    p for p in found
    if p.suffix in (".engine", ".plan")
]

if engines:
    for p in engines:
        print(
            f"{p} ({p.stat().st_size / 1e6:.1f} MB). "
            "It cannot be opened with PyTorch or ONNX tools;"
        )

        print(
            "only TensorRT on a matching NVIDIA GPU can load it."
        )

else:
    print(
        "No engine found. This is expected on a computer without an NVIDIA GPU."
    )

    print(
        "An engine is built from the ONNX file in Session 13, "
        "on the GPU that will run it."
    )


# ---- PART 6: Why they are not interchangeable -------------------------------

print("\n=== 5. COMPARISON ===")

rows = [
    (
        "Checkpoint (.pt)",
        "weights + architecture",
        "PyTorch + Python",
        "training, fine-tuning, export"
    ),
    (
        "ONNX (.onnx)",
        "graph + weights, no Python",
        "ONNX Runtime or TensorRT",
        "portable hand-over format"
    ),
    (
        "Engine (.engine)",
        "optimised, compiled graph",
        "NVIDIA GPU + TensorRT",
        "fast serving on that GPU"
    ),
]

print(
    f"{'artefact':<18}"
    f"{'contains':<28}"
    f"{'needs':<26}"
    f"used for"
)

for row in rows:
    print(
        f"{row[0]:<18}"
        f"{row[1]:<28}"
        f"{row[2]:<26}"
        f"{row[3]}"
    )
