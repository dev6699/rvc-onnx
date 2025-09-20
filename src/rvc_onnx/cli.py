import argparse
import librosa
import soundfile as sf

from .onnxrvc import OnnxRVC

def run() -> None:
    parser = argparse.ArgumentParser(
        prog="rvc-onnx",
        description="Run RVC inference with ONNX models"
    )
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to save output audio file")
    parser.add_argument("--sid", type=int, default=0, help="Speaker ID")
    parser.add_argument("--f0-up-key", type=float, default=0.0, help="Pitch shift amount")
    parser.add_argument("--model-path", default="models/rvc.onnx", help="Path to rvc ONNX model")
    parser.add_argument("--vec-path", default="models/vec-768-layer-12.onnx", help="Path to ContentVec model")
    parser.add_argument("--rmvpe-path", default="models/rmvpe.onnx", help="Path to RMVPE model")

    args = parser.parse_args()

    model = OnnxRVC(
        model_path=args.model_path,
        vec_path=args.vec_path,
        rmvpe_path=args.rmvpe_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    out_audio = model.inference(args.input, sid=args.sid, f0_up_key=args.f0_up_key)
    sf.write(args.output, out_audio, 40000)
    print(f"✅ Saved output to {args.output}")

