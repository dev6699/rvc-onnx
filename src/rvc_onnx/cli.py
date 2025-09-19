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
    parser.add_argument("--sr", type=int, default=16000, help="Input sample rate (default=16000)")

    parser.add_argument("--model-path", default="onnx/model.onnx", help="Path to main ONNX model")
    parser.add_argument("--vec-path", default="onnx/vec-768-layer-12.onnx", help="Path to ContentVec model")
    parser.add_argument("--rmvpe-path", default="onnx/rmvpe.onnx", help="Path to RMVPE model")

    args = parser.parse_args()

    model = OnnxRVC(
        model_path=args.model_path,
        vec_path=args.vec_path,
        rmvpe_path=args.rmvpe_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    audio, _ = librosa.load(args.input, sr=args.sr, mono=True)
    out_audio = model.inference(audio, sid=args.sid, f0_up_key=args.f0_up_key)
    sf.write(args.output, out_audio, 40000)
    print(f"✅ Saved output to {args.output}")

