# RVC-ONNX

RVC-ONNX is a lightweight inference wrapper around ONNX Runtime for running Retrieval-based Voice Conversion (RVC) models.  
This project supports RVC models and performs inference in FP16 for faster execution and reduced memory usage.

## Features

- RVC model inference with ONNX Runtime
- FP16 inference for performance
- F0 extraction with RMVPE
- Command-line interface for quick usage

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for packaging and execution.

Clone the repository:
```bash
git clone https://github.com/dev6699/rvc-onnx.git
cd rvc-onnx
uv sync
```

Get models from: [here](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)
- vec-256-layer-9.onnx (for V1)
- vec-768-layer-12.onnx (for V2)
- rmvpe.onnx

Usage:
```bash
# For V1 model
uv run rvc-onnx --input assets/input_1.wav --output assets/output_1.wav --model-path models/v1.onnx --vec-path models/vec-256-layer-9.onnx

# For V2 model
uv run rvc-onnx --input assets/input_1.wav --output assets/output_1.wav --model-path models/v2.onnx --vec-path models/vec-768-layer-12.onnx

# Help
uv run rvc-onnx --help
usage: rvc-onnx [-h] --input INPUT --output OUTPUT [--sid SID] [--f0-up-key F0_UP_KEY]
                [--model-path MODEL_PATH] [--vec-path VEC_PATH] [--rmvpe-path RMVPE_PATH]

Run RVC inference with ONNX models

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to input audio file
  --output OUTPUT       Path to save output audio file
  --sid SID             Speaker ID
  --f0-up-key F0_UP_KEY
                        Pitch shift amount
  --model-path MODEL_PATH
                        Path to rvc ONNX model
  --vec-path VEC_PATH   Path to ContentVec model
  --rmvpe-path RMVPE_PATH
                        Path to RMVPE model
```

## Result
| Input | Output |
|-------|---------|
| <audio controls src="assets/input_1.wav"></audio> | <audio controls src="assets/output_1.wav"></audio> |
| <audio controls src="assets/input_2.wav"></audio> | <audio controls src="assets/output_2.wav"></audio> |


## License

[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/dev6699/rntv/blob/main/LICENSE)

This project is licensed under the terms of the [MIT license](/LICENSE).