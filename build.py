import platform
import subprocess

target = "src/rvc_onnx/cli.py"
cmd = ["uv", "run", "pyinstaller", "--onefile", target]

print(f"Building for {platform.system()}...")
subprocess.run(cmd, check=True)
