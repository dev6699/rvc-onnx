import onnxruntime
import numpy as np
from typing import Optional, List

class ContentVec:
    """
    ONNX Runtime wrapper for ContentVec model.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.
    providers : (Optional[List[str]])
        List of execution providers.
    """
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        
        available_providers = onnxruntime.get_available_providers()

        if providers is None:
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.model = onnxruntime.InferenceSession(model_path, providers=providers)

    def forward(self, wav: np.ndarray) -> np.ndarray:
        """
        Run inference on input waveform.

        Parameters
        ----------
        wav : np.ndarray
            Input waveform (1D or 2D).
            - If 2D, it will average across channels to mono.

        Returns
        -------
        result : np.ndarray
            Model output of shape (batch, time, features).
        """
        feats = wav
        if feats.ndim == 2:
            feats = feats.mean(-1)  # mix stereo -> mono
        assert feats.ndim == 1, f"Expected 1D waveform, got {feats.ndim}D"

        # Shape: (1, 1, T)
        feats = np.expand_dims(np.expand_dims(feats.astype(np.float32), 0), 0)

        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)[0]

        # Transpose to (batch, time, features)
        return logits.transpose(0, 2, 1)