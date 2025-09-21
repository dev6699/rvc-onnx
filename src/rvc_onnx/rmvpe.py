import onnxruntime
import numpy as np
from typing import List, Optional

from rvc_onnx.mel import MelSpectrogram

class RMVPE:
    """
    RMVPE (Robust Multi-View Pitch Estimation)

    Uses a pretrained ONNX model to estimate F0 (fundamental frequency) 
    from raw audio by extracting mel spectrograms, passing them through
    the network, and decoding pitch in cents.

    Parameters
    ----------
    model_path : str
        Path to ONNX model file.
    providers : (Optional[List[str]])
        List of execution providers.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        # Mel-spectrogram extractor
        self.mel_extractor = MelSpectrogram(
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=None,
            mel_fmin=30,
            mel_fmax=8000,
        )

        self.model = onnxruntime.InferenceSession(
            model_path, providers=providers
        )

        # Cents mapping for decoding (with padding for local averaging)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping: np.ndarray = np.pad(cents_mapping, (4, 4))  # (368,)

    def mel2hidden(self, mel: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram into hidden features using the ONNX model.

        Parameters
        ----------
        mel : np.ndarray
            Mel spectrogram of shape (n_mels, time) or (1, n_mels, time).

        Returns
        -------
        hidden : np.ndarray
            Hidden representation of shape (batch, frames, bins).
        """
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames

        # Pad along time axis
        if n_pad > 0:
            mel = np.pad(mel, ((0, 0), (0, n_pad)), mode="constant")

        # Ensure input has batch dimension
        if mel.ndim == 2:
            mel = np.expand_dims(mel, 0)

        onnx_input_name = self.model.get_inputs()[0].name
        onnx_outputs_name = self.model.get_outputs()[0].name

        hidden: np.ndarray = self.model.run(
            [onnx_outputs_name],
            input_feed={onnx_input_name: mel.astype(np.float32)},
        )[0]

        return hidden[:, :, :n_frames]

    def decode(self, hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """
        Decode hidden features into F0 values.

        Parameters
        ----------
        hidden : np.ndarray
            Hidden salience representation (frames, bins).
        thred : float, default=0.03
            Threshold for salience confidence.

        Returns
        -------
        f0 : np.ndarray
            Estimated fundamental frequency (Hz) per frame.
        """
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))  # cents → Hz
        f0[f0 == 10] = 0  # mask low-confidence regions
        return f0

    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """
        Run pitch estimation directly from audio.

        Parameters
        ----------
        audio : np.ndarray
            1D numpy array of raw audio waveform (float32).
        thred : float, default=0.03
            Threshold for salience confidence.

        Returns
        -------
        f0 : np.ndarray
            Estimated fundamental frequency (Hz) per frame.
        """
        mel = self.mel_extractor.forward(audio.astype(np.float32), center=True)
        hidden = self.mel2hidden(mel).squeeze(0)  # remove batch dim
        f0 = self.decode(hidden, thred=thred)
        return f0

    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
        """
        Compute local average of pitch salience in cents.

        Parameters
        ----------
        salience : np.ndarray
            Salience map of shape (frames, bins).
        thred : float, default=0.05
            Minimum peak confidence.

        Returns
        -------
        cents : np.ndarray
            Estimated pitch in cents (0 if below threshold).
        """
        # Find argmax along bins (per frame)
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4

        todo_salience, todo_cents_mapping = [], []
        for idx in range(salience.shape[0]):
            start, end = center[idx] - 4, center[idx] + 5
            todo_salience.append(salience[idx, start:end])
            todo_cents_mapping.append(self.cents_mapping[start:end])

        todo_salience = np.array(todo_salience)        # (frames, 9)
        todo_cents_mapping = np.array(todo_cents_mapping)  # (frames, 9)

        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)

        averaged = product_sum / weight_sum
        max_val = np.max(salience, axis=1)
        averaged[max_val <= thred] = 0
        return averaged
