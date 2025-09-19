
from typing import Optional
import numpy as np
import scipy.signal
import librosa

class MelSpectrogram:
    """
    Compute a log-Mel spectrogram from raw audio.

    Parameters
    ----------
    n_mel_channels : int
        Number of mel filterbank channels (mel bins).
    sampling_rate : int
        Audio sampling rate in Hz.
    win_length : int
        Window size (in samples) for STFT.
    hop_length : int
        Hop size (in samples) for STFT.
    n_fft : Optional[int], default=None
        FFT size. If None, defaults to win_length.
    mel_fmin : float, default=0
        Minimum frequency (Hz) for the mel filterbank.
    mel_fmax : Optional[float], default=None
        Maximum frequency (Hz) for the mel filterbank.
    clamp : float, default=1e-5
        Minimum value used before log to avoid log(0).
    """

    def __init__(
        self,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: float = 0,
        mel_fmax: Optional[float] = None,
        clamp: float = 1e-5,
    ) -> None:
        n_fft = win_length if n_fft is None else n_fft

        # Precompute mel filterbank matrix
        mel_basis: np.ndarray = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)

        self.mel_basis: np.ndarray = mel_basis
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.sampling_rate: int = sampling_rate
        self.n_mel_channels: int = n_mel_channels
        self.clamp: float = clamp

    def forward(
        self,
        audio: np.ndarray,
        keyshift: int = 0,
        speed: float = 1,
        center: bool = True,
    ) -> np.ndarray:
        """
        Convert raw audio to a log-Mel spectrogram.

        Parameters
        ----------
        audio : np.ndarray
            1D numpy array containing the audio waveform.
        keyshift : int, default=0
            Pitch shift in semitones (affects FFT size and window length).
        speed : float, default=1.0
            Time-stretch factor (affects hop length).
        center : bool, default=True
            Whether to pad the input so that frames are centered.

        Returns
        -------
        log_mel_spec : np.ndarray
            2D array of shape (n_mel_channels, time) containing
            the log-Mel spectrogram.
        """
        # Adjust FFT and window size based on keyshift
        factor: float = 2 ** (keyshift / 12)
        n_fft_new: int = int(np.round(self.n_fft * factor))
        win_length_new: int = int(np.round(self.win_length * factor))
        hop_length_new: int = int(np.round(self.hop_length * speed))

        # Create Hann window for STFT
        hann_window: np.ndarray = scipy.signal.windows.hann(win_length_new, sym=False)

        # Compute STFT (complex values)
        stft_result: np.ndarray = librosa.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=hann_window,
            center=center,
        )
        magnitude: np.ndarray = np.abs(stft_result)  # magnitude spectrogram

        # Adjust frequency bins if keyshift was applied
        if keyshift != 0:
            size: int = self.n_fft // 2 + 1
            if magnitude.shape[0] < size:
                pad_amount: int = size - magnitude.shape[0]
                magnitude = np.pad(magnitude, ((0, pad_amount), (0, 0)), mode="constant")
            magnitude = magnitude[:size, :] * (self.win_length / win_length_new)

        # Project magnitude spectrogram onto mel filterbank
        mel_output: np.ndarray = np.dot(self.mel_basis, magnitude)

        # Convert to log scale (avoid log(0) with clamp)
        log_mel_spec: np.ndarray = np.log(np.maximum(mel_output, self.clamp))
        return log_mel_spec
