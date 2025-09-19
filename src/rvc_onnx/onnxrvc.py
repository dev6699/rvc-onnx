import librosa
import onnxruntime
import numpy as np
from scipy import signal
from typing import List, Optional

from .rmvpe import RMVPE
from .vec import ContentVec

class OnnxRVC:
    """
    OnnxRVC

    Parameters
    ----------
    model_path : str
        Path to main ONNX model.
    vec_path : str
        Path to ContentVec model.
    rmvpe_path : str
        Path to RMVPE model.
    providers : Optional[List[str]]
        Execution providers.
            Defaults to CUDA if available, otherwise CPU.
    """
    def __init__(
        self,
        model_path: str,
        vec_path: str,
        rmvpe_path: str,
        providers: Optional[List[str]] = None,
    ):
        self.vec_path = vec_path
        self.rmvpe_path = rmvpe_path

        available_providers = onnxruntime.get_available_providers()
        if providers is None:
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_opts.enable_profiling = True
        # sess_opts.enable_mem_pattern = False
        # sess_opts.enable_cpu_mem_arena = False
        self.model = onnxruntime.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

        # --- Audio and windowing params ---
        self.sr = 16000                # Input sample rate (Hubert/ContentVec)
        self.tgt_sr = 40000            # Target sample rate
        self.window = 160              # Samples per frame (at sr=16k)

        # Padding / slicing windows
        self.x_pad = 3
        self.x_query = 10
        self.x_center = 50
        self.x_max = 50

        # Derived params (in samples)
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = self.tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max

        # Processing params
        self.rms_mix_rate = 0.25
        self.protect = 0.33

    def forward(self, 
                hubert: np.ndarray, 
                hubert_length: np.ndarray, 
                pitch: np.ndarray, 
                pitchf: np.ndarray, 
                ds: np.ndarray, 
                rnd: np.ndarray):
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
 
        return self.model.run(None, onnx_input)[0]

    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio by:
        1. Normalizing amplitude
        2. Applying a high-pass Butterworth filter
        3. Padding with reflection for windowed processing

        Parameters
        ----------
        audio : np.ndarray
            Input waveform (1D)

        Returns
        -------
        audio_pad: np.ndarray
            Preprocessed + padded audio
        """
        # --- Step 1: Normalize amplitude ---
        # Ensure the waveform peak does not exceed ~0.95
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max

        # --- Step 2: High-pass filter (remove DC & rumble < 48 Hz) ---
        bh, ah = signal.butter(
            N=5,             # 5th order filter
            Wn=48,           # cutoff frequency at 48 Hz
            btype="high",    # high-pass filter
            fs=self.sr
        )
        # Zero-phase filtering (applied forward & backward → no phase distortion)
        audio = signal.filtfilt(bh, ah, audio)

        # --- Step 3: Reflection padding ---
        # Pad both sides by window//2 to prevent edge issues in later processing
        audio_pad = np.pad(
            audio,
            (self.window // 2, self.window // 2),
            mode="reflect"
        )

        return audio_pad

    def find_optimal_timestamps(self, audio: np.ndarray, audio_pad: np.ndarray):
        """
        Find optimal timestamps for cutting long audio into segments.
        Cuts are placed near local minima in the audio energy (silence regions).

        Parameters
        ----------
        audio : np.ndarray
            Original audio
        audio_pad : np.ndarray
            Padded audio

        Returns
        -------
        timestamps: list[int]
            Optimal cut timestamps
        """
        opt_ts = []

        # Only segment if audio length exceeds maximum threshold
        if audio_pad.shape[0] > self.t_max:
            # Build energy-like envelope
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                # Accumulate absolute values across shifted windows
                audio_sum += np.abs(audio_pad[i : i + audio.shape[0]])

            # Step through audio in strides of t_center
            for t in range(self.t_center, audio.shape[0], self.t_center):
                search_window = audio_sum[t - self.t_query : t + self.t_query]
                min_index = np.argmin(search_window)
                cut_point = t - self.t_query + min_index
                opt_ts.append(cut_point)

        audio_pad_reflect = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        return opt_ts, audio_pad_reflect

    def extract_f0(self, audio_pad: np.ndarray, f0_up_key: int = 0):
        """
        Extract and normalize fundamental frequency (F0) from audio.

        Parameters
        ----------
        audio_pad : np.ndarray
            Preprocessed + padded audio
        f0_up_key : int
            Pitch shift in semitones (e.g. 12 = +1 octave)

        Returns
        -------
        pitchf : np.ndarray
            Continuous F0 values [1, T] (Hz, float16)
        pitch : np.ndarray
            Quantized F0 bins [1, T] (int64, range 1-255)
        """

        # --- Step 1: Extract F0 using RMVPE ---
        f0_predictor = RMVPE(self.rmvpe_path)
        pitchf = f0_predictor.infer_from_audio(audio_pad, thred=0.03)
        del f0_predictor  # free GPU memory

        # --- Step 2: Apply pitch shift ---
        pitchf = pitchf * (2 ** (f0_up_key / 12))
        pitch = pitchf.copy()

        # --- Step 3: Normalize to Mel scale bins (1–255) ---
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0_mel = 1127 * np.log(1 + pitchf / 700)
        f0_mel[f0_mel > 0] = (
            (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min)
        ) + 1

        # Clip + convert to int bins
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)

        # --- Step 4: Frame alignment + reshape ---
        p_len = audio_pad.shape[0] // self.window
        pitch = pitch[:p_len].reshape(1, -1)
        pitchf = pitchf[:p_len].reshape(1, -1).astype(np.float16)

        return pitchf, pitch

    def change_rms(self, 
                   data1: np.ndarray, sr1: int,
                   data2: np.ndarray, sr2: int,
                   rate: float) -> np.ndarray:
        """
        Adjust the RMS energy of data2 to follow data1's RMS profile.

        Parameters
        ----------
        data1 : np.ndarray
            Reference audio signal (input).
        sr1 : int
            Sample rate of data1.
        data2 : np.ndarray
            Target audio signal (output, will be scaled).
        sr2 : int
            Sample rate of data2.
        rate : float
            Blending ratio:
                0 → match data1 fully,
                1 → keep data2 as is,
                between 0 and 1 → interpolate.

        Returns
        -------
        np.ndarray
            RMS-adjusted audio (same shape as data2).
        """

        # Compute RMS every 0.5s
        hop1 = sr1 // 2
        hop2 = sr2 // 2
        rms1 = librosa.feature.rms(y=data1,
                                   frame_length=hop1 * 2,
                                   hop_length=hop1).flatten()
        rms2 = librosa.feature.rms(y=data2,
                                   frame_length=hop2 * 2,
                                   hop_length=hop2).flatten()

        # Interpolate RMS curves to match data2 length
        x_old1 = np.linspace(0, len(data2), num=len(rms1))
        x_old2 = np.linspace(0, len(data2), num=len(rms2))
        x_new = np.arange(len(data2))

        rms1 = np.interp(x_new, x_old1, rms1)
        rms2 = np.interp(x_new, x_old2, rms2)

        # Avoid divide by zero
        rms2 = np.maximum(rms2, 1e-6)

        # Compute gain curve
        gain = np.power(rms1, 1 - rate) * np.power(rms2, rate - 1)

        # Apply gain
        return data2 * gain
    
    def run(self, audio_pad: np.ndarray, pitch: np.ndarray, pitchf: np.ndarray, sid: int):
        """
        Run full inference pipeline:
        - Extract features with ContentVec
        - Apply pitch conditioning & protection
        - Run synthesis forward pass
        - Apply RMS normalization and post-processing

        Parameters
        ----------
        audio_pad : np.ndarray
            Input waveform (1D, float32).
        pitch : np.ndarray
            Pitch indices [1, T].
        pitchf : np.ndarray
            Continuous pitch [1, T].
        sid : int
            Speaker ID.

        Returns
        -------
        np.ndarray: 
            Final audio waveform (int16).
        """
        # --- Feature extraction ---
        vec_model = ContentVec(self.vec_path, self.providers)
        feats = vec_model.forward(audio_pad.astype(np.float32))
        del vec_model # release ONNX session

        # Expand features (repeat along feature axis)
        feats = np.repeat(feats, 2, axis=2).transpose(0, 2, 1).astype(np.float16)
        feats_length = feats.shape[1]

        # --- Align feature length with pitch length ---
        p_len = audio_pad.shape[0] // self.window
        if feats_length < p_len:
            p_len = feats_length

        pitch = pitch[:, :p_len]
        pitchf = pitchf[:, :p_len]

        # --- Protect unvoiced regions ---
        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.copy()  
            pitchff = pitchf.copy()
            pitchff[pitchf > 0] = 1.0
            pitchff[pitchf < 1] = self.protect

            pitchff = pitchff[..., np.newaxis]
            feats = feats * pitchff + feats0 * (1.0 - pitchff)
            feats = feats.astype(feats0.dtype)
        
        # --- Prepare additional inputs ---
        ds = np.array([sid]).astype(np.int64)
        rnd = np.random.randn(1, 192, p_len).astype(np.float16) 
        p_len = np.array([p_len]).astype(np.int64)
        
        # --- Model forward pass ---
        audio_opt = self.forward(
            feats, 
            p_len, 
            pitch, 
            pitchf, 
            ds, 
            rnd
        ).squeeze()

        # --- Post-process audio ---
        audio_opt = self.change_rms(audio_pad, self.sr, audio_opt, self.tgt_sr, self.rms_mix_rate)
        
        # Normalize to int16
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        # Trim padding
        audio_opt = audio_opt[self.t_pad_tgt : -self.t_pad_tgt]

        return audio_opt
    
    def infer_segments(self, 
                       audio_pad: np.ndarray, 
                       pitch: np.ndarray, 
                       pitchf: np.ndarray, 
                       opt_ts: list, 
                       sid: int) -> list:
        """
        Run model inference on audio, segmented by opt_ts (optimal cut points).

        Parameters
        ----------
        audio_pad : np.ndarray
            Preprocessed + padded audio
        pitch : np.ndarray
            Quantized pitch features [1, T]
        pitchf : np.ndarray
            Continuous pitch features [1, T]
        opt_ts : list[int]
            Optimal segmentation timestamps
        sid: Speaker ID (or other conditioning input)

        Returns
        -------
        list: Inference results for each segment
        """
        results = []
        s = 0  # start index in samples

        for t in opt_ts:
            # Align cut to window boundary
            t_aligned = (t // self.window) * self.window

            # Convert to frame indices
            start_win = s // self.window
            end_win = (t_aligned + self.t_pad2) // self.window

            # Slice inputs
            _audio_pad = audio_pad[s : t_aligned + self.t_pad2 + self.window]
            _pitch = pitch[:, start_win:end_win]
            _pitchf = pitchf[:, start_win:end_win]

            # Run inference for this chunk
            results.append(self.run(_audio_pad, _pitch, _pitchf, sid))

            # Update start for next segment
            s = t_aligned

        # Handle leftover tail
        if s < len(audio_pad):
            start_win = s // self.window
            results.append(
                self.run(audio_pad[s:], pitch[:, start_win:], pitchf[:, start_win:], sid)
            )

        return results

    def inference(
        self,
        audio: np.ndarray, 
        sid: int,
        f0_up_key: float,
    ):
        """
        Run full inference pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input waveform (1D float array).
        sid : int
            Speaker ID.
        f0_up_key : float
            Pitch shift (in semitones).

        Returns
        -------
        np.ndarray: 
            Final synthesized waveform.
        """
        audio_pad = self.preprocess(audio)
        opt_ts, audio_pad = self.find_optimal_timestamps(audio, audio_pad)
        pitchf, pitch = self.extract_f0(audio_pad, f0_up_key)
        results = self.infer_segments(audio_pad, pitch, pitchf, opt_ts, sid)
        final_output = np.concatenate(results)
        return final_output
    