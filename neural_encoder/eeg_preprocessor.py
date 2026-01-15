import numpy as np
import yaml


# import pylsl # Commented out to avoid runtime error if liblsl not present
# import mne
import scipy.signal
# from mne.preprocessing import ICA

# Load config
with open("configs/eeg_config.yaml", "r") as f:
    config = yaml.safe_load(f)["neural_encoder"]


class EEGPreprocessor:
    def __init__(self):
        self.sampling_rate = config["sampling_rate"]
        self.bandpass = config["bandpass"]
        self.epoch_window = config["epoch_window"]
        self.ica_components = config["ica_components"]
        self.streams = []

        # MNE Info visualization structure (mocked)
        # self.info = mne.create_info(ch_names=32, sfreq=self.sampling_rate, ch_types='eeg')

    def connect_stream(self, name: str = "OpenBCI"):
        """
        Connect to LSL stream.
        """
        print(f"Connecting to LSL stream: {name} (Mocked)")
        # streams = pylsl.resolve_stream('type', 'EEG')
        # self.inlet = pylsl.StreamInlet(streams[0])
        self.inlet = None

    def fetch_data(self, duration: float = 1.0) -> np.ndarray:
        """
        Fetch raw data chunk. Returns [channels, samples]
        """
        n_samples = int(duration * self.sampling_rate)
        # if self.inlet:
        #     sample, timestamp = self.inlet.pull_chunk()
        #     return np.array(sample).T

        # Return synthetic noise for now
        return np.random.randn(32, n_samples)

    def process_epoch(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply filtering, artifact rejection, and scaling.
        Input: [channels, timepoints]
        """
        # 1. Bandpass Filter
        filtered = self._apply_filter(raw_data)

        # 2. ICA Artifact Rejection (Disabled)
        # ica = ICA(n_components=self.ica_components, method='fastica')
        # ica.fit(mne_raw)
        # cleaned = ica.apply(mne_raw)
        cleaned = filtered  # ICA skipped for performance

        # 3. Epoching handled by upstream usually, but here we process window
        # 4. robust scaling (simple standardization for now)
        mean = np.mean(cleaned, axis=1, keepdims=True)
        std = np.std(cleaned, axis=1, keepdims=True)
        scaled = (cleaned - mean) / (std + 1e-6)

        return scaled

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        """
        b, a = scipy.signal.butter(
            4, self.bandpass, btype="bandpass", fs=self.sampling_rate
        )
        # Apply filter along time axis (last axis)
        return scipy.signal.filtfilt(b, a, data, axis=-1)
