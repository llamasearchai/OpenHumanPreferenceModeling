import numpy as np
import scipy.signal


def generate_noise(n_samples: int, color="pink") -> np.ndarray:
    """
    Generate colored noise.
    """
    if color == "pink":
        # Simple pink noise approximation: 1/f
        uneven = n_samples % 2
        X = np.random.randn(n_samples // 2 + 1 + uneven) + 1j * np.random.randn(
            n_samples // 2 + 1 + uneven
        )
        S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid div by zero
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        return y
    return np.random.randn(n_samples)


def generate_synthetic_eeg(
    duration_sec: int = 10, sampling_rate: int = 250, channels: int = 32
):
    """
    Generates synthetic EEG data with alpha waves (8-12Hz) and artifacts.
    """
    n_samples = duration_sec * sampling_rate
    time = np.linspace(0, duration_sec, n_samples)

    data = np.zeros((channels, n_samples))

    for ch in range(channels):
        # 1. Background Pink Noise
        noise = generate_noise(n_samples, "pink") * 5e-6  # 5 microvolts

        # 2. Alpha Rhythm (10Hz)
        alpha = np.sin(2 * np.pi * 10 * time) * 10e-6
        # Modulate alpha amplitude slowly
        modulation = np.sin(2 * np.pi * 0.1 * time) + 1
        alpha = alpha * modulation

        # 3. Artifacts (Blinks)
        metrics = np.zeros_like(time)
        if ch < 2:  # Frontal channels
            # inject blinks every ~3 sec
            blink_times = np.arange(1, duration_sec, 3)
            for t_blink in blink_times:
                idx = int(t_blink * sampling_rate)
                if idx < n_samples:
                    # Gauss pulse
                    width = int(0.2 * sampling_rate)
                    metrics[idx : idx + width] += (
                        100e-6 * scipy.signal.windows.gaussian(width, std=width / 6)
                    )

        data[ch] = noise + alpha + metrics[:n_samples]

    return data


if __name__ == "__main__":
    eeg = generate_synthetic_eeg()
    print(f"Generated EEG: {eeg.shape}")
