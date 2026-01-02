import pytest
import torch
import numpy as np
from neural_encoder.eeg_preprocessor import EEGPreprocessor
from neural_encoder.eeg_encoder import EEGNet


def test_preprocessor_filter():
    """
    Verify bandpass filter frequency response.
    """
    processor = EEGPreprocessor()
    # Create a dummy signal with 50Hz noise (outside 1-40Hz band)
    fs = processor.sampling_rate
    t = np.linspace(0, 1, fs, endpoint=False)
    sig_10hz = np.sin(2 * np.pi * 10 * t)  # Keep
    sig_50hz = np.sin(2 * np.pi * 50 * t)  # Reject
    combined = sig_10hz + sig_50hz

    # Needs shape [channels, samples]
    raw = np.tile(combined, (32, 1))

    filtered = processor._apply_filter(raw)

    # Check if 50Hz component is attenuated
    # Simple RMS check
    rms_original = np.sqrt(np.mean(combined**2))
    rms_filtered = np.sqrt(np.mean(filtered[0] ** 2))

    assert rms_filtered < rms_original * 0.8  # Expect significant reduction


def test_encoder_shape():
    batch = 4
    channels = 32
    samples = 250
    input_tensor = torch.randn(batch, channels, samples)

    model = EEGNet()
    output = model(input_tensor)

    assert output.shape == (batch, 768)


def test_encoder_gradients():
    batch = 2
    channels = 32
    samples = 250
    input_tensor = torch.randn(batch, channels, samples, requires_grad=True)

    model = EEGNet()
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    assert input_tensor.grad is not None
