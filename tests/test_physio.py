import pytest
torch = pytest.importorskip("torch")
from neural_encoder.physio_encoder import PhysioEncoder


def test_physio_shape():
    batch = 5
    features = 16
    timepoints = 100

    model = PhysioEncoder()
    input_tensor = torch.randn(batch, features, timepoints)

    output = model(input_tensor)

    assert output.shape == (batch, 768)
