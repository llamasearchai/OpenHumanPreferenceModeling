from calibration.prediction_store import PredictionStore


def test_prediction_store_record_and_consume(tmp_path):
    path = tmp_path / "predictions.sqlite"
    store = PredictionStore(str(path))

    stored = store.record(0.8, 1, sample_rate=1.0)
    assert stored is True
    assert store.count() == 1

    confidences, corrects = store.consume(10)
    assert confidences == [0.8]
    assert corrects == [1]
    assert store.count() == 0


def test_prediction_store_sampling_skip(tmp_path):
    path = tmp_path / "predictions.sqlite"
    store = PredictionStore(str(path))

    stored = store.record(0.6, 0, sample_rate=0.0)
    assert stored is False
    assert store.count() == 0
