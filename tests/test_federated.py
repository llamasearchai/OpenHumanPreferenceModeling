import pytest
import numpy as np
from privacy_engine.federated_learning.coordinator import Coordinator
from privacy_engine.federated_learning.client import Client


def test_federated_round():
    coordinator = Coordinator(num_clients=10, fraction_fit=0.5)

    # Check initial model
    initial_sum = np.sum(coordinator.global_model)

    # Start round
    selected_client_ids, global_model_payload = coordinator.start_round()
    assert len(selected_client_ids) == 5

    # Clients train
    encrypted_updates = []
    for cid in selected_client_ids:
        client = Client(cid)
        update = client.train(global_model_payload)
        encrypted_updates.append(update)

    # Coordinator aggregates
    new_model = coordinator.aggregate_gradients(encrypted_updates)

    # Check model changed
    new_sum = np.sum(new_model)
    assert new_sum != initial_sum
    status_dict = coordinator.get_status()
    assert status_dict
    assert status_dict["round"] == 1
    assert "Epsilon" in status_dict["privacy_status"]
