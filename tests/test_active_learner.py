import pytest
from active_learning.active_learner import ActiveLearner


def test_active_learner_initialization():
    learner = ActiveLearner()
    learner.initialize_pools(total_pool_size=200)
    assert len(learner.labeled_pool) > 0
    assert len(learner.unlabeled_pool) > 0
    assert len(learner.labeled_pool) + len(learner.unlabeled_pool) == 200


def test_active_learner_step():
    learner = ActiveLearner()
    learner.batch_size = 10
    learner.budget = 50
    learner.initialize_pools(total_pool_size=100)

    initial_labeled = len(learner.labeled_pool)
    learner.run_step("uncertainty")

    assert len(learner.labeled_pool) == initial_labeled + 10
    assert len(learner.unlabeled_pool) == 100 - (initial_labeled + 10)
    assert learner.budget == 40
