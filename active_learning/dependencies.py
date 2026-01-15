from active_learning.active_learner import ActiveLearner
import logging

logger = logging.getLogger(__name__)

_active_learner_instance = None


def get_active_learner() -> ActiveLearner:
    global _active_learner_instance
    if _active_learner_instance is None:
        logger.info("Initializing Global ActiveLearner instance")
        _active_learner_instance = ActiveLearner()
        # Initialize with a larger pool for the demo
        _active_learner_instance.initialize_pools(1000)
    return _active_learner_instance
