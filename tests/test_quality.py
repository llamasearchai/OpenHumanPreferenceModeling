from annotation_interface.backend.quality_control import detect_spam, compute_agreement
from annotation_interface.backend.models import Annotation
import datetime


def test_detect_spam_speed():
    # create 6 annotations very fast
    anns = []
    now = datetime.datetime.now()
    for i in range(6):
        a = Annotation(
            task_id=f"t{i}",
            annotator_id="spammer",
            annotation_type="pairwise",
            response_data={"winner": "A"},
            time_spent_seconds=1.0,  # Too fast
            confidence=5,
            created_at=now,
        )
        anns.append(a)

    warnings = detect_spam(anns)
    assert len(warnings) > 0
    assert "too fast" in warnings[0]


def test_detect_spam_variance():
    # 10 annotations all "A"
    anns = []
    now = datetime.datetime.now()
    for i in range(10):
        a = Annotation(
            task_id=f"t{i}",
            annotator_id="bot",
            annotation_type="pairwise",
            response_data={"winner": "A"},  # No variance
            time_spent_seconds=10.0,
            confidence=5,
            created_at=now,
        )
        anns.append(a)

    warnings = detect_spam(anns)
    assert any("No variation" in w for w in warnings)


def test_compute_agreement():
    # 3 annotations for same task? No, function takes list for ONE task usually based on usage.
    # Our implementation: takes list, computes majority agreement on that list.

    anns = []
    # 3 say A, 1 says B
    for _ in range(3):
        anns.append(
            Annotation(
                task_id="t1",
                annotator_id="u",
                annotation_type="p",
                response_data={"winner": "A"},
                time_spent_seconds=1,
                confidence=1,
            )
        )
    anns.append(
        Annotation(
            task_id="t1",
            annotator_id="u2",
            annotation_type="p",
            response_data={"winner": "B"},
            time_spent_seconds=1,
            confidence=1,
        )
    )

    score = compute_agreement(anns)
    # Majority is A (3 votes). Total 4. 3/4 = 0.75
    assert score == 0.75
