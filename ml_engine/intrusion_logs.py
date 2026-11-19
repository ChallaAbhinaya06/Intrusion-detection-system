"""
Intrusion Logging & Analytics Module

Persists and aggregates IDS results with versioning support.

Key design:
- log_predictions receives label_mapping explicitly — never relies on global state.
- Benign traffic (any label containing "benign" or "normal") is always stored
  under the canonical AttackType(attack_name="Normal") so dashboard queries
  work consistently across datasets.
"""

from django.db.models import Count, Max
from django.utils import timezone
from core.models import Dataset, AttackType, AttackPattern, IntrusionLog, TestRun
from ml_engine.attack_knowledge_base import get_attack_info, set_label_mapping


# Keywords that identify benign/normal traffic regardless of dataset
_BENIGN_KEYWORDS = {"benign", "normal"}


def _is_benign(label_name):
    """Return True if this label represents normal/benign traffic."""
    return any(kw in str(label_name).lower() for kw in _BENIGN_KEYWORDS)


def _fix_stale_benign_records():
    """
    One-time repair: migrate any AttackType rows named "Benign", "BENIGN", etc.
    into the canonical "Normal" row so existing DB data is consistent.
    """
    normal_type, _ = AttackType.objects.get_or_create(
        attack_name="Normal",
        defaults={"severity": "None", "description": "Benign network traffic"}
    )
    # Ensure severity is correct even if record already existed
    if normal_type.severity != "None":
        normal_type.severity = "None"
        normal_type.save(update_fields=["severity"])

    # Find stale benign rows and re-point their IntrusionLogs to Normal
    for at in AttackType.objects.exclude(id=normal_type.id):
        if _is_benign(at.attack_name):
            IntrusionLog.objects.filter(attack_type=at).update(attack_type=normal_type)
            at.delete()


# =====================================================
# WRITE: Log predictions with test run versioning
# =====================================================
def log_predictions(dataset_name, dataset_size, y_true, y_pred,
                    user=None, run_type='test', accuracy=None, label_mapping=None):
    """
    Log predictions with test run versioning.
    Each call creates a new TestRun record.

    Parameters
    ----------
    label_mapping : dict {int -> str}
        Maps class_id -> original label name from the dataset.
        Passed explicitly from train_eval so this function never
        depends on the module-level global in attack_knowledge_base.
    """
    # Set the mapping so get_attack_info works for pattern/severity lookups
    if label_mapping:
        set_label_mapping(label_mapping)

    # Repair any stale Benign/BENIGN AttackType records from old runs
    _fix_stale_benign_records()

    # Get or create dataset record
    dataset_obj, created = Dataset.objects.get_or_create(
        dataset_name=dataset_name,
        defaults={"dataset_size": dataset_size, "dataset_path": ""}
    )
    if not created and dataset_obj.dataset_size != dataset_size:
        dataset_obj.dataset_size = dataset_size
        dataset_obj.save(update_fields=["dataset_size"])

    # Create a new TestRun for this execution
    test_run = TestRun.objects.create(
        dataset=dataset_obj,
        user=user,
        run_type=run_type,
        accuracy=accuracy,
        status='running'
    )

    logs = []

    for true_label, pred_label in zip(y_true, y_pred):
        pred_id = int(pred_label)

        # Resolve label name DIRECTLY from the mapping dict passed in
        # This is the ground truth — no global state involved
        if label_mapping and pred_id in label_mapping:
            label_name = str(label_mapping[pred_id])
        else:
            # Fallback when no mapping available
            label_name = get_attack_info(pred_id)["attack_type"]

        # Get pattern/severity info for this label
        attack_info = get_attack_info(pred_id)

        # ---- BENIGN TRAFFIC ----
        # Covers: "Benign" (5G-NIDD), "BENIGN", "Normal" (CIC-IDS2017), "normal", etc.
        if _is_benign(label_name):
            attack_type, _ = AttackType.objects.get_or_create(
                attack_name="Normal",
                defaults={
                    "severity":    "None",
                    "description": "Benign network traffic"
                }
            )
            attack_pattern = None

        # ---- ATTACK TRAFFIC ----
        else:
            attack_type, _ = AttackType.objects.get_or_create(
                attack_name=label_name,
                defaults={
                    "severity":    attack_info["severity"],
                    "description": attack_info["signature"]
                }
            )
            attack_pattern, _ = AttackPattern.objects.get_or_create(
                pattern_name=attack_info["pattern_name"],
                defaults={
                    "signature_details": attack_info["signature"]
                }
            )

        logs.append(
            IntrusionLog(
                dataset=dataset_obj,
                test_run=test_run,
                attack_type=attack_type,
                attack_pattern=attack_pattern,
                confidence_score=1.0
            )
        )

    # Bulk insert for performance
    IntrusionLog.objects.bulk_create(logs, batch_size=1000)

    # Mark run as completed
    test_run.status = 'completed'
    test_run.completed_at = timezone.now()
    test_run.save(update_fields=['status', 'completed_at'])

    return test_run


# =====================================================
# READ: Dashboard Analytics
# =====================================================
def get_attack_vs_normal_count(test_run_id=None):
    """
    Return (attack_count, normal_count) for the latest run (or a specific run).
    All benign traffic is stored as attack_name="Normal" by log_predictions.
    """
    if test_run_id:
        queryset = IntrusionLog.objects.filter(test_run_id=test_run_id)
    else:
        latest_runs = (
            TestRun.objects.filter(status='completed')
            .values('dataset')
            .annotate(latest_id=Max('id'))
            .values_list('latest_id', flat=True)
        )
        queryset = IntrusionLog.objects.filter(test_run_id__in=latest_runs)

    normal_count = queryset.filter(attack_type__attack_name="Normal").count()
    attack_count = queryset.exclude(attack_type__attack_name="Normal").count()
    return attack_count, normal_count


def get_attack_distribution(test_run_id=None):
    """
    Return attack type distribution (excluding Normal) for the latest run.
    """
    if test_run_id:
        queryset = IntrusionLog.objects.filter(test_run_id=test_run_id)
    else:
        latest_runs = (
            TestRun.objects.filter(status='completed')
            .values('dataset')
            .annotate(latest_id=Max('id'))
            .values_list('latest_id', flat=True)
        )
        queryset = IntrusionLog.objects.filter(test_run_id__in=latest_runs)

    qs = (
        queryset
        .exclude(attack_type__attack_name="Normal")
        .values("attack_type__attack_name")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    return [
        {"predicted_class": row["attack_type__attack_name"], "count": row["count"]}
        for row in qs
    ]


def get_test_runs_by_dataset(dataset_name):
    """Get all completed test runs for a specific dataset."""
    return TestRun.objects.filter(
        dataset__dataset_name=dataset_name,
        status='completed'
    ).order_by('-created_at')


def get_latest_test_runs(limit=None):
    """Get all completed test runs ordered by most recent."""
    queryset = (
        TestRun.objects.filter(status='completed')
        .select_related('dataset', 'user')
        .order_by('-created_at')
    )
    return queryset[:limit] if limit else queryset


def has_logs():
    """Check if there are any completed test runs."""
    return TestRun.objects.filter(status='completed').exists()