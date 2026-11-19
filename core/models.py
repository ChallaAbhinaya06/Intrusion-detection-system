from django.db import models
from django.contrib.auth.models import User


class Dataset(models.Model):
    dataset_name = models.CharField(max_length=100)
    dataset_path = models.CharField(max_length=255)
    dataset_size = models.IntegerField()

    def __str__(self):
        return self.dataset_name


class AttackType(models.Model):
    attack_name = models.CharField(max_length=50)
    severity = models.CharField(max_length=20)
    description = models.TextField()

    def __str__(self):
        return self.attack_name


class AttackPattern(models.Model):
    pattern_name = models.CharField(max_length=50)
    signature_details = models.TextField()

    def __str__(self):
        return self.pattern_name


class TestRun(models.Model):
    """
    Tracks individual test/training runs for historical analysis
    Each test run creates a new version of results
    """
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='test_runs')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    run_type = models.CharField(
        max_length=10,
        choices=[('train', 'Training'), ('test', 'Testing')],
        default='test'
    )
    accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='running'
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['dataset', '-created_at']),
        ]

    def __str__(self):
        return f"{self.dataset.dataset_name} - {self.run_type} - {self.created_at}"


class IntrusionLog(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    test_run = models.ForeignKey(
        TestRun, 
        on_delete=models.CASCADE, 
        related_name='logs',
        null=True,  # For backward compatibility
        blank=True
    )
    attack_type = models.ForeignKey(AttackType, on_delete=models.CASCADE)
    attack_pattern = models.ForeignKey(
        AttackPattern, 
        on_delete=models.CASCADE,
        null=True,  # Normal traffic doesn't have patterns
        blank=True
    )
    detection_time = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField()

    class Meta:
        permissions = [
            ("can_view_dashboard", "Can view IDS dashboard"),
            ("can_test_model", "Can test IDS model"),
            ("can_train_model", "Can train IDS model"),
        ]
        indexes = [
            models.Index(fields=['test_run', 'attack_type']),
        ]

    def __str__(self):
        return f"{self.attack_type} @ {self.detection_time}"
