from django.contrib import admin
from .models import Dataset, AttackType, AttackPattern, IntrusionLog

admin.site.register(Dataset)
admin.site.register(AttackType)
admin.site.register(AttackPattern)
admin.site.register(IntrusionLog)