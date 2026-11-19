from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from core.models import IntrusionLog


class Command(BaseCommand):
    help = "Create IDS roles and assign permissions (idempotent)"

    def handle(self, *args, **options):
        self.stdout.write("Setting up IDS roles and permissions...")

        content_type = ContentType.objects.get_for_model(IntrusionLog)

        # Permissions
        perms = {
            "can_train_model": Permission.objects.get(
                codename="can_train_model",
                content_type=content_type
            ),
            "can_test_model": Permission.objects.get(
                codename="can_test_model",
                content_type=content_type
            ),
            "can_view_dashboard": Permission.objects.get(
                codename="can_view_dashboard",
                content_type=content_type
            ),
        }

        # Role → permissions mapping
        # Admin: Can train and test models
        # User: Can test models (formerly "Analyst")
        # Guest: Can only view dashboard (formerly "Viewer")
        role_map = {
            "Admin": [
                perms["can_train_model"],
                perms["can_test_model"],
                perms["can_view_dashboard"],
            ],
            "User": [
                perms["can_test_model"],
                perms["can_view_dashboard"],
            ],
            "Guest": [
                perms["can_view_dashboard"],
            ],
        }

        for role_name, permissions in role_map.items():
            group, created = Group.objects.get_or_create(name=role_name)

            if created:
                self.stdout.write(f"✔ Created group: {role_name}")
            else:
                self.stdout.write(f"ℹ Group already exists: {role_name}")

            for perm in permissions:
                group.permissions.add(perm)

        self.stdout.write(self.style.SUCCESS("✅ Role setup completed successfully"))