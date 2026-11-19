from django.urls import path
from core.views import ids_pipeline
from core.views import ids_dashboard
from core.views import delete_all_runs
from core.views import realtime_capture_page, realtime_capture_start, realtime_capture_status
from django.contrib.auth.views import LoginView, LogoutView


urlpatterns = [
    path("", ids_pipeline, name="ids_pipeline"),
    path('dashboard/', ids_dashboard, name='ids_dashboard'),
    path("delete-runs/", delete_all_runs, name="delete_all_runs"),

    # Real-time capture
    path("capture/", realtime_capture_page, name="realtime_capture"),
    path("capture/start/", realtime_capture_start, name="realtime_capture_start"),
    path("capture/status/<str:session_key>/", realtime_capture_status, name="realtime_capture_status"),
]
