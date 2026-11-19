from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import os

from ml_engine.dataset_manager import get_active_dataset_paths, load_datasets
from ml_engine.train_eval import train_and_evaluate
from ml_engine.metrics import compute_overall_metrics
from ml_engine.pdf_report import generate_pdf_report
from ml_engine.preprocess import Preprocessor
from ml_engine.intrusion_logs import (
    get_attack_vs_normal_count,
    get_attack_distribution,
    has_logs
)

from django.contrib.auth.views import LoginView

class IDSLoginView(LoginView):
    template_name = "core/login.html"

# ---------------------------
# Dashboard
# ---------------------------
@login_required
@permission_required("core.can_view_dashboard", raise_exception=True)
def ids_dashboard(request):

    if not has_logs():
        messages.error(request, "Please train or evaluate a dataset before viewing the dashboard.")
        return redirect('ids_pipeline')

    test_run_id = request.GET.get('test_run')

    attack_count, normal_count = get_attack_vs_normal_count(test_run_id)
    attack_distribution        = get_attack_distribution(test_run_id)

    from ml_engine.intrusion_logs import get_latest_test_runs
    test_runs = get_latest_test_runs()

    return render(request, "core/dashboard.html", {
        "attack_count":        attack_count or 0,
        "normal_count":        normal_count or 0,
        "attack_distribution": attack_distribution or [],
        "test_runs":           test_runs
    })


# ---------------------------
# Delete all run history (admin only)
# ---------------------------
@login_required
@require_POST
def delete_all_runs(request):
    """Delete every TestRun and its related IntrusionLogs. Admin only."""
    if not request.user.is_staff:
        messages.error(request, "Only administrators can delete run history.")
        return redirect('ids_dashboard')

    from core.models import TestRun, IntrusionLog, Dataset

    # 1. All logs (including any orphans where test_run is NULL)
    IntrusionLog.objects.all().delete()
    # 2. All test-run records
    TestRun.objects.all().delete()
    # 3. Dataset rows that have no remaining logs or runs (orphan cleanup)
    Dataset.objects.filter(
        intrusionlog__isnull=True,
        test_runs__isnull=True
    ).delete()

    messages.success(request, "All run history has been deleted.")
    return redirect('ids_pipeline')      # dashboard would redirect anyway with zero logs


# ---------------------------
# Pipeline Controller
# ---------------------------
@login_required
def ids_pipeline(request):

    if request.method == "GET":
        return render(request, "core/ids_pipeline.html")

    action = request.POST.get("action")

    # ---------------------------
    # VIEW DASHBOARD
    # ---------------------------
    if action == "view":
        return redirect("ids_dashboard")

    # ---------------------------
    # TRAIN / TEST  ->  FILE REQUIRED
    # ---------------------------
    files = request.FILES.getlist("datasets")
    if not files:
        return render(request, "core/ids_pipeline.html", {
            "error": "Dataset upload is required for training or evaluation."
        })

    dataset_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    uploaded_paths = []
    for f in files:
        path = os.path.join(dataset_dir, f.name)
        with open(path, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)
        uploaded_paths.append(path)

    dataset_paths = get_active_dataset_paths(uploaded_paths)
    datasets      = load_datasets(dataset_paths)

    # --- auto-detect num_classes using the same logic as Preprocessor
    num_classes = 4          # safe fallback
    if datasets:
        first_df   = datasets[0]["dataframe"]
        label_col, _ = Preprocessor.detect_label_column(first_df)
        if label_col is not None:
            num_classes = first_df[label_col].nunique()

    # ---------------------------
    # TRAIN & EVALUATE
    # ---------------------------
    if action == "train" and request.user.has_perm("core.can_train_model"):
        try:
            dataset_results = train_and_evaluate(
                datasets,
                num_classes=num_classes,
                use_pretrained=False,
                user=request.user
            )
        except Exception as e:
            messages.error(request, f"Training failed: {str(e)}")
            return redirect('ids_pipeline')

    # ---------------------------
    # EVALUATE ONLY
    # ---------------------------
    elif action == "test" and (request.user.has_perm("core.can_test_model") or
         request.user.has_perm("core.can_train_model")):
        try:
            dataset_results = train_and_evaluate(
                datasets,
                num_classes=num_classes,
                use_pretrained=True,
                user=request.user
            )
        except Exception as e:
            messages.error(request, f"Evaluation failed: {str(e)}")
            return redirect('ids_pipeline')

    else:
        return render(request, "core/unauthorized.html")

    # ---------------------------
    # CLASS MISMATCH WARNINGS
    # ---------------------------
    # Attach a warning to any result where dataset class count ≠ model class count
    if action == "test":
        from ml_engine.dataset_compatibility import (
            get_dataset_num_classes, load_label_mapping, find_compatible_model
        )
        from django.conf import settings as _settings
        import os as _os
        MODEL_DIR = _os.path.join(_settings.MEDIA_ROOT, "saved_models")

        for res, ds in zip(dataset_results, datasets):
            df        = ds["dataframe"]
            dataset_n = get_dataset_num_classes(df)

            # Get the TRUE model class count from .meta file — NOT from
            # classification_report keys, which only shows classes that
            # appeared in predictions and gives wrong counts (e.g. 13 vs 14).
            model_n = None
            compatible = find_compatible_model(df, MODEL_DIR)
            if compatible:
                mapping = load_label_mapping(compatible, MODEL_DIR)
                if mapping:
                    model_n = len(mapping)

            if dataset_n and model_n and dataset_n != model_n:
                if model_n > dataset_n:
                    res["class_mismatch_warning"] = (
                        f"ℹ️ Using a model trained on {model_n} classes to evaluate "
                        f"this {dataset_n}-class dataset. This is normal — the model applies "
                        f"its broader knowledge to your data. For maximum accuracy, run "
                        f"Train & Evaluate on this specific dataset to build a dedicated model."
                    )
                else:
                    res["class_mismatch_warning"] = (
                        f"⚠️ This dataset has {dataset_n} classes but the loaded model was "
                        f"trained on only {model_n} classes. Some attack types cannot be "
                        f"detected. Run Train & Evaluate on this dataset first."
                    )

    # ---------------------------
    # RESULTS + PDF REPORT
    # ---------------------------
    overall_metrics = compute_overall_metrics(dataset_results)

    pdf_path = os.path.join(settings.MEDIA_ROOT, "results", "ids_report.pdf")
    generate_pdf_report(
        output_path=pdf_path,
        dataset_results=dataset_results,
        overall_metrics=overall_metrics
    )

    return render(request, "core/results.html", {
        "dataset_results": dataset_results,
        "overall_metrics": overall_metrics,
        "pdf_path":        "/media/results/ids_report.pdf"
    })

# ---------------------------
# Real-Time Capture Views
# ---------------------------
import threading
from ml_engine.realtime_capture import capture, list_interfaces

# Simple in-memory status store (per-session is fine for single-user dev)
_capture_status = {}

@login_required
def realtime_capture_page(request):
    """Render the real-time capture page with available interfaces."""
    interfaces = []
    scapy_available = True
    try:
        interfaces = list_interfaces()
    except ImportError:
        scapy_available = False

    return render(request, "core/realtime_capture.html", {
        "interfaces":       interfaces,
        "scapy_available":  scapy_available,
    })


@login_required
@require_POST
def realtime_capture_start(request):
    """Start a background capture and return JSON status."""
    dataset_type  = request.POST.get("dataset_type", "cic")   # "cic" or "5g"
    iface         = request.POST.get("iface", "")
    packet_count  = int(request.POST.get("packet_count", 500))
    label         = request.POST.get("label", "BENIGN")
    attack_type   = request.POST.get("attack_type", "") or label
    attack_tool   = request.POST.get("attack_tool", "Live-Capture")

    if not iface:
        return JsonResponse({"status": "error", "message": "No interface selected."})

    # Build output path inside MEDIA_ROOT/datasets/
    import time as _time
    ts_str   = str(int(_time.time()))
    filename = f"realtime_{dataset_type.upper()}_{ts_str}.csv"
    out_path = os.path.join(settings.MEDIA_ROOT, "datasets", filename)

    session_key = f"{request.user.id}_{ts_str}"
    _capture_status[session_key] = {"status": "running", "message": "Capturing...", "file": ""}

    def _run():
        ok, msg, count = capture(
            dataset_type  = dataset_type,
            iface         = iface,
            packet_count  = packet_count,
            label         = label,
            output_path   = out_path,
            attack_type   = attack_type,
            attack_tool   = attack_tool,
        )
        _capture_status[session_key] = {
            "status":  "done" if ok else "error",
            "message": msg,
            "file":    filename if ok else "",
            "count":   count,
        }

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return JsonResponse({
        "status":      "started",
        "session_key": session_key,
        "message":     f"Capturing {packet_count} packets on {iface}...",
    })


@login_required
def realtime_capture_status(request, session_key):
    """Poll endpoint — returns current capture status as JSON."""
    info = _capture_status.get(session_key, {"status": "unknown", "message": "Session not found."})
    return JsonResponse(info)