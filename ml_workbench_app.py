import os
import sys

import pandas as pd
import streamlit as st


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.append(_SCRIPT_DIR)

try:
    from src import ml_workbench
except ImportError as e:
    st.error(f"Could not import workbench modules: {e}")
    st.stop()


RAW_DATA_PATH = "data/raw/first_file.xlsx"
SPLIT_DIR = "split"
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8501")


def _require_access() -> None:
    """Optional simple access gate for permission control via environment variable."""
    required_key = os.environ.get("ML_WORKBENCH_ACCESS_KEY", "").strip()
    if not required_key:
        return

    if st.session_state.get("ml_workbench_authorized"):
        return

    st.title("ML Development Workbench")
    st.warning("Restricted site. Enter access key.")
    entered = st.text_input("Access Key", type="password")
    if st.button("Unlock"):
        if entered == required_key:
            st.session_state["ml_workbench_authorized"] = True
            st.rerun()
        else:
            st.error("Invalid key")
    st.stop()


def ensure_ml_state() -> None:
    if "ml_selected_models" not in st.session_state:
        st.session_state["ml_selected_models"] = ml_workbench.default_model_ids()
    if "ml_selected_metrics" not in st.session_state:
        st.session_state["ml_selected_metrics"] = list(ml_workbench.DEFAULT_METRICS)
    if "ml_hyperparams" not in st.session_state:
        st.session_state["ml_hyperparams"] = ml_workbench.default_hyperparams()
    if "ml_threshold" not in st.session_state:
        st.session_state["ml_threshold"] = 0.5
    if "ml_split_mode" not in st.session_state:
        st.session_state["ml_split_mode"] = "random"
    if "ml_ranking_metric" not in st.session_state:
        st.session_state["ml_ranking_metric"] = "AUC_ROC"
    if "ml_seed" not in st.session_state:
        st.session_state["ml_seed"] = 42
    if "ml_test_size" not in st.session_state:
        st.session_state["ml_test_size"] = 0.2
    if "ml_val_size" not in st.session_state:
        st.session_state["ml_val_size"] = 0.25
    if "ml_active_job_id" not in st.session_state:
        st.session_state["ml_active_job_id"] = ""


def render_stage_timeline(job: dict) -> None:
    stage_order = [
        ("data_analysis", "Data Analysis", "🧭"),
        ("training", "Training", "🏋️"),
        ("validation", "Validation", "🧪"),
        ("evaluation", "Evaluation", "📊"),
        ("results_ready", "Results", "✅"),
    ]
    current = job.get("stage", "queued")
    stage_keys = [s[0] for s in stage_order]
    current_idx = stage_keys.index(current) if current in stage_keys else -1

    cols = st.columns(len(stage_order))
    for idx, (col, (stage_key, label, icon)) in enumerate(zip(cols, stage_order)):
        if job.get("status") == "failed":
            marker = "❌"
        elif stage_key == current:
            marker = "🔵"
        elif current_idx >= 0 and idx < current_idx:
            marker = "🟢"
        else:
            marker = "⚪"
        col.markdown(f"**{icon} {label}**")
        col.caption(marker)


def _render_model_selection(model_options):
    st.markdown("### Model Selection")
    st.caption("Enable or disable models for this run. Hover the help icon for each model summary.")

    selected_models = []
    model_cols = st.columns(2)

    for idx, model_meta in enumerate(model_options):
        model_id = model_meta["id"]
        widget_key = f"ml_model_enabled_{model_id}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = model_id in st.session_state["ml_selected_models"]

        with model_cols[idx % 2]:
            enabled = st.checkbox(
                f"{model_meta['icon']} {model_meta['label']}",
                value=bool(st.session_state[widget_key]),
                key=widget_key,
                help=model_meta.get("description", ""),
            )
            if enabled:
                selected_models.append(model_id)

    st.session_state["ml_selected_models"] = selected_models


def _render_metric_selection():
    st.markdown("### Metric Selection")
    st.caption("Select comparison metrics. Each metric has a short tooltip on how it is calculated.")

    selected_metrics = []
    metric_cols = st.columns(2)

    for idx, metric_name in enumerate(ml_workbench.SUPPORTED_METRICS):
        widget_key = f"ml_metric_enabled_{metric_name}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = metric_name in st.session_state["ml_selected_metrics"]

        with metric_cols[idx % 2]:
            enabled = st.checkbox(
                metric_name,
                value=bool(st.session_state[widget_key]),
                key=widget_key,
                help=ml_workbench.metric_tooltip(metric_name),
            )
            if enabled:
                selected_metrics.append(metric_name)

    st.session_state["ml_selected_metrics"] = selected_metrics


def _render_sidebar_hyperparams(model_options):
    st.sidebar.markdown("## Hyper-Parameters")
    st.sidebar.caption("Tune only selected models")

    for model_meta in model_options:
        model_id = model_meta["id"]
        if model_id not in st.session_state["ml_selected_models"]:
            continue

        with st.sidebar.expander(f"{model_meta['icon']} {model_meta['label']}", expanded=False):
            params = model_meta.get("params", {})
            if not params:
                st.caption("No tunable hyperparameters for this model.")
            for param_name, spec in params.items():
                current_value = st.session_state["ml_hyperparams"][model_id][param_name]
                widget_key = f"ml_hp_{model_id}_{param_name}"
                if spec["type"] == "int":
                    updated = st.number_input(
                        spec["label"],
                        min_value=int(spec["min"]),
                        max_value=int(spec["max"]),
                        value=int(current_value),
                        step=int(spec["step"]),
                        key=widget_key,
                    )
                    st.session_state["ml_hyperparams"][model_id][param_name] = int(updated)
                else:
                    updated = st.number_input(
                        spec["label"],
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(current_value),
                        step=float(spec["step"]),
                        format="%.4f",
                        key=widget_key,
                    )
                    st.session_state["ml_hyperparams"][model_id][param_name] = float(updated)


def render_ml_workbench() -> None:
    ensure_ml_state()

    st.title("ML Development Workbench")
    st.markdown("Configure, train, and compare turnover models without coding.")
    st.info("This is a separate ML-only site. Use your dashboard site for HR-facing analytics.")

    model_options = ml_workbench.list_model_options()

    top_controls1, top_controls2 = st.columns(2)
    with top_controls1:
        if st.button("Select All Models", use_container_width=True):
            st.session_state["ml_selected_models"] = ml_workbench.default_model_ids()
            for model_meta in model_options:
                st.session_state[f"ml_model_enabled_{model_meta['id']}"] = True
            st.rerun()
    with top_controls2:
        if st.button("Reset Defaults", use_container_width=True):
            st.session_state["ml_selected_models"] = ml_workbench.default_model_ids()
            st.session_state["ml_selected_metrics"] = list(ml_workbench.DEFAULT_METRICS)
            st.session_state["ml_hyperparams"] = ml_workbench.default_hyperparams()
            st.session_state["ml_threshold"] = 0.5
            st.session_state["ml_split_mode"] = "random"
            st.session_state["ml_ranking_metric"] = "AUC_ROC"
            st.session_state["ml_seed"] = 42
            st.session_state["ml_test_size"] = 0.2
            st.session_state["ml_val_size"] = 0.25
            st.rerun()

    st.markdown("---")
    _render_model_selection(model_options)
    st.markdown("---")
    _render_metric_selection()

    st.markdown("---")
    st.markdown("### Run Configuration")
    st.session_state["ml_ranking_metric"] = st.selectbox(
        "Primary Metric for Best Model",
        options=st.session_state["ml_selected_metrics"] or ml_workbench.SUPPORTED_METRICS,
        index=0,
        help="Model ranking uses this metric on the validation set.",
    )

    st.session_state["ml_threshold"] = st.slider(
        "Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state["ml_threshold"]),
        step=0.01,
        help="Employees with predicted probability >= threshold are classified as likely leavers.",
    )

    split_col1, split_col2, split_col3 = st.columns(3)
    with split_col1:
        st.session_state["ml_split_mode"] = st.selectbox(
            "Split Strategy",
            options=["random", "fixed"],
            format_func=lambda x: "Random 60/20/20" if x == "random" else "Fixed ID Split",
            index=0 if st.session_state["ml_split_mode"] == "random" else 1,
            help="Fixed uses split/train_ids.txt and split/test_ids.txt; random uses configurable train/val/test fractions.",
        )
    with split_col2:
        st.session_state["ml_test_size"] = st.slider(
            "Test Size",
            min_value=0.10,
            max_value=0.40,
            value=float(st.session_state["ml_test_size"]),
            step=0.05,
            disabled=st.session_state["ml_split_mode"] == "fixed",
        )
    with split_col3:
        st.session_state["ml_val_size"] = st.slider(
            "Validation Size",
            min_value=0.10,
            max_value=0.40,
            value=float(st.session_state["ml_val_size"]),
            step=0.05,
            disabled=st.session_state["ml_split_mode"] == "fixed",
        )

    st.session_state["ml_seed"] = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=999999,
        value=int(st.session_state["ml_seed"]),
        step=1,
    )

    _render_sidebar_hyperparams(model_options)

    st.markdown("---")
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        start_run = st.button("Start Training Run", use_container_width=True)
    with run_col2:
        refresh_status = st.button("Refresh Status", use_container_width=True)

    if start_run:
        if not st.session_state["ml_selected_models"]:
            st.error("Select at least one model before starting a run.")
        elif not st.session_state["ml_selected_metrics"]:
            st.error("Select at least one comparison metric before starting a run.")
        else:
            job_config = {
                "selected_models": st.session_state["ml_selected_models"],
                "selected_metrics": st.session_state["ml_selected_metrics"],
                "threshold": st.session_state["ml_threshold"],
                "split_mode": st.session_state["ml_split_mode"],
                "test_size": st.session_state["ml_test_size"],
                "val_size": st.session_state["ml_val_size"],
                "random_seed": st.session_state["ml_seed"],
                "ranking_metric": st.session_state["ml_ranking_metric"],
                "hyperparams": st.session_state["ml_hyperparams"],
            }
            new_job_id = ml_workbench.start_training_job(job_config, data_path=RAW_DATA_PATH, split_dir=SPLIT_DIR)
            st.session_state["ml_active_job_id"] = new_job_id
            st.success(f"Training job started: {new_job_id}")
            st.rerun()

    active_job_id = st.session_state.get("ml_active_job_id", "")
    if refresh_status and active_job_id:
        st.rerun()

    if active_job_id:
        job = ml_workbench.get_job(active_job_id)
        if job:
            if job.get("status") == "running":
                st.caption("Live refresh enabled while training is running.")
                st.markdown(
                    "<meta http-equiv='refresh' content='5'>",
                    unsafe_allow_html=True,
                )

            st.markdown("### Processing Timeline")
            render_stage_timeline(job)
            st.progress(min(max(int(job.get("progress", 0)), 0), 100))

            job_status = job.get("status", "queued")
            stage_label = job.get("stage_label", "Queued")
            active_model = job.get("active_model")
            active_model_label = ml_workbench.MODEL_REGISTRY.get(active_model, {}).get("label", "") if active_model else ""

            if job_status == "running":
                st.info(f"Stage: {stage_label}" + (f" | Active model: {active_model_label}" if active_model_label else ""))
            elif job_status == "queued":
                st.warning("Run is queued and will start automatically.")
            elif job_status == "failed":
                st.error(f"Training failed: {job.get('error', 'unknown error')}")
            elif job_status == "completed":
                st.success(f"Training complete. Best model: {job.get('best_model', 'N/A')}")

            st.markdown("#### Selected Models Status")
            chips = []
            selected_model_ids = job.get("config", {}).get("selected_models", [])
            for model_id in selected_model_ids:
                meta = ml_workbench.MODEL_REGISTRY.get(model_id)
                if not meta:
                    continue
                if job_status == "completed":
                    marker = "✅"
                elif job_status == "running" and model_id == active_model:
                    marker = "🔄"
                else:
                    marker = "⏳"
                chips.append(f"{marker} {meta['icon']} {meta['label']}")
            if chips:
                st.caption(" | ".join(chips))

            if job.get("logs"):
                with st.expander("Run Logs", expanded=False):
                    for line in job["logs"][-15:]:
                        st.text(line)

            if job_status == "completed" and job.get("results"):
                st.markdown("### Model Comparison Results")
                result_df = pd.DataFrame(job["results"])
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.success(f"Summary saved: {job.get('summary_path', 'N/A')}")

    st.markdown("---")
    st.markdown("### Recent Jobs")
    recent_jobs = ml_workbench.list_jobs(limit=8)
    if recent_jobs:
        jobs_df = pd.DataFrame(
            [
                {
                    "Job ID": j.get("job_id"),
                    "Status": j.get("status"),
                    "Stage": j.get("stage_label"),
                    "Progress": f"{int(j.get('progress', 0))}%",
                    "Created": j.get("created_at"),
                    "Best Model": j.get("best_model", ""),
                }
                for j in recent_jobs
            ]
        )
        st.dataframe(jobs_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No training jobs yet.")

    st.markdown("---")
    st.markdown("### Open HR Dashboards")
    st.markdown(f"Use this link after model comparison: [Go to HR Dashboard]({DASHBOARD_URL})")


if __name__ == "__main__":
    st.set_page_config(page_title="ML Development Workbench", page_icon="🧪", layout="wide")
    _require_access()
    render_ml_workbench()
