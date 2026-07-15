import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.append(_SCRIPT_DIR)

try:
    from src import ml_workbench
except ImportError as e:
    st.error(f"Could not import workbench modules: {e}")
    st.stop()


FINAL_WORKFLOW_LABEL = "file1 + file2 validation -> file3 external test"


def _require_access() -> None:
    """Optional simple access gate for permission control via environment variable."""
    required_key = os.environ.get("ML_WORKBENCH_ACCESS_KEY", "").strip()
    if not required_key:
        return

    if st.session_state.get("ml_workbench_authorized"):
        return

    st.title("Final ML Workbench")
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
        st.session_state["ml_split_mode"] = "final"
    if "ml_ranking_metric" not in st.session_state:
        st.session_state["ml_ranking_metric"] = "PR_AUC"
    if "ml_seed" not in st.session_state:
        st.session_state["ml_seed"] = 42
    if "ml_test_size" not in st.session_state:
        st.session_state["ml_test_size"] = 0.0
    if "ml_val_size" not in st.session_state:
        st.session_state["ml_val_size"] = 0.20
    if "ml_active_job_id" not in st.session_state:
        st.session_state["ml_active_job_id"] = ""
    if "ml_drop_feature_groups" not in st.session_state:
        st.session_state["ml_drop_feature_groups"] = []
    if "ml_train_sources" not in st.session_state:
        st.session_state["ml_train_sources"] = ["file1", "file2"]
    if "ml_test_sources" not in st.session_state:
        st.session_state["ml_test_sources"] = ["file3"]


FEATURE_GROUPS = ml_workbench.FEATURE_GROUPS

def _node_state(stage_key: str, current_stage: str, job_status: str, model_id: str = None, active_model: str = None, completed_models: list = None) -> str:
    """Return 'pending', 'active', 'completed', or 'failed' for a pipeline node."""
    stage_order = ["data_analysis", "training", "evaluation", "results_ready"]
    # Normalize: "validation" maps to "evaluation" in the pipeline
    mapped_stage = "evaluation" if current_stage == "validation" else current_stage
    current_idx = stage_order.index(mapped_stage) if mapped_stage in stage_order else -1

    if stage_key == "model":
        if completed_models and model_id in completed_models:
            return "completed"
        if model_id == active_model and mapped_stage == "training":
            return "active"
        if current_idx > stage_order.index("training"):
            return "completed"
        return "pending"

    stage_idx = stage_order.index(stage_key) if stage_key in stage_order else -1

    if job_status == "failed":
        if stage_idx < current_idx:
            return "completed"
        if stage_key == mapped_stage:
            return "failed"
        return "pending"

    if stage_idx < current_idx:
        return "completed"
    if stage_idx == current_idx:
        return "active"
    return "pending"


def render_pipeline_visualization(job: dict) -> None:
    """Render a tree-pipeline diagram using HTML/CSS embedded in Streamlit."""
    stage = job.get("stage", "queued")
    status = job.get("status", "queued")
    active_model = job.get("active_model")
    completed_models = job.get("completed_models", [])
    selected_models = job.get("config", {}).get("selected_models", [])
    split_info = job.get("split_info", {})

    def _cls(stage_key, model_id=None):
        return _node_state(stage_key, stage, status, model_id, active_model, completed_models)

    def _dot(state):
        return {"pending": "&#9898;", "active": "&#128309;", "completed": "&#9989;", "failed": "&#128308;"}.get(state, "&#9898;")

    # Build model node HTML
    model_chips = []
    for mid in selected_models:
        meta = ml_workbench.MODEL_REGISTRY.get(mid, {})
        state = _cls("model", mid)
        label = f"{meta.get('icon', '')} {meta.get('label', mid)}"
        model_chips.append(f'<div class="p-node model-chip {state}">{_dot(state)} {label}</div>')
    models_html = "\n".join(model_chips)

    split_label = ""
    if split_info:
        split_label = f"Train {split_info.get('train', '?')} &middot; Val {split_info.get('val', '?')} &middot; Test {split_info.get('test', '?')}"

    data_state = _cls("data_analysis")
    train_state = _cls("training")
    eval_state = _cls("evaluation")
    result_state = _cls("results_ready")

    html = f"""
    <style>
        .pipeline-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0;
            padding: 18px 0 10px 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .p-node {{
            padding: 10px 22px;
            border-radius: 10px;
            border: 2px solid #D1D5DB;
            background: #F9FAFB;
            color: #6B7280;
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            min-width: 170px;
            transition: all 0.3s;
        }}
        .p-node.active {{
            border-color: #3B82F6;
            background: #EFF6FF;
            color: #1D4ED8;
            box-shadow: 0 0 12px rgba(59,130,246,0.25);
            animation: ppulse 2s ease-in-out infinite;
        }}
        .p-node.completed {{
            border-color: #10B981;
            background: #ECFDF5;
            color: #047857;
        }}
        .p-node.failed {{
            border-color: #EF4444;
            background: #FEF2F2;
            color: #B91C1C;
        }}
        .p-connector {{
            width: 3px;
            height: 28px;
            background: #D1D5DB;
            border-radius: 2px;
        }}
        .p-connector.active {{
            background: linear-gradient(180deg, #3B82F6 0%, #93C5FD 100%);
            animation: flowAnim 1.2s ease-in-out infinite;
        }}
        .p-connector.completed {{
            background: #10B981;
        }}
        .model-grid {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            max-width: 700px;
        }}
        .model-chip {{
            padding: 7px 14px;
            font-size: 13px;
            min-width: 120px;
        }}
        .branch-lines {{
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 0;
            height: 22px;
        }}
        .branch-lines .b-line {{
            width: 2px;
            height: 22px;
            background: #D1D5DB;
        }}
        .branch-lines .b-line.active {{
            background: #3B82F6;
        }}
        .branch-lines .b-line.completed {{
            background: #10B981;
        }}
        .h-bar {{
            height: 2px;
            background: #D1D5DB;
            border-radius: 1px;
            min-width: 60px;
        }}
        .h-bar.active {{ background: #3B82F6; }}
        .h-bar.completed {{ background: #10B981; }}
        .split-caption {{
            font-size: 12px;
            color: #9CA3AF;
            margin-top: 2px;
        }}
        @keyframes ppulse {{
            0%, 100% {{ box-shadow: 0 0 6px rgba(59,130,246,0.15); }}
            50% {{ box-shadow: 0 0 18px rgba(59,130,246,0.45); }}
        }}
        @keyframes flowAnim {{
            0%   {{ opacity: 0.4; }}
            50%  {{ opacity: 1; }}
            100% {{ opacity: 0.4; }}
        }}
    </style>

    <div class="pipeline-wrap">
        <!-- Data Preprocessing -->
        <div class="p-node {data_state}">{_dot(data_state)} &#128229; Data Loading &amp; Preprocessing</div>
        <div class="p-connector {data_state}"></div>

        <!-- Split info -->
        <div class="p-node {data_state}" style="min-width:200px;">
            {_dot(data_state)} &#9986; Train / Val / Test Split
            {"<div class='split-caption'>" + split_label + "</div>" if split_label else ""}
        </div>
        <div class="p-connector {train_state}"></div>

        <!-- Branch out to models -->
        <div style="display:flex;align-items:flex-end;justify-content:center;">
            <div class="h-bar {train_state}" style="width:{max(len(selected_models)*60, 120)}px;"></div>
        </div>
        <div class="branch-lines" style="width:{max(len(selected_models)*60, 120)}px;justify-content:space-around;">
            {"".join(f'<div class="b-line {_cls("model", m)}"></div>' for m in selected_models)}
        </div>

        <!-- Model nodes -->
        <div class="model-grid">
            {models_html}
        </div>

        <!-- Converge -->
        <div class="branch-lines" style="width:{max(len(selected_models)*60, 120)}px;justify-content:space-around;">
            {"".join(f'<div class="b-line {_cls("model", m)}"></div>' for m in selected_models)}
        </div>
        <div style="display:flex;align-items:flex-start;justify-content:center;">
            <div class="h-bar {eval_state}" style="width:{max(len(selected_models)*60, 120)}px;"></div>
        </div>
        <div class="p-connector {eval_state}"></div>

        <!-- Metrics Evaluation -->
        <div class="p-node {eval_state}">{_dot(eval_state)} &#128202; Metrics Evaluation</div>
        <div class="p-connector {result_state}"></div>

        <!-- Results -->
        <div class="p-node {result_state}">{_dot(result_state)} &#127942; Results &amp; Ranking</div>
    </div>
    """
    n_model_rows = max(1, -(-len(selected_models) // 3))
    pipeline_height = 460 + n_model_rows * 55
    components.html(html, height=pipeline_height, scrolling=False)


def _render_model_selection(model_options):
    st.markdown("### Model Selection")
    st.caption("Models that cannot fit the selected training sources are disabled automatically.")

    selected_models = []
    model_cols = st.columns(2)

    for idx, model_meta in enumerate(model_options):
        model_id = model_meta["id"]
        widget_key = f"ml_model_enabled_{model_id}"
        disabled = bool(model_meta.get("disabled", False))
        if widget_key not in st.session_state:
            st.session_state[widget_key] = model_id in st.session_state["ml_selected_models"] and not disabled
        if disabled:
            st.session_state[widget_key] = False

        with model_cols[idx % 2]:
            enabled = st.checkbox(
                f"{model_meta['icon']} {model_meta['label']}",
                key=widget_key,
                help=model_meta.get("description", ""),
                disabled=disabled,
            )
            if enabled and not disabled:
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
                st.session_state["ml_hyperparams"].setdefault(model_id, {})
                st.session_state["ml_hyperparams"][model_id].setdefault(param_name, spec["default"])
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



def _render_source_selection() -> None:
    st.markdown("### Data Sources")
    st.caption("Choose which files train the models and which files evaluate them. Validation is always split from the selected training sources.")

    source_options = ml_workbench.SOURCE_OPTIONS
    col_train, col_test = st.columns(2)
    with col_train:
        train_sources = st.multiselect(
            "Training Sources",
            options=source_options,
            default=[source for source in st.session_state["ml_train_sources"] if source in source_options],
            help="Selected sources are cleaned together and then split into inner-train and validation employees.",
        )
    with col_test:
        test_sources = st.multiselect(
            "Test Sources",
            options=source_options,
            default=[source for source in st.session_state["ml_test_sources"] if source in source_options],
            help="Selected sources are evaluated after validation selects the best model.",
        )

    if not train_sources:
        train_sources = ["file1"]
        st.warning("Select at least one training source. Using file1 for now.")
    if not test_sources:
        test_sources = train_sources.copy()
        st.warning("Select at least one test source. Mirroring training sources for now.")

    st.session_state["ml_train_sources"] = train_sources
    st.session_state["ml_test_sources"] = test_sources

    allowed = set(ml_workbench.allowed_model_ids(train_sources))
    st.session_state["ml_selected_models"] = [
        model_id for model_id in st.session_state["ml_selected_models"] if model_id in allowed
    ]
def _render_feature_selection():
    st.markdown("### Feature Selection")
    st.caption(
        "Exclude feature groups from training to test model sensitivity. "
        "Excluded features are dropped only during training — they remain in the final predictions output."
    )

    dropped = []
    fg_cols = st.columns(3)
    for idx, (group_name, spec) in enumerate(FEATURE_GROUPS.items()):
        widget_key = f"ml_fg_drop_{group_name}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = group_name in st.session_state["ml_drop_feature_groups"]
        with fg_cols[idx % 3]:
            excluded = st.checkbox(
                f"Drop: {group_name}",
                key=widget_key,
                help=spec["description"],
            )
            if excluded:
                dropped.append(group_name)

    st.session_state["ml_drop_feature_groups"] = dropped
    if dropped:
        st.warning(f"Dropping feature groups for training: {', '.join(dropped)}")


def _render_overfit_indicators(result_df: pd.DataFrame, selected_metrics: list):
    """Show train-vs-val-vs-test gap analysis for overfitting detection."""
    # Key metrics to compare (skip informational ones)
    key_metrics = [m for m in ["AUC", "F1", "Recall", "Precision", "Recall@Top20%"]
                   if m in selected_metrics]
    if not key_metrics:
        return

    has_train = any(f"Train_{m}" in result_df.columns for m in key_metrics)
    has_val = any(f"Val_{m}" in result_df.columns for m in key_metrics)
    if not has_train:
        return

    st.markdown("#### Overfitting Analysis")

    rows = []
    for _, row in result_df.iterrows():
        model_name = row.get("Model", "Unknown")
        for m in key_metrics:
            train_v = row.get(f"Train_{m}")
            val_v = row.get(f"Val_{m}")
            test_v = row.get(m)
            if train_v is None or test_v is None:
                continue
            train_v, test_v = float(train_v), float(test_v)
            gap_train_test = train_v - test_v
            gap_train_val = (train_v - float(val_v)) if val_v is not None else None

            # Traffic-light thresholds
            gap = gap_train_test
            if gap > 0.15:
                status, icon = "Severe Overfit", "🔴"
            elif gap > 0.07:
                status, icon = "Moderate Overfit", "🟡"
            elif gap > 0.03:
                status, icon = "Mild Overfit", "🟠"
            else:
                status, icon = "Good Fit", "🟢"

            entry = {
                "Model": model_name,
                "Metric": m,
                "Train": round(train_v, 4),
                "Test": round(test_v, 4),
                "Train−Test Gap": round(gap_train_test, 4),
                "Status": f"{icon} {status}",
            }
            if has_val and val_v is not None:
                entry["Val"] = round(float(val_v), 4)
                entry["Train−Val Gap"] = round(gap_train_val, 4) if gap_train_val is not None else None
            rows.append(entry)

    if not rows:
        return

    gap_df = pd.DataFrame(rows)

    # Color the gap columns
    def _color_gap(val):
        if not isinstance(val, (int, float)):
            return ""
        if val > 0.15:
            return "background-color: #fee2e2; color: #991b1b"
        elif val > 0.07:
            return "background-color: #fef3c7; color: #92400e"
        elif val > 0.03:
            return "background-color: #ffedd5; color: #9a3412"
        return "background-color: #d1fae5; color: #065f46"

    gap_style_cols = ["Train−Test Gap"]
    if "Train−Val Gap" in gap_df.columns:
        gap_style_cols.append("Train−Val Gap")

    styled_gap = gap_df.style.map(_color_gap, subset=gap_style_cols).format(
        {c: "{:.4f}" for c in gap_df.columns if c not in ("Model", "Metric", "Status")}
    )
    st.dataframe(styled_gap, width="stretch", hide_index=True)

    # Summary verdict
    worst_gap = max(r["Train−Test Gap"] for r in rows)
    if worst_gap > 0.15:
        st.error("⚠️ Severe overfitting detected. Consider adding regularization, reducing model complexity, or increasing training data.")
    elif worst_gap > 0.07:
        st.warning("⚠️ Moderate overfitting detected. The model may not generalize well to unseen data.")
    elif worst_gap > 0.03:
        st.info("ℹ️ Mild overfitting present. Performance is reasonable but could improve with regularization.")
    else:
        st.success("✅ Models show good generalization — no significant overfitting detected.")


def _render_results_section(job: dict) -> None:
    """Render a rich results section after training completes."""
    results = job.get("results", [])
    if not results:
        return

    config = job.get("config", {})
    selected_metrics = config.get("selected_metrics", list(ml_workbench.DEFAULT_METRICS))
    ranking_metric = config.get("ranking_metric", "PR_AUC")
    best_model_name = job.get("best_model", "")

    result_df = pd.DataFrame(results)

    st.markdown("---")
    st.markdown("### Model Comparison Results")

    # --- Summary cards ---
    split_info = job.get("split_info", {})
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Models Trained", len(results))
    with info_cols[1]:
        st.metric("Best Model", best_model_name)
    with info_cols[2]:
        best_row = next((r for r in results if r.get("Model") == best_model_name), None)
        best_metric_value = best_row.get(f"Val_{ranking_metric}", best_row.get(ranking_metric, 0)) if best_row else None
        best_val = f"{float(best_metric_value):.4f}" if best_metric_value is not None else "N/A"
        st.metric(f"Best Val {ranking_metric}", best_val)
    with info_cols[3]:
        st.metric("Total Samples", split_info.get("total", "N/A"))

    dropped_cols = job.get("dropped_columns", [])
    drop_groups = config.get("drop_feature_groups", [])
    if drop_groups:
        st.info(f"🔧 Feature groups excluded from training: **{', '.join(drop_groups)}** ({len(dropped_cols)} columns dropped)")

    # --- Test set metrics table ---
    st.markdown("#### Test Set Metrics")
    test_cols = ["Model"] + [m for m in selected_metrics if m in result_df.columns]
    test_display = result_df[test_cols].copy()

    # Highlight best value per metric column
    def _highlight_best(col):
        if col.name == "Model":
            return [""] * len(col)
        best_idx = col.astype(float).idxmax()
        return ["background-color: #d1fae5; font-weight: 700" if i == best_idx else "" for i in col.index]

    styled = test_display.style.apply(_highlight_best, axis=0).format(
        {m: "{:.4f}" for m in test_cols if m != "Model"}
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    # --- Validation set metrics table ---
    val_cols_available = [f"Val_{m}" for m in selected_metrics if f"Val_{m}" in result_df.columns]
    if val_cols_available:
        st.markdown("#### Validation Set Metrics")
        val_display = result_df[["Model"] + val_cols_available].copy()
        val_display.columns = ["Model"] + [c.replace("Val_", "") for c in val_cols_available]

        styled_val = val_display.style.apply(_highlight_best, axis=0).format(
            {c.replace("Val_", ""): "{:.4f}" for c in val_cols_available}
        )
        st.dataframe(styled_val, width="stretch", hide_index=True)

    # --- Overfitting Indicators ---
    _render_overfit_indicators(result_df, selected_metrics)

    # --- Grouped bar chart comparison ---
    st.markdown("#### Visual Comparison")
    chart_metrics = [m for m in selected_metrics if m in result_df.columns and m not in ("Primary_Probability", "Error_Cost")]

    if chart_metrics:
        fig = go.Figure()
        colors = [
            "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
            "#8B5CF6", "#EC4899", "#06B6D4",
        ]
        for idx, model_row in result_df.iterrows():
            fig.add_trace(go.Bar(
                name=model_row["Model"],
                x=chart_metrics,
                y=[float(model_row.get(m, 0)) for m in chart_metrics],
                marker_color=colors[idx % len(colors)],
            ))

        fig.update_layout(
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Score",
            legend_title="Model",
            height=420,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # --- Per-model detail expanders ---
    st.markdown("#### Per-Model Details")
    for _, row in result_df.iterrows():
        model_name = row.get("Model", "Unknown")
        meta = next(
            (v for v in ml_workbench.MODEL_REGISTRY.values() if v["label"] == model_name),
            {},
        )
        icon = meta.get("icon", "")
        with st.expander(f"{icon} {model_name}", expanded=False):
            n_metric_cols = min(len(selected_metrics), 4)
            detail_cols = st.columns(n_metric_cols)
            for ci, m in enumerate(selected_metrics):
                val = row.get(m)
                val_val = row.get(f"Val_{m}")
                with detail_cols[ci % n_metric_cols]:
                    if val is not None:
                        delta = None
                        if val_val is not None:
                            delta = f"{float(val) - float(val_val):+.4f} vs val"
                        st.metric(m, f"{float(val):.4f}", delta=delta)

    # --- Feature Importance ---
    importance = job.get("feature_importance", [])
    if importance:
        st.markdown("---")
        st.markdown(f"### Feature Importance -- {best_model_name}")
        imp_method = job.get("feature_importance_method", "")
        if imp_method:
            st.caption(f"Method: {imp_method}")

        imp_df = pd.DataFrame(importance)
        imp_df.index = imp_df.index + 1
        imp_df.index.name = "Rank"

        # Bar chart
        fig_imp = go.Figure(go.Bar(
            x=[round(s, 4) for s in imp_df["score"]],
            y=imp_df["description"],
            orientation="h",
            marker_color="#3B82F6",
        ))
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Importance Score",
            height=max(300, len(imp_df) * 28),
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_imp, width="stretch")

        with st.expander("Feature Importance Table", expanded=False):
            st.dataframe(
                imp_df[["description", "feature", "score"]].rename(
                    columns={"description": "Feature", "feature": "Raw Name", "score": "Score"}
                ),
                width="stretch",
            )

    # --- Predictive Uncertainty ---
    uncertainty = job.get("uncertainty")
    if uncertainty:
        st.markdown("---")
        st.markdown("### Predictive Uncertainty (Bootstrap Posterior)")
        st.caption(
            "Each bootstrap refit is an approximate draw from the posterior over models. "
            "The spread of predicted probabilities across refits approximates the "
            "posterior predictive distribution at each test point."
        )
        uc = st.columns(4)
        with uc[0]:
            st.metric("Bootstrap Refits", uncertainty["n_bootstrap"])
        with uc[1]:
            st.metric("Mean Posterior Std", f"{uncertainty['mean_posterior_std']:.4f}")
        with uc[2]:
            st.metric("Mean 95% CI Width", f"{uncertainty['mean_ci_width']:.4f}")
        with uc[3]:
            st.metric("Decisive Predictions", f"{uncertainty['decisive_pct']:.1f}%",
                       help="Percentage of predictions where the 95% credible interval stays on one side of 0.5")

    # --- Risk Distribution ---
    risk_dist = job.get("risk_distribution", {})
    if risk_dist:
        st.markdown("---")
        st.markdown("### Risk Distribution")
        risk_colors = {
            "Low Risk": "#10B981",
            "Medium Risk": "#F59E0B",
            "High Risk": "#EF4444",
            "Very High Risk": "#991B1B",
        }
        risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
        ordered_risks = [(k, risk_dist.get(k, 0)) for k in risk_order if k in risk_dist]

        fig_risk = go.Figure(go.Bar(
            x=[r[0] for r in ordered_risks],
            y=[r[1] for r in ordered_risks],
            marker_color=[risk_colors.get(r[0], "#6B7280") for r in ordered_risks],
            text=[r[1] for r in ordered_risks],
            textposition="auto",
        ))
        fig_risk.update_layout(
            xaxis_title="Risk Category",
            yaxis_title="Employees",
            height=300,
            margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig_risk, width="stretch")

    # --- Output Files ---
    st.markdown("---")
    st.markdown("### Generated Output Files")
    file_info = []
    if job.get("predictions_path"):
        file_info.append(("Predictions Excel", job["predictions_path"]))
    if job.get("artifact_path"):
        file_info.append(("Model Pipeline (pickle)", job["artifact_path"]))
    if job.get("results_text_path"):
        file_info.append(("Results Report (text)", job["results_text_path"]))
    if job.get("summary_path"):
        file_info.append(("Summary (JSON)", job["summary_path"]))

    for label, path in file_info:
        if os.path.exists(path):
            with open(path, "rb") as fp:
                st.download_button(
                    f"Download {label}",
                    data=fp,
                    file_name=os.path.basename(path),
                    width="stretch",
                )
        else:
            st.caption(f"{label}: {path}")


def render_ml_workbench() -> None:
    ensure_ml_state()

    st.title("Final ML Workbench")
    st.markdown("Configure source splits, tune candidate models, and compare turnover experiments without coding.")

    model_options = ml_workbench.list_model_options(st.session_state["ml_train_sources"])

    top_controls1, top_controls2 = st.columns(2)
    with top_controls1:
        if st.button("Select All Models", width="stretch"):
            st.session_state["ml_selected_models"] = ml_workbench.allowed_model_ids(st.session_state["ml_train_sources"])
            for model_meta in model_options:
                st.session_state[f"ml_model_enabled_{model_meta['id']}"] = True
            st.rerun()
    with top_controls2:
        if st.button("Reset Defaults", width="stretch"):
            st.session_state["ml_selected_models"] = ml_workbench.default_model_ids(st.session_state["ml_train_sources"])
            st.session_state["ml_selected_metrics"] = list(ml_workbench.DEFAULT_METRICS)
            st.session_state["ml_hyperparams"] = ml_workbench.default_hyperparams()
            st.session_state["ml_threshold"] = 0.7
            st.session_state["ml_split_mode"] = "final"
            st.session_state["ml_ranking_metric"] = "PR_AUC"
            st.session_state["ml_seed"] = 42
            st.session_state["ml_test_size"] = 0.0
            st.session_state["ml_val_size"] = 0.20
            st.session_state["ml_train_sources"] = ["file1", "file2"]
            st.session_state["ml_test_sources"] = ["file3"]
            st.rerun()

    st.markdown("---")
    _render_source_selection()
    model_options = ml_workbench.list_model_options(st.session_state["ml_train_sources"])

    st.markdown("---")
    _render_model_selection(model_options)
    st.markdown("---")
    _render_metric_selection()

    st.markdown("---")
    _render_feature_selection()

    st.markdown("---")
    st.markdown("### Run Configuration")
    st.session_state["ml_ranking_metric"] = st.selectbox(
        "Primary Metric for Best Model",
        options=st.session_state["ml_selected_metrics"] or ml_workbench.SUPPORTED_METRICS,
        index=(st.session_state["ml_selected_metrics"] or ml_workbench.SUPPORTED_METRICS).index(st.session_state["ml_ranking_metric"]) if st.session_state["ml_ranking_metric"] in (st.session_state["ml_selected_metrics"] or ml_workbench.SUPPORTED_METRICS) else 0,
        help="The workbench ranks models by this metric on the validation split created from the selected training sources.",
    )

    workflow_label = f"Workflow: {', '.join(st.session_state['ml_train_sources'])} train/validation -> {', '.join(st.session_state['ml_test_sources'])} test"
    st.info(workflow_label)

    st.session_state["ml_val_size"] = st.slider(
        "Validation Size within Training Sources",
        min_value=0.10,
        max_value=0.40,
        value=float(st.session_state["ml_val_size"]),
        step=0.05,
        help="Employee-grouped validation split inside the selected training sources. The selected test sources are evaluated after model selection.",
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
        start_run = st.button("Start Training Run", width="stretch")
    with run_col2:
        refresh_status = st.button("Refresh Status", width="stretch")

    if start_run:
        if not st.session_state["ml_selected_models"]:
            st.error("Select at least one model before starting a run.")
        elif not st.session_state["ml_selected_metrics"]:
            st.error("Select at least one comparison metric before starting a run.")
        else:
            job_config = {
                "selected_models": st.session_state["ml_selected_models"],
                "selected_metrics": st.session_state["ml_selected_metrics"],
                "val_size": st.session_state["ml_val_size"],
                "train_sources": st.session_state["ml_train_sources"],
                "test_sources": st.session_state["ml_test_sources"],
                "random_seed": st.session_state["ml_seed"],
                "ranking_metric": st.session_state["ml_ranking_metric"],
                "hyperparams": st.session_state["ml_hyperparams"],
                "drop_feature_groups": st.session_state["ml_drop_feature_groups"],
            }
            new_job_id = ml_workbench.start_training_job(job_config)
            st.session_state["ml_active_job_id"] = new_job_id
            st.success(f"Training job started: {new_job_id}")
            st.rerun()

    active_job_id = st.session_state.get("ml_active_job_id", "")
    if refresh_status and active_job_id:
        st.rerun()

    needs_auto_refresh = False

    if active_job_id:
        job = ml_workbench.get_job(active_job_id)
        if job:
            job_status = job.get("status", "queued")

            st.markdown("### Training Pipeline")
            render_pipeline_visualization(job)

            stage_label = job.get("stage_label", "Queued")
            active_model = job.get("active_model")
            active_model_label = ml_workbench.MODEL_REGISTRY.get(active_model, {}).get("label", "") if active_model else ""

            if job_status == "running":
                st.info(f"Stage: {stage_label}" + (f" -- Training **{active_model_label}**" if active_model_label else ""))
                needs_auto_refresh = True
            elif job_status == "queued":
                st.warning("Run is queued and will start automatically.")
                needs_auto_refresh = True
            elif job_status == "failed":
                st.error(f"Training failed: {job.get('error', 'unknown error')}")
            elif job_status == "completed":
                st.success(f"Training complete. Best model: **{job.get('best_model', 'N/A')}**")

            if job.get("logs"):
                with st.expander("Run Logs", expanded=False):
                    for line in job["logs"][-15:]:
                        st.text(line)

            if job_status == "completed" and job.get("results"):
                _render_results_section(job)

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
        st.dataframe(jobs_df, width="stretch", hide_index=True)
    else:
        st.caption("No training jobs yet.")

    if needs_auto_refresh:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="Final ML Workbench", page_icon="M", layout="wide")
    _require_access()
    render_ml_workbench()

