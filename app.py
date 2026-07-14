import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import random
from pathlib import Path
import joblib

# ---------------------------------------------------------------------------
# Visual / risk constants
# ---------------------------------------------------------------------------
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
RISK_COLOR_MAP = {
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f1c40f",
    "High Risk": "#e67e22",
    "Very High Risk": "#e74c3c",
}
# Continuous green->red scale for risk %
RISK_COLORSCALE = [
    [0.00, "#2ecc71"],
    [0.30, "#f1c40f"],
    [0.50, "#e67e22"],
    [0.70, "#e74c3c"],
    [1.00, "#922b21"],
]

# Resolve all paths relative to this script's directory, not CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.append(_SCRIPT_DIR)

try:
    from src.inference import TurnoverInferenceAPI
    from src.config import FEATURE_DESCRIPTIONS
    from src.datasets import discover_dataset_specs, read_excel_with_header_detection
    from src.final_dashboard import FINAL_ARTIFACT_PATH, load_final_dashboard_bundle, apply_final_what_if, available_options
    from src import chatbot
except ImportError as e:
    st.error(f"Could not import internal modules: {e}")
    st.stop()

st.set_page_config(
    page_title="Executive Turnover Dashboard",
    page_icon="T",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 18px 20px;
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] * { 
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] label { 
        font-size: 0.85rem; 
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 0;
        font-size: 1.1rem;
    }
    /* 1. Pin the Popover launcher to the bottom-right (clear of the sidebar) */
    div[data-testid="stPopover"] {
        position: fixed !important;
        right: 24px !important;
        left: auto !important;
        bottom: 24px !important;
        z-index: 1000000 !important;
        width: auto !important;
        height: auto !important;
    }

    /* 2. Force the circular blue button styling */
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopover"] button {
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        border: none !important;
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        box-shadow: 0 10px 24px rgba(2, 132, 199, 0.35) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease-in-out !important;
    }

    div[data-testid="stPopover"] button:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        transform: scale(1.05) !important;
    }

    /* Ensure the emoji/text inside the button stays centered and white */
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button span,
    div[data-testid="stPopover"] button div {
        font-size: 1.6rem !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* 3. Position the tooltip to the LEFT of the button */
    .chat-launcher-hint {
        position: fixed !important;
        right: 96px !important;
        left: auto !important;
        bottom: 37px !important;
        z-index: 999999 !important;
        background: #ffffff !important;
        border: 1px solid #dbeafe !important;
        border-radius: 999px !important;
        padding: 6px 14px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.18) !important;
        white-space: nowrap !important;
    }

    /* The little triangle pointing right toward the button */
    .chat-launcher-hint::after {
        content: "" !important;
        position: absolute !important;
        right: -7px !important;
        left: auto !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        border-left: 8px solid #ffffff !important;
        border-top: 6px solid transparent !important;
        border-bottom: 6px solid transparent !important;
    }

    /* 4. Fix the opened chat window position so it pops up above the button */
    div[data-testid="stPopoverContent"] {
        position: fixed !important;
        right: 24px !important;
        left: auto !important;
        bottom: 96px !important;
        top: auto !important;
        transform: none !important;
        width: min(92vw, 480px) !important;
        max-height: 85vh !important;
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.15) !important;
        z-index: 1000001 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Executive Turnover Dashboard")
st.caption("HR-focused turnover risk analysis. Click charts to drill in. ML training lives in `streamlit run ml_workbench_app.py`.")

# ---------------------------------------------------------------------------
# Dynamic Dataset Selection & Caching
# ---------------------------------------------------------------------------

st.sidebar.title("Dashboard Configuration")


def _source_label(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return " + ".join(str(item) for item in value)
    return str(value)


@st.cache_data
def discover_model_configurations() -> list[dict]:
    configs = []
    for artifact_path in sorted(Path("artifacts").glob("*.pkl")):
        try:
            artifact = joblib.load(artifact_path)
        except Exception:
            continue
        if not (artifact.get("candidate") and artifact.get("feature_columns") and artifact.get("pipeline") is not None):
            continue
        train_sources = artifact.get("train_sources", ["file1", "file2"])
        test_sources = artifact.get("test_sources", artifact.get("external_test_source", "file3"))
        created_by = artifact.get("created_by", "final_modeling")
        label = (
            f"{artifact.get('candidate', artifact_path.stem)} | "
            f"train: {_source_label(train_sources)} -> test: {_source_label(test_sources)}"
        )
        if artifact_path.name == FINAL_ARTIFACT_PATH.name:
            label = "Selected final model | " + label
        elif created_by == "ml_workbench":
            label = "Workbench | " + label
        configs.append({
            "label": label,
            "path": str(artifact_path),
            "candidate": artifact.get("candidate", artifact_path.stem),
            "train_sources": train_sources,
            "test_sources": test_sources,
            "created_by": created_by,
        })
    return configs


model_configs = discover_model_configurations()
dataset_specs, skipped_specs = discover_dataset_specs(include_root_excels=False)
spec_by_tag = {spec.tag: spec for spec in dataset_specs}
legacy_datasets = [
    spec.tag
    for spec in dataset_specs
    if os.path.exists(f"output/predictions_{spec.tag}.xlsx")
]

config_options = [config["label"] for config in model_configs]
config_options.extend([f"Legacy dataset | {tag}" for tag in legacy_datasets])

if not config_options:
    st.error("No trained predictions found. Run `python main.py` or train a model in `streamlit run ml_workbench_app.py`.")
    st.stop()

selected_config_label = st.sidebar.selectbox("Select Model Configuration", config_options)
selected_model_config = next((config for config in model_configs if config["label"] == selected_config_label), None)
is_final_dataset = selected_model_config is not None

if is_final_dataset:
    selected_dataset = selected_model_config["candidate"]
    DATA_PATH = "generated from selected artifact"
    RAW_DATA_PATH = "prepared selected test-source rows"
    API_PATH = selected_model_config["path"]
    st.sidebar.caption(f"Train: {_source_label(selected_model_config['train_sources'])}")
    st.sidebar.caption(f"Test: {_source_label(selected_model_config['test_sources'])}")
else:
    selected_dataset = selected_config_label.replace("Legacy dataset | ", "")
    selected_spec = spec_by_tag[selected_dataset]
    DATA_PATH = f"output/predictions_{selected_dataset}.xlsx"
    RAW_DATA_PATH = str(selected_spec.path)
    fixed_model = f"artifacts/model_pipeline_{selected_dataset}_fixed.pkl"
    random_model = f"artifacts/model_pipeline_{selected_dataset}_random.pkl"
    API_PATH = fixed_model if os.path.exists(fixed_model) else random_model

@st.cache_data
def load_dashboard_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_excel(filepath)

@st.cache_data
def load_raw_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df, _ = read_excel_with_header_detection(filepath)
    return df

@st.cache_resource
def load_inference_api(api_path: str):
    try:
        if os.path.exists(api_path):
            from src.inference import TurnoverInferenceAPI
            return TurnoverInferenceAPI(api_path)
        return None
    except Exception:
        return None

@st.cache_resource
def load_final_bundle_cached(api_path: str):
    return load_final_dashboard_bundle(api_path)

if is_final_dataset:
    final_bundle = load_final_bundle_cached(API_PATH)
    dashboard_df = final_bundle["dashboard_df"].copy()
    raw_df = final_bundle["raw_df"].copy()
    api = load_inference_api(API_PATH)
    final_metadata = final_bundle["metadata"]
    st.sidebar.caption(f"Model: {final_metadata['candidate']}")
    st.sidebar.caption(f"Artifact: {os.path.basename(API_PATH)}")
else:
    final_bundle = None
    final_metadata = {}
    dashboard_df = load_dashboard_data(DATA_PATH)
    raw_df = load_raw_data(RAW_DATA_PATH)
    api = load_inference_api(API_PATH)

if dashboard_df.empty:
    if is_final_dataset:
        st.error(f"Final predictions could not be built from **{API_PATH}**.")
    else:
        st.error(f"Predictions file **{DATA_PATH}** not found for legacy dataset `{selected_dataset}`.")
    st.stop()


def enrich_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-friendly display-name aliases (from FEATURE_DESCRIPTIONS) for
    raw columns so the drill-down visuals resolve the same column names across
    both final-bundle and legacy datasets. Intake modules stay untouched."""
    df = df.copy()
    for raw_col, display_name in FEATURE_DESCRIPTIONS.items():
        if raw_col in df.columns and display_name not in df.columns:
            df[display_name] = df[raw_col]
    return df


dashboard_df = enrich_display_columns(dashboard_df)


def ensure_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def process_chat_prompt(prompt: str, dashboard_df: pd.DataFrame, raw_df: pd.DataFrame, api, chat_emp, chat_dept) -> None:
    if not prompt or not prompt.strip():
        return

    clean_prompt = prompt.strip()
    st.session_state.chat_messages.append({"role": "user", "content": clean_prompt})

    with st.spinner("Thinking..."):
        reply = chatbot.chat(
            user_message=clean_prompt,
            history=st.session_state.chat_messages[:-1],
            dashboard_df=dashboard_df,
            raw_df=raw_df,
            api=api,
            selected_employee_id=chat_emp,
            selected_dept=chat_dept,
        )

    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


def render_floating_chat_widget(dashboard_df: pd.DataFrame, raw_df: pd.DataFrame, api) -> None:
    ensure_chat_state()

    _EXAMPLES = [
        "Why is employee <ID>'s risk so high?",
        "What are the top drivers of turnover across the company?",
        "How does risk correlate with salary and workload?",
        "Who are the 5 highest-risk employees?",
        "If I raise employee <ID>'s salary 10%, how does risk change?",
        "Which department has the highest average risk?",
    ]

    if "chat_suggested_q" not in st.session_state:
        st.session_state["chat_suggested_q"] = random.choice(_EXAMPLES)

    all_departments = sorted(
        [str(x) for x in dashboard_df["Budget Section"].dropna().unique() if str(x).strip() != ""]
    )
    dept_default = "All Departments"
    emp_default = "All Employees"

    if "chat_filter_dept" not in st.session_state:
        st.session_state["chat_filter_dept"] = dept_default
    valid_dept_options = [dept_default] + all_departments
    if st.session_state.get("chat_filter_dept") not in valid_dept_options:
        st.session_state["chat_filter_dept"] = dept_default

    st.markdown('<div class="chat-launcher-hint">HR Assistant</div>', unsafe_allow_html=True)

    with st.popover("💬", help="Open HR Assistant"):
        st.markdown("#### 💬 HR Assistant")

        # Editable scope row
        sc1, sc2 = st.columns(2)
        with sc1:
            st.caption("Department scope")
            selected_dept_opt = st.selectbox(
                "Department scope", options=valid_dept_options, key="chat_filter_dept", label_visibility="collapsed"
            )
        selected_dept = None if selected_dept_opt == dept_default else selected_dept_opt
        scoped_df = dashboard_df if not selected_dept else dashboard_df[
            dashboard_df["Budget Section"].astype(str) == selected_dept
        ]
        employee_options = sorted(scoped_df["Employee ID"].astype(str).unique().tolist())
        valid_emp_options = [emp_default] + employee_options
        if "chat_filter_emp" not in st.session_state:
            st.session_state["chat_filter_emp"] = emp_default
        if st.session_state.get("chat_filter_emp") not in valid_emp_options:
            st.session_state["chat_filter_emp"] = emp_default
        with sc2:
            st.caption("Employee scope")
            selected_emp_opt = st.selectbox(
                "Employee scope", options=valid_emp_options, key="chat_filter_emp", label_visibility="collapsed"
            )
        selected_emp = None if selected_emp_opt == emp_default else selected_emp_opt

        # Last 3 messages
        msgs = st.session_state.chat_messages
        if msgs:
            for msg in msgs[-3:]:
                prefix = "**You:** " if msg["role"] == "user" else "**Assistant:** "
                body = msg["content"][:140] + ("…" if len(msg["content"]) > 140 else "")
                st.markdown(prefix + body)

        st.divider()

        with st.form("floating_chat_form", clear_on_submit=True):
            typed_prompt = st.text_input(
                "Ask about turnover risk",
                value=st.session_state.get("chat_suggested_q", _EXAMPLES[0]),
                label_visibility="collapsed",
            )
            col_s, col_c = st.columns([3, 1])
            send_clicked = col_s.form_submit_button("Send", use_container_width=True)
            clear_clicked = col_c.form_submit_button("Clear", use_container_width=True)

        if send_clicked and typed_prompt and typed_prompt.strip():
            process_chat_prompt(typed_prompt.strip(), dashboard_df, raw_df, api, selected_emp, selected_dept)
            st.session_state["chat_suggested_q"] = random.choice(_EXAMPLES)
            st.rerun()

        if clear_clicked:
            st.session_state.chat_messages = []
            st.session_state["chat_suggested_q"] = random.choice(_EXAMPLES)
            st.rerun()

# ---------------------------------------------------------------------------
# Navigation State (allows programmatic transfer between views)
# ---------------------------------------------------------------------------
VIEWS = ["Macro (Company)", "Meso (Department)", "Micro (Employee)"]

if "active_view" not in st.session_state:
    st.session_state["active_view"] = VIEWS[0]
if "_nav_target" not in st.session_state:
    st.session_state["_nav_target"] = None
if "meso_selected_dept" not in st.session_state:
    st.session_state["meso_selected_dept"] = None
if "meso_risk_filter" not in st.session_state:
    st.session_state["meso_risk_filter"] = None
if "micro_selected_emp" not in st.session_state:
    st.session_state["micro_selected_emp"] = None


def goto_meso(dept: str) -> None:
    # Use _nav_target so we never modify active_view after the widget is instantiated
    st.session_state["_nav_target"] = VIEWS[1]
    st.session_state["meso_selected_dept"] = str(dept)
    st.session_state["meso_risk_filter"] = None
    st.rerun()


def goto_micro(emp_id) -> None:
    st.session_state["_nav_target"] = VIEWS[2]
    st.session_state["micro_selected_emp"] = emp_id
    st.rerun()


# Style the nav radio to look like tabs
st.markdown(
    """
    <style>
        div[role="radiogroup"] > label[data-baseweb="radio"] {
            background: #f1f5f9;
            border-radius: 8px;
            padding: 8px 16px;
            margin-right: 6px;
            cursor: pointer;
        }
        div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
            background: #0ea5e9;
            color: #fff;
        }
        div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) p { color: #fff !important; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)
# Apply pending programmatic navigation BEFORE the radio widget is instantiated
if st.session_state.get("_nav_target"):
    st.session_state["active_view"] = st.session_state["_nav_target"]
    st.session_state["_nav_target"] = None

st.radio("View", options=VIEWS, horizontal=True, key="active_view", label_visibility="collapsed")
active_view = st.session_state["active_view"]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default


def safe_val(val, default=0.0):
    return val if pd.notna(val) else default


def safe_str(val, default=""):
    return str(val) if pd.notna(val) else default


def add_donut_center(fig, value_pct: float, label: str = "Avg Risk") -> None:
    fig.add_annotation(
        text=f"<b>{value_pct:.1f}%</b><br><span style='font-size:11px;color:#64748b'>{label}</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=22, color="#0f172a"),
        align="center",
    )


def make_risk_donut(df: pd.DataFrame, key: str, height: int = 360, clickable: bool = False):
    """Donut of Risk Category counts with mean-risk center label. Optionally clickable for filtering."""
    counts = (
        df["Risk Category"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    )
    counts.columns = ["Risk Category", "Count"]
    mean_pct = (df["Turnover Probability"].mean() if not df.empty else 0.0) * 100

    fig = go.Figure(data=[
        go.Pie(
            labels=counts["Risk Category"],
            values=counts["Count"],
            hole=0.6,
            marker=dict(colors=[RISK_COLOR_MAP[r] for r in counts["Risk Category"]]),
            textinfo="value+label",
            textposition="auto",
            sort=False,
        )
    ])
    add_donut_center(fig, mean_pct)
    fig.update_layout(dragmode=False, showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=height)

    if clickable:
        return st.plotly_chart(
            fig, width="stretch", key=key,
            on_select="rerun", selection_mode="points",
            config={"displayModeBar": False},
        )
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    return None


def color_bar_by_risk(values_pct):
    """Return list of hex colors mapping each risk percentage (0-100) onto the green->red scale."""
    colors = []
    for v in values_pct:
        if pd.isna(v):
            colors.append("#cbd5e1")
        elif v < 30:
            colors.append("#2ecc71")
        elif v < 50:
            colors.append("#f1c40f")
        elif v < 70:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")
    return colors


# ---------------------------------------------------------------------------
# View: Macro
# ---------------------------------------------------------------------------
def view_macro():
    st.markdown("Monitor company-wide turnover risk. **Click a department bar** to drill into that team.")

    st.markdown("### Top Indicators")
    c1, c2, c3 = st.columns(3)
    total_employees = len(dashboard_df)
    avg_turnover_prob = dashboard_df["Turnover Probability"].mean()
    high_risk_count = int(
        dashboard_df[dashboard_df["Risk Category"].isin(["High Risk", "Very High Risk"])].shape[0]
    )
    c1.metric("Total Active Employees", f"{total_employees:,}")
    c2.metric("Company-Wide Turnover Risk", f"{avg_turnover_prob * 100:.1f}%")
    c3.metric("High-Risk Headcount", f"{high_risk_count:,}")

    st.markdown("---")
    st.markdown("### Risk Analysis Insights")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Risk Distribution")
        st.caption("Donut center shows the company-wide mean turnover probability.")
        make_risk_donut(dashboard_df, key="macro_donut", height=400)

    with right_col:
        st.markdown("#### Avg Turnover Risk by Budget Section (click a bar to drill in)")
        dept_risk = (
            dashboard_df.groupby("Budget Section")["Turnover Probability"]
            .agg(["mean", "count"]).reset_index()
        )
        dept_risk["Risk %"] = dept_risk["mean"] * 100
        dept_risk = dept_risk.sort_values("Risk %", ascending=False).head(15)

        fig_bar = go.Figure(data=[
            go.Bar(
                x=dept_risk["Budget Section"].astype(str),
                y=dept_risk["Risk %"],
                marker=dict(
                    color=dept_risk["Risk %"],
                    colorscale=RISK_COLORSCALE,
                    cmin=0, cmax=100,
                    showscale=True,
                    colorbar=dict(title="Risk %", thickness=10),
                ),
                customdata=np.stack([dept_risk["count"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>Avg Risk: %{y:.1f}%<br>Headcount: %{customdata[0]}<extra></extra>",
            )
        ])
        fig_bar.update_layout(
            dragmode=False, xaxis_title="Budget Section",
            yaxis_title="Avg Turnover Risk (%)", xaxis_type="category",
            margin=dict(t=20, b=20, l=40, r=20), height=400,
        )
        ev = st.plotly_chart(
            fig_bar, width="stretch", key="macro_dept_bar",
            on_select="rerun", selection_mode="points",
            config={"displayModeBar": False},
        )
        try:
            pts = ev.selection.points if ev else []
        except Exception:
            pts = (ev or {}).get("selection", {}).get("points", [])
        if pts:
            dept_clicked = pts[0].get("x")
            if dept_clicked:
                goto_meso(dept_clicked)

    st.markdown("---")
    st.markdown("#### Actionable Drivers: Risk Heatmap Across Employee Segments")
    st.caption(
        "Average turnover risk split across two categorical dimensions. Pick the dimensions "
        "to spot high-risk pockets (e.g., a contract type inside a city) that give HR concrete "
        "intervention angles. Pick any two of the available segments below."
    )

    # Candidate categorical dimensions (label -> max rows/cols to display).
    _CATEGORICAL_DIMS = [
        ("City of Residence", 18),
        ("Contract Type", 12),
        ("Employment Type", 12),
        ("Job Rank", 12),
        ("Marital Status", 8),
        ("Gender", 4),
    ]
    available_dims = [
        (col, cap) for col, cap in _CATEGORICAL_DIMS
        if col in dashboard_df.columns and dashboard_df[col].dropna().nunique() >= 2
    ]
    dim_caps = dict(available_dims)
    dim_labels = [d[0] for d in available_dims]

    if len(dim_labels) < 2:
        st.info("Not enough categorical features are available to build a risk heatmap.")
    else:
        pick_cols = st.columns(2)
        with pick_cols[0]:
            y_dim = st.selectbox("Rows (segment)", dim_labels, index=0, key="macro_heat_y")
        x_choices = [d for d in dim_labels if d != y_dim]
        with pick_cols[1]:
            x_dim = st.selectbox("Columns (segment)", x_choices, index=0, key="macro_heat_x")

        _min_n = 5
        _d = dashboard_df[[y_dim, x_dim, "Turnover Probability"]].dropna().copy()
        _d[y_dim] = _d[y_dim].astype(str)
        _d[x_dim] = _d[x_dim].astype(str)

        if _d.empty:
            st.info("No data available for the selected segments.")
        else:
            # Keep row categories with enough employees; relax if that empties the frame.
            row_counts = _d[y_dim].value_counts()
            keep_rows = row_counts[row_counts >= _min_n].index
            filtered = _d[_d[y_dim].isin(keep_rows)]
            if filtered.empty:
                filtered = _d
            # Cap rows to the highest-risk top-N for readability.
            row_avg = (
                filtered.groupby(y_dim)["Turnover Probability"]
                .mean().sort_values(ascending=False).head(dim_caps.get(y_dim, 15))
            )
            filtered = filtered[filtered[y_dim].isin(row_avg.index)]
            # Order columns by overall risk (highest on the right), cap for readability.
            col_avg = (
                filtered.groupby(x_dim)["Turnover Probability"]
                .mean().sort_values(ascending=True).tail(dim_caps.get(x_dim, 12))
            )
            _pivot = (
                filtered.groupby([y_dim, x_dim])["Turnover Probability"]
                .mean().unstack(level=x_dim)
            )
            _pivot = _pivot.reindex(columns=[c for c in col_avg.index if c in _pivot.columns])
            _pivot = _pivot.loc[row_avg.index[::-1]]  # highest-risk row on top
            _pivot_pct = _pivot * 100
            _text = [
                [f"{v:.0f}%" if pd.notna(v) else "" for v in row]
                for row in _pivot_pct.values
            ]
            fig_heat = go.Figure(data=go.Heatmap(
                z=_pivot_pct.values.tolist(),
                x=_pivot_pct.columns.astype(str).tolist(),
                y=_pivot_pct.index.astype(str).tolist(),
                colorscale=RISK_COLORSCALE,
                zmin=0, zmax=100,
                text=_text,
                texttemplate="%{text}",
                hoverongaps=False,
                colorbar=dict(title="Avg Risk %", thickness=14),
            ))
            fig_heat.update_layout(
                dragmode=False,
                title=f"Avg Risk: {y_dim} x {x_dim} (rows with >= {_min_n} employees)",
                xaxis_title=x_dim,
                yaxis_title=y_dim,
                xaxis_type="category",
                yaxis_type="category",
                height=max(380, len(_pivot) * 30 + 110),
                margin=dict(t=50, b=30, l=20, r=20),
            )
            st.plotly_chart(fig_heat, width="stretch", key="macro_heatmap", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# View: Meso
# ---------------------------------------------------------------------------
def render_demographics_panel(df: pd.DataFrame, key_prefix: str):
    """Render small bar charts: gender, age bucket, tenure bucket, employment type, marital status."""

    def small_bar(category_series: pd.Series, risk_series: pd.Series, title: str, key: str, order=None):
        if category_series.dropna().empty:
            st.caption(f"_{title}: no data_")
            return
        tmp = pd.DataFrame({"cat": category_series.astype(str), "risk": risk_series})
        agg = tmp.groupby("cat")["risk"].agg(["mean", "count"]).reset_index()
        if order is not None:
            agg["cat"] = pd.Categorical(agg["cat"], categories=order, ordered=True)
            agg = agg.sort_values("cat")
        else:
            agg = agg.sort_values("count", ascending=False)
        agg["Risk %"] = agg["mean"] * 100

        fig = go.Figure(data=[
            go.Bar(
                x=agg["cat"].astype(str),
                y=agg["count"],
                marker=dict(color=color_bar_by_risk(agg["Risk %"])),
                text=[f"{n}<br>{v:.0f}%" for n, v in zip(agg["count"], agg["Risk %"])],
                textposition="auto",
                hovertemplate="%{x}<br>Headcount: %{y}<br>Avg Risk: %{customdata:.1f}%<extra></extra>",
                customdata=agg["Risk %"],
            )
        ])
        fig.update_layout(
            dragmode=False, height=240, margin=dict(t=30, b=20, l=20, r=10),
            title=title, xaxis_type="category",
            yaxis_title="Headcount",
        )
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})

    risk = df["Turnover Probability"]

    row1 = st.columns(3)
    with row1[0]:
        if "Gender" in df.columns:
            small_bar(df["Gender"], risk, "Gender", f"{key_prefix}_gender")
    with row1[1]:
        if "Age" in df.columns:
            age = pd.to_numeric(df["Age"], errors="coerce")
            buckets = pd.cut(age, bins=[0, 25, 35, 45, 55, 65, 200],
                             labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
            small_bar(buckets, risk, "Age", f"{key_prefix}_age",
                      order=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    with row1[2]:
        if "Tenure (Months)" in df.columns:
            ten = pd.to_numeric(df["Tenure (Months)"], errors="coerce")
            buckets = pd.cut(ten, bins=[-1, 12, 36, 60, 120, 1e9],
                             labels=["<1y", "1-3y", "3-5y", "5-10y", "10y+"])
            small_bar(buckets, risk, "Tenure", f"{key_prefix}_tenure",
                      order=["<1y", "1-3y", "3-5y", "5-10y", "10y+"])

    row2 = st.columns(2)
    with row2[0]:
        if "Employment Type" in df.columns:
            small_bar(df["Employment Type"], risk, "Employment Type", f"{key_prefix}_emp_type")
    with row2[1]:
        if "Marital Status" in df.columns:
            small_bar(df["Marital Status"], risk, "Marital Status", f"{key_prefix}_marital")


def view_meso():
    st.markdown(
        "Analyze a single team. **Click the donut** to filter by risk category. "
        "**Click an employee row** in the table to inspect them."
    )

    departments = sorted(
        [str(x) for x in dashboard_df["Budget Section"].dropna().unique() if str(x).strip() != ""]
    )
    if not departments:
        st.warning("No departments available.")
        return

    pre_dept = st.session_state.get("meso_selected_dept")
    default_idx = departments.index(pre_dept) if pre_dept in departments else 0

    selected_dept = st.selectbox(
        "Select Team / Budget Section", options=departments, index=default_idx, key="meso_dept_widget"
    )
    if selected_dept != st.session_state.get("meso_selected_dept"):
        # User changed dept; clear filter
        st.session_state["meso_selected_dept"] = selected_dept
        st.session_state["meso_risk_filter"] = None
    st.session_state["chat_selected_dept"] = selected_dept

    dept_df = dashboard_df[dashboard_df["Budget Section"].astype(str) == selected_dept].copy()

    # Active risk-category filter (set by donut clicks)
    active_filter = st.session_state.get("meso_risk_filter")
    if active_filter and active_filter not in RISK_ORDER:
        active_filter = None
        st.session_state["meso_risk_filter"] = None

    if active_filter:
        cols = st.columns([8, 2])
        cols[0].info(f"Filter active: **{active_filter}** (click donut again or press Clear to remove)")
        if cols[1].button("Clear Filter", key="meso_clear_filter"):
            st.session_state["meso_risk_filter"] = None
            st.rerun()

    filtered_df = dept_df if not active_filter else dept_df[dept_df["Risk Category"] == active_filter]

    st.markdown(f"### Team Overview: {selected_dept}")

    # KPI row (uses dept_df for context; shows filtered headcount too)
    k1, k2, k3, k4 = st.columns(4)
    overall_avg = dashboard_df["Turnover Probability"].mean()
    dept_avg = dept_df["Turnover Probability"].mean()
    delta_prob = (dept_avg - overall_avg) * 100
    k1.metric("Team Members", f"{len(dept_df):,}")
    k2.metric(
        "Avg Team Risk", f"{dept_avg * 100:.1f}%",
        f"{delta_prob:+.1f}% vs Company", delta_color="inverse",
    )
    k3.metric(
        "High-Risk Members",
        f"{int(dept_df['Risk Category'].isin(['High Risk', 'Very High Risk']).sum()):,}",
    )
    k4.metric("In Current Filter", f"{len(filtered_df):,}")

    st.markdown("---")

    # Charts row 1 - donut (clickable, always full dept) + salary/workload (filtered)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Team Risk Distribution")
        st.caption("Center = team mean turnover probability. Click a slice to filter.")
        ev = make_risk_donut(dept_df, key="meso_donut", height=360, clickable=True)
        try:
            pts = ev.selection.points if ev else []
        except Exception:
            pts = (ev or {}).get("selection", {}).get("points", [])
        if pts:
            new_filter = pts[0].get("label")
            if new_filter:
                # Toggle: same slice clicked again clears the filter
                if new_filter == st.session_state.get("meso_risk_filter"):
                    st.session_state["meso_risk_filter"] = None
                else:
                    st.session_state["meso_risk_filter"] = new_filter
                st.rerun()

    with col2:
        st.markdown("#### Salary & Workload vs. Risk")
        st.caption("Shows how avg risk shifts across salary and workload bands (filtered view).")
        sub_a, sub_b = st.columns(2)

        with sub_a:
            if "Avg Salary (12m)" in filtered_df.columns and not filtered_df.empty:
                d = filtered_df[["Avg Salary (12m)", "Turnover Probability"]].dropna()
                if not d.empty:
                    try:
                        d["Bucket"] = pd.qcut(d["Avg Salary (12m)"], q=4, duplicates="drop")
                        agg = d.groupby("Bucket", observed=True)["Turnover Probability"].agg(["mean", "count"]).reset_index()
                        agg["Risk %"] = agg["mean"] * 100
                        agg["Label"] = agg["Bucket"].apply(
                            lambda b: f"{int(b.left/1000)}k-{int(b.right/1000)}k"
                        )
                        fig = go.Figure(data=[go.Bar(
                            x=agg["Label"], y=agg["Risk %"],
                            marker=dict(color=color_bar_by_risk(agg["Risk %"])),
                            text=[f"{v:.0f}%<br>n={n}" for v, n in zip(agg["Risk %"], agg["count"])],
                            textposition="auto",
                        )])
                        fig.update_layout(
                            dragmode=False, height=300, margin=dict(t=30, b=20, l=30, r=10),
                            title="Risk by Salary Band", xaxis_title="NIS", yaxis_title="Avg Risk (%)",
                        )
                        st.plotly_chart(fig, width="stretch", key="meso_salary_bar", config={"displayModeBar": False})
                    except Exception:
                        st.caption("_Not enough variation in salary to bucket._")

        with sub_b:
            if "Avg Workload" in filtered_df.columns and not filtered_df.empty:
                d = filtered_df[["Avg Workload", "Turnover Probability"]].dropna()
                if not d.empty:
                    try:
                        d["Bucket"] = pd.qcut(d["Avg Workload"], q=4, duplicates="drop")
                        agg = d.groupby("Bucket", observed=True)["Turnover Probability"].agg(["mean", "count"]).reset_index()
                        agg["Risk %"] = agg["mean"] * 100
                        quartile_labels = ["Low (Q1)", "Mid-Low (Q2)", "Mid-High (Q3)", "High (Q4)"]
                        agg["Label"] = [quartile_labels[i] if i < len(quartile_labels) else f"Q{i+1}" for i in range(len(agg))]
                        fig = go.Figure(data=[go.Bar(
                            x=agg["Label"], y=agg["Risk %"],
                            marker=dict(color=color_bar_by_risk(agg["Risk %"])),
                            text=[f"{v:.0f}%<br>n={n}" for v, n in zip(agg["Risk %"], agg["count"])],
                            textposition="auto",
                        )])
                        fig.update_layout(
                            dragmode=False, height=300, margin=dict(t=30, b=20, l=30, r=10),
                            title="Risk by Workload Band", xaxis_title="Workload",
                            yaxis_title="Avg Risk (%)",
                        )
                        st.plotly_chart(fig, width="stretch", key="meso_workload_bar", config={"displayModeBar": False})
                    except Exception:
                        st.caption("_Not enough variation in workload to bucket._")

    st.markdown("---")
    st.markdown("#### Unit Demographics (bar height = headcount, color = avg turnover risk)")
    if filtered_df.empty:
        st.info("No employees in current filter.")
    else:
        render_demographics_panel(filtered_df, key_prefix="meso_demo")

    st.markdown("---")
    title_suffix = f" - {active_filter}" if active_filter else ""
    st.markdown(f"#### Employees in {selected_dept}{title_suffix} (sorted by risk; click a row to inspect)")

    sorted_df = filtered_df.sort_values("Turnover Probability", ascending=False).reset_index(drop=True)

    display_cols = ["Employee ID", "Turnover Probability"]
    for c in ["Job Rank", "Maamad", "Role Code", "tafkidCode", "Tenure (Months)",
              "Avg Salary (12m)", "Avg Workload", "Avg Sick Days",
              "Employment Type", "Gender"]:
        if c in sorted_df.columns and c not in display_cols:
            display_cols.append(c)

    table_df = sorted_df[display_cols].copy()

    # Human-friendly formatting
    if "Tenure (Months)" in table_df.columns:
        table_df["Tenure (Years)"] = (pd.to_numeric(table_df["Tenure (Months)"], errors="coerce") / 12).round(1)
        table_df = table_df.drop(columns=["Tenure (Months)"])
    if "Avg Salary (12m)" in table_df.columns:
        table_df["Salary (\u20aa)"] = pd.to_numeric(table_df["Avg Salary (12m)"], errors="coerce").fillna(0).round(0).astype(int)
        table_df = table_df.drop(columns=["Avg Salary (12m)"])
    if "Avg Workload" in table_df.columns:
        table_df["Workload %"] = (pd.to_numeric(table_df["Avg Workload"], errors="coerce") * 100).fillna(0).round(0).astype(int)
        table_df = table_df.drop(columns=["Avg Workload"])
    if "Role Code" in table_df.columns:
        table_df["Role Code"] = pd.to_numeric(table_df["Role Code"], errors="coerce").fillna(0).astype(int)
    if "Avg Sick Days" in table_df.columns:
        table_df = table_df.rename(columns={"Avg Sick Days": "Sick Days/Mo"})

    if table_df.empty:
        st.info("No employees to display.")
    else:
        fmt = {"Turnover Probability": "{:.1%}"}
        if "Sick Days/Mo" in table_df.columns:
            fmt["Sick Days/Mo"] = "{:.2f}"
        if "Tenure (Years)" in table_df.columns:
            fmt["Tenure (Years)"] = "{:.1f}"
        styler = (
            table_df.style
            .background_gradient(
                subset=["Turnover Probability"], cmap="RdYlGn_r", vmin=0, vmax=1,
            )
            .format(fmt)
        )
        event = st.dataframe(
            styler, width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row",
            key="meso_employee_table",
        )
        try:
            rows = event.selection.rows if event else []
        except Exception:
            rows = (event or {}).get("selection", {}).get("rows", [])
        if rows:
            picked_emp = sorted_df.iloc[rows[0]]["Employee ID"]
            goto_micro(picked_emp)


# ---------------------------------------------------------------------------
# View: Micro
# ---------------------------------------------------------------------------
def view_micro():
    st.markdown("Inspect an individual employee. Adjust levers to simulate risk changes.")

    if raw_df.empty or api is None:
        st.warning(
            f"Raw data or model not properly loaded for `{selected_dataset}`. "
            f"Missing `{RAW_DATA_PATH}` or `{API_PATH}`. Please run `python main.py` first."
        )
        return

    if is_final_dataset:
        employee_id_col = getattr(api, 'employee_id_col', 'fictive_employee')
        time_col = getattr(api, 'time_col', 'calc_month')
    else:
        dataset_config = api.pipeline.get('dataset_config', {})
        employee_id_col = dataset_config.get('employee_id_col', 'fictive2')
        time_col = dataset_config.get('time_col')

    valid_employees = sorted(dashboard_df["Employee ID"].unique().tolist())
    options = [""] + valid_employees

    pre_emp = st.session_state.get("micro_selected_emp")
    default_idx = options.index(pre_emp) if pre_emp in valid_employees else 0

    col_sel, col_rst = st.columns([8, 2])
    with col_sel:
        selected_emp_id = st.selectbox(
            "Search / Select Employee ID", options=options, index=default_idx, key="micro_emp_widget"
        )
    st.session_state["micro_selected_emp"] = selected_emp_id or None
    st.session_state["chat_selected_emp_id"] = selected_emp_id or None

    if not selected_emp_id:
        st.info("Pick an employee above, or click a row in the Meso view to drill in here.")
        return

    emp_records = raw_df[raw_df[employee_id_col] == selected_emp_id].copy()
    if time_col and time_col in emp_records.columns:
        emp_records = emp_records.sort_values(time_col)
    latest_record = emp_records.iloc[-1].copy()

    # ------------------------------------------------------------------
    # Baseline lever values + slider ranges (snapped to the widget grid so
    # an untouched slider exactly reproduces the baseline).
    # ------------------------------------------------------------------
    def _snap(value, vmin, vmax, step):
        value = min(max(float(value), vmin), vmax)
        if step and step > 0:
            value = vmin + round((value - vmin) / step) * step
            value = min(max(value, vmin), vmax)
        return float(value)

    base_salary_raw = safe_float(latest_record.get("avg_Payment", 0.0))
    base_workload_raw = safe_float(latest_record.get("avg_omes", 0.0))   # ratio: 1.0 == 100% of full-time
    base_illness_raw = safe_float(latest_record.get("avg_illness", 0.0))
    base_maamad = safe_val(latest_record.get("Maamad", np.nan))
    base_seif = safe_val(latest_record.get("Seif", np.nan))
    base_contract = safe_val(latest_record.get("contract_type", np.nan))
    base_emp_type = safe_str(latest_record.get("TeurGroupHscm", ""))

    # Salary: 0 up to 2x the employee's baseline (baseline sits mid-slider).
    step_salary = 100.0
    min_salary = 0.0
    base_salary_clamped = max(0.0, base_salary_raw)
    max_salary = max(2.0 * base_salary_clamped, 1000.0)
    base_salary = _snap(base_salary_clamped, min_salary, max_salary, step_salary)

    # Sick days: 0 up to 2x the employee's baseline (baseline sits mid-slider).
    step_illness = 0.5
    base_illness_clamped = max(0.0, base_illness_raw)
    max_illness = max(2.0 * base_illness_clamped, 5.0)
    base_illness = _snap(base_illness_clamped, 0.0, max_illness, step_illness)

    # Workload: shown as a percentage of full-time, 0-200%.
    max_workload_pct = 200.0
    step_workload = 5.0
    base_workload_pct = _snap(base_workload_raw * 100.0, 0.0, max_workload_pct, step_workload)

    if f"sl_salary_{selected_emp_id}" not in st.session_state:
        st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
        st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload_pct
        st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
        st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
        st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
        st.session_state[f"sel_contract_{selected_emp_id}"] = base_contract
        st.session_state[f"sel_emp_type_{selected_emp_id}"] = base_emp_type

    def reset_levers():
        st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
        st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload_pct
        st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
        st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
        st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
        st.session_state[f"sel_contract_{selected_emp_id}"] = base_contract
        st.session_state[f"sel_emp_type_{selected_emp_id}"] = base_emp_type

    with col_rst:
        st.button("🔄 Reset Levers", on_click=reset_levers)

    # ------------------------------------------------------------------
    # Lever application helper. The baseline (Current) and the simulated
    # (What-If) predictions BOTH flow through this same transform so that,
    # when no lever is moved, the two are identical.
    # ------------------------------------------------------------------
    def _apply_levers(records, *, salary, workload_ratio, illness, contract, maamad, seif, emp_type):
        if is_final_dataset:
            return apply_final_what_if(
                records,
                salary=salary,
                workload=workload_ratio,
                illness=illness,
                contract_type=contract,
                maamad=maamad,
                seif=seif,
            )
        r = records.copy()
        if r.empty:
            return r
        idx = r.index[-1]
        r.at[idx, "avg_Payment"] = salary
        r.at[idx, "avg_omes"] = workload_ratio
        r.at[idx, "avg_illness"] = illness
        r.at[idx, "Maamad"] = maamad
        if "TeurGroupHscm" in r.columns:
            r.at[idx, "TeurGroupHscm"] = emp_type
        return r

    base_levers = dict(
        salary=base_salary,
        workload_ratio=base_workload_pct / 100.0,
        illness=base_illness,
        contract=base_contract,
        maamad=base_maamad,
        seif=base_seif,
        emp_type=base_emp_type,
    )
    baseline_records = _apply_levers(emp_records, **base_levers)

    try:
        # "Current" uses the raw record so it matches the Macro/Meso tabs.
        current_prob, current_category = api.predict_risk(emp_records)
    except Exception as e:
        st.error(f"Error predicting current risk: {e}")
        match = dashboard_df[dashboard_df["Employee ID"] == selected_emp_id]
        if not match.empty:
            current_prob = match.iloc[0]["Turnover Probability"]
            current_category = match.iloc[0]["Risk Category"]
        else:
            current_prob, current_category = 0.0, "Unknown"

    # -------------------------------------------------------------------------
    # TOP SECTION: Employee demographics / profile strip
    # -------------------------------------------------------------------------
    tenure = safe_float(latest_record.get("vetek_months", 0))
    age = safe_float(latest_record.get("age", 0))
    gender = safe_str(latest_record.get("gender", "-"))
    city = safe_str(latest_record.get("Yishuv", "-"))
    role = safe_str(latest_record.get("tafkidCode", "-"))
    emp_type = safe_str(latest_record.get("TeurGroupHscm", "-"))
    marital = safe_str(latest_record.get("EMP_Matzav_Mishpachti", "-"))

    risk_color = RISK_COLOR_MAP.get(current_category, "#64748b")
    st.markdown(
        f"""<div style="background:#f8fafc;border-left:6px solid {risk_color};border-radius:10px;
        padding:10px 18px;margin-bottom:10px;display:flex;align-items:center;gap:1rem;">
        <span style="font-size:1.05rem;font-weight:700;">{selected_emp_id}</span>
        <span style="background:{risk_color};color:#fff;border-radius:999px;padding:2px 12px;
        font-size:0.82rem;font-weight:600;">{current_category}</span>
        <span style="margin-left:auto;font-size:1.7rem;font-weight:700;color:{risk_color};">
        {current_prob*100:.1f}% risk</span></div>""",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
    d1.metric("Age", f"{int(age)} y")
    d2.metric("Tenure", f"{tenure/12:.1f} y")
    d3.metric("Gender", gender)
    d4.metric("Marital", marital[:8])
    d5.metric("City", city[:10])
    d6.metric("Role", str(role)[:8])
    d7.metric("Emp. Type", emp_type[:8])

    st.markdown("---")

    # -------------------------------------------------------------------------
    # MIDDLE: Levers (left) + Combined risk chart (right)
    # -------------------------------------------------------------------------
    col_levers, col_chart = st.columns([1, 1], gap="large")

    with col_levers:
        st.markdown("### What-If Scenario Levers")

        new_salary = st.slider(
            "Average Salary (NIS)",
            min_value=min_salary, max_value=max_salary,
            step=step_salary, key=f"sl_salary_{selected_emp_id}",
        )

        new_workload_pct = st.slider(
            "Average Workload (% of full-time)",
            min_value=0.0, max_value=max_workload_pct,
            step=step_workload, format="%.0f%%", key=f"sl_workload_{selected_emp_id}",
        )
        new_workload_ratio = new_workload_pct / 100.0

        new_illness = st.slider(
            "Average Sick Days",
            min_value=0.0, max_value=max_illness,
            step=step_illness, key=f"sl_ill_{selected_emp_id}",
        )

        maamad_opts = available_options(raw_df, 'Maamad', base_maamad)
        new_maamad = st.selectbox("Job Rank (Maamad)", options=maamad_opts, key=f"sel_maamad_{selected_emp_id}") if maamad_opts else base_maamad

        if is_final_dataset:
            contract_opts = available_options(raw_df, 'contract_type', base_contract)
            new_contract = st.selectbox("Contract Type", options=contract_opts, key=f"sel_contract_{selected_emp_id}") if contract_opts else base_contract
            seif_opts = available_options(raw_df, 'Seif', base_seif)
            new_seif = st.selectbox("Budget Section (Seif)", options=seif_opts, key=f"sel_seif_{selected_emp_id}") if seif_opts else base_seif
            new_emp_type = base_emp_type
        else:
            emp_type_opts = [str(x) for x in raw_df["TeurGroupHscm"].dropna().unique().tolist()] if "TeurGroupHscm" in raw_df.columns else []
            if base_emp_type and base_emp_type not in emp_type_opts:
                emp_type_opts.append(base_emp_type)
            emp_type_opts = sorted(set(emp_type_opts))
            new_emp_type = st.selectbox("Employment Type", options=emp_type_opts, key=f"sel_emp_type_{selected_emp_id}") if emp_type_opts else base_emp_type
            new_contract = base_contract
            new_seif = base_seif

    # Apply modifications & predict (same transform as the baseline).
    mod_records = _apply_levers(
        emp_records,
        salary=new_salary,
        workload_ratio=new_workload_ratio,
        illness=new_illness,
        contract=new_contract,
        maamad=new_maamad,
        seif=new_seif,
        emp_type=new_emp_type,
    )

    try:
        # Isolate the lever effect through the SAME transform (recompute drift
        # cancels out in the subtraction), then apply it to the raw current so
        # "no lever change" reproduces the current risk exactly.
        baseline_prob, _ = api.predict_risk(baseline_records)
    except Exception:
        baseline_prob = current_prob

    try:
        mod_prob, _ = api.predict_risk(mod_records)
        new_prob = float(np.clip(current_prob + (mod_prob - baseline_prob), 0.0, 1.0))
        new_category = api.risk_category(new_prob)
    except Exception as e:
        st.error(f"Inference error: {e}")
        new_prob = current_prob
        new_category = current_category

    two_stage_baseline = None
    if is_final_dataset:
        try:
            two_stage_baseline = api.two_stage_decomposition(baseline_records)
        except Exception:
            two_stage_baseline = None

    # Per-lever risk impact (percentage points): change ONE lever at a time from
    # the baseline and measure how far it moves predicted risk. Includes the
    # categorical levers (Job Rank, Contract Type, Budget Section) too.
    lever_scenarios = [
        ("Salary", {**base_levers, "salary": new_salary}),
        ("Workload", {**base_levers, "workload_ratio": new_workload_ratio}),
        ("Sick Days", {**base_levers, "illness": new_illness}),
        ("Job Rank", {**base_levers, "maamad": new_maamad}),
    ]
    if is_final_dataset:
        lever_scenarios.append(("Contract Type", {**base_levers, "contract": new_contract}))
        lever_scenarios.append(("Budget Section", {**base_levers, "seif": new_seif}))
    else:
        lever_scenarios.append(("Employment Type", {**base_levers, "emp_type": new_emp_type}))

    impact_pp = {}
    for label, scenario in lever_scenarios:
        try:
            recs = _apply_levers(emp_records, **scenario)
            p, _ = api.predict_risk(recs)
            impact_pp[label] = (p - baseline_prob) * 100.0
        except Exception:
            impact_pp[label] = 0.0
    # Biggest movers on top.
    impact_items = sorted(impact_pp.items(), key=lambda kv: abs(kv[1]))

    # -------------------------------------------------------------------------
    # COMBINED CHART: current gauge | simulated gauge + diffs bar (one figure)
    # -------------------------------------------------------------------------
    with col_chart:
        st.markdown("### Risk: Current vs. Simulated")
        st.caption("Left gauge is the current baseline. Right gauge reflects your lever settings — identical to the left until you move a lever.")

        gauge_steps = [
            {"range": [0, 30], "color": "#2ecc71"},
            {"range": [30, 50], "color": "#f1c40f"},
            {"range": [50, 70], "color": "#e67e22"},
            {"range": [70, 100], "color": "#e74c3c"},
        ]

        fig_combined = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            subplot_titles=[
                f"Current - {current_category}",
                f"Simulated - {new_category}",
                "Per-Lever Risk Impact (pp)",
            ],
            vertical_spacing=0.18,
            row_heights=[0.55, 0.45],
        )

        fig_combined.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "bgcolor": "white",
                    "steps": gauge_steps,
                    "threshold": {"line": {"color": "black", "width": 4},
                                  "thickness": 0.75, "value": current_prob * 100},
                },
            ),
            row=1, col=1,
        )

        fig_combined.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=new_prob * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                delta={
                    "reference": current_prob * 100, "position": "top",
                    "valueformat": ".1f", "suffix": "%",
                    "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"},
                },
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "bgcolor": "white",
                    "steps": gauge_steps,
                    "threshold": {"line": {"color": "black", "width": 4},
                                  "thickness": 0.75, "value": new_prob * 100},
                },
            ),
            row=1, col=2,
        )

        diff_colors = [
            "lightgray" if abs(v) < 1e-9 else ("#e74c3c" if v > 0 else "#2ecc71")
            for _, v in impact_items
        ]

        fig_combined.add_trace(
            go.Bar(
                x=[v for _, v in impact_items],
                y=[lbl for lbl, _ in impact_items],
                orientation="h",
                marker_color=diff_colors,
                text=[f"+{v:.1f} pp" if v > 0 else f"{v:.1f} pp" for _, v in impact_items],
                textposition="auto",
                showlegend=False,
            ),
            row=2, col=1,
        )

        fig_combined.update_layout(
            dragmode=False,
            height=580,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        fig_combined.update_xaxes(
            row=2, zeroline=True, zerolinewidth=2, zerolinecolor="black",
            title_text="Risk change vs. baseline (pp)",
        )
        fig_combined.update_yaxes(row=2, title_text="")
        st.plotly_chart(fig_combined, width="stretch", config={"displayModeBar": False})

    if two_stage_baseline is not None:
        st.markdown("---")
        st.markdown("#### How this risk breaks down")
        st.caption(
            "The model separates each employee's risk into a part you cannot influence and a part you can. "
            "Watch the actionable part to see whether your lever changes actually reduce risk. "
            "(\"pp\" = percentage points.)"
        )
        # Anchor to the gauges: fixed (Stage 1) + actionable = total shown above.
        stage1 = float(np.clip(two_stage_baseline["stage1_probability"], 0.0, 1.0))
        combined_base = current_prob
        combined_new = new_prob
        stage2_base = combined_base - stage1
        stage2_new = combined_new - stage1

        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Fixed risk (who they are)",
            f"{stage1*100:.1f}%",
            help="Baseline risk from age, tenure, gender and marital status. These can't be changed, so this stays fixed no matter what levers you move.",
        )
        b2.metric(
            "Actionable risk - now",
            f"{stage2_base*100:+.1f} pp",
            help="How much the employee's CURRENT working conditions (salary, workload, sick days, contract, rank) add to (+) or subtract from (-) the fixed risk.",
        )
        b3.metric(
            "Actionable risk - after changes",
            f"{stage2_new*100:+.1f} pp",
            delta=f"{(stage2_new - stage2_base)*100:+.1f} pp vs now",
            delta_color="inverse",
            help="The same actionable component recomputed with your what-if levers. The arrow shows how far your changes moved it (down/green is good).",
        )

        _arrow = "▼" if combined_new < combined_base else ("▲" if combined_new > combined_base else "■")
        st.markdown(
            f"**Bottom line:** total predicted risk goes from **{combined_base*100:.1f}%** "
            f"to **{combined_new*100:.1f}%** ({_arrow} {(combined_new - combined_base)*100:+.1f} pp) "
            f"once your changes are applied. The fixed part stays at {stage1*100:.1f}%."
        )


# ---------------------------------------------------------------------------
# Render selected view
# ---------------------------------------------------------------------------
if active_view == VIEWS[0]:
    view_macro()
elif active_view == VIEWS[1]:
    view_meso()
else:
    view_micro()

render_floating_chat_widget(dashboard_df, raw_df, api)
