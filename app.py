import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import random

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
# Continuous red->yellow->green scale for risk %
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
    /* 1. Pin the Popover wrapper to the exact bottom left */
    div[data-testid="stPopover"] {
        position: fixed !important;
        left: 20px !important;
        bottom: 20px !important;
        z-index: 1000 !important;
        width: auto !important;
        height: auto !important;
    }

    /* 2. Force the circular blue button styling */
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
    div[data-testid="stPopover"] button p {
        font-size: 1.6rem !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* 3. Position the tooltip perfectly next to the button */
    .chat-launcher-hint {
        position: fixed !important;
        left: 95px !important;
        bottom: 33px !important;
        z-index: 999 !important;
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

    /* The little triangle pointing left toward the button */
    .chat-launcher-hint::after {
        content: "" !important;
        position: absolute !important;
        left: -7px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        border-right: 8px solid #ffffff !important;
        border-top: 6px solid transparent !important;
        border-bottom: 6px solid transparent !important;
    }

    /* 4. Fix the opened chat window position so it pops up above the button */
    div[data-testid="stPopoverContent"] {
        position: fixed !important;
        left: 20px !important;
        bottom: 90px !important;
        top: auto !important;
        transform: none !important;
        width: min(92vw, 480px) !important;
        max-height: 85vh !important;
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Executive Turnover Dashboard")
st.caption("HR-focused turnover risk analysis. Click charts to drill in. ML training lives in `streamlit run ml_workbench_app.py`.")

# ---------------------------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------------------------
# DATA_PATH = "output/predictions_output.xlsx"
DATA_PATH = "output/predictions_first_file.xlsx"
RAW_DATA_PATH = "data/raw/first_file.xlsx"

@st.cache_data
def load_dashboard_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_excel(filepath)

@st.cache_data
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_excel(filepath)

@st.cache_resource
def load_inference_api():
    try:
        return TurnoverInferenceAPI()
    except Exception as e:
        return None

dashboard_df = load_dashboard_data()
raw_df = load_raw_data()
api = load_inference_api()

if dashboard_df.empty:
    st.error(f"Data file **{DATA_PATH}** not found. Please run `python main.py` first.")
    st.stop()


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
        "Which department has the highest average risk?",
        "Who are the 5 highest-risk employees?",
        "How does risk vary by job rank?",
        "Show employees with risk above 70%.",
        "Give me the overall risk statistics.",
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
                "", options=valid_dept_options, key="chat_filter_dept", label_visibility="collapsed"
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
                "", options=valid_emp_options, key="chat_filter_emp", label_visibility="collapsed"
            )
        selected_emp = None if selected_emp_opt == emp_default else selected_emp_opt

        # Last 3 messages
        msgs = st.session_state.chat_messages
        if msgs:
            for msg in msgs[-3:]:
                prefix = "**You:** " if msg["role"] == "user" else "**🤖:** "
                body = msg["content"][:140] + ("…" if len(msg["content"]) > 140 else "")
                st.markdown(prefix + body)

        st.divider()

        with st.form("floating_chat_form", clear_on_submit=True):
            typed_prompt = st.text_input(
                "",
                value=st.session_state.get("chat_suggested_q", _EXAMPLES[0]),
                label_visibility="collapsed",
            )
            col_s, col_c = st.columns([3, 1])
            send_clicked = col_s.form_submit_button("Send ▶", use_container_width=True)
            clear_clicked = col_c.form_submit_button("✕ Clear", use_container_width=True)

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
    st.markdown("#### Actionable Drivers: Risk by City of Residence & Commute")
    st.caption(
        "Tenure is largely fixed. These features (commute distance, city) tend to drive risk and "
        "give HR concrete intervention angles (relocation support, hybrid work, transport)."
    )

    city_avail = "City of Residence" in dashboard_df.columns
    commute_avail = "Estimated Commute Distance" in dashboard_df.columns

    if city_avail and commute_avail:
        _d = dashboard_df[["City of Residence", "Estimated Commute Distance", "Turnover Probability"]].dropna().copy()
        if not _d.empty:
            _bins = [-0.001, 5, 15, 30, 50, 1e9]
            _clabels = ["0-5 km", "5-15 km", "15-30 km", "30-50 km", "50+ km"]
            _d["Commute Bucket"] = pd.cut(_d["Estimated Commute Distance"], bins=_bins, labels=_clabels)
            _min_n = 5
            _valid_cities = _d["City of Residence"].value_counts()
            _valid_cities = _valid_cities[_valid_cities >= _min_n].index
            _d = _d[_d["City of Residence"].isin(_valid_cities)]
            _city_avg = (
                _d.groupby("City of Residence")["Turnover Probability"]
                .mean().sort_values(ascending=False).head(15)
            )
            _d = _d[_d["City of Residence"].isin(_city_avg.index)]
            _pivot = (
                _d.groupby(["City of Residence", "Commute Bucket"], observed=True)["Turnover Probability"]
                .mean().unstack(level="Commute Bucket")
            )
            _pivot = _pivot.reindex(columns=[l for l in _clabels if l in _pivot.columns])
            # Sort highest risk at top
            _pivot = _pivot.loc[_city_avg.index[::-1]]
            _pivot_pct = _pivot * 100
            _text = [
                [f"{v:.0f}%" if pd.notna(v) else "" for v in row]
                for row in _pivot_pct.values
            ]
            fig_heat = go.Figure(data=go.Heatmap(
                z=_pivot_pct.values.tolist(),
                x=_pivot_pct.columns.astype(str).tolist(),
                y=_pivot_pct.index.tolist(),
                colorscale=RISK_COLORSCALE,
                zmin=0, zmax=100,
                text=_text,
                texttemplate="%{text}",
                hoverongaps=False,
                colorbar=dict(title="Avg Risk %", thickness=14),
            ))
            fig_heat.update_layout(
                dragmode=False,
                title=f"Risk Heatmap: City × Commute Distance (top {len(_pivot)} cities, min {_min_n} employees)",
                xaxis_title="Commute Distance Bucket",
                yaxis_title="City of Residence",
                height=max(380, len(_pivot) * 30 + 90),
                margin=dict(t=50, b=30, l=20, r=20),
            )
            st.plotly_chart(fig_heat, width="stretch", key="macro_heatmap", config={"displayModeBar": False})
        else:
            st.info("No city/commute data available.")
    else:
        missing = []
        if not city_avail: missing.append("`City of Residence`")
        if not commute_avail: missing.append("`Estimated Commute Distance`")
        st.info(f"{' and '.join(missing)} column(s) not available for heatmap.")


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

    # Charts row 1 — donut (clickable, always full dept) + salary/workload (filtered)
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
    title_suffix = f" — {active_filter}" if active_filter else ""
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
            "Raw data or model not properly loaded. "
            "Missing `data/raw/first_file.xlsx` or `artifacts/model_pipeline.pkl`. "
            "Please run `python main.py` first."
        )
        return

    employee_id_col = "fictive2"
    time_col = "fictive-ovedmiun"

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
    if time_col in emp_records.columns:
        emp_records = emp_records.sort_values(time_col)
    latest_record = emp_records.iloc[-1].copy()

    try:
        current_prob, current_category = api.predict_risk(emp_records)
    except Exception as e:
        st.error(f"Error predicting current risk: {e}")
        match = dashboard_df[dashboard_df["Employee ID"] == selected_emp_id]
        if not match.empty:
            current_prob = match.iloc[0]["Turnover Probability"]
            current_category = match.iloc[0]["Risk Category"]
        else:
            current_prob, current_category = 0.0, "Unknown"

    base_salary = safe_float(latest_record.get("avg_Payment", 4000.0))
    base_workload = safe_float(latest_record.get("avg_omes", 0.0))
    base_illness = safe_float(latest_record.get("avg_illness", 0.0))
    base_maamad = safe_val(latest_record.get("Maamad", np.nan))
    base_emp_type = safe_str(latest_record.get("TeurGroupHscm", ""))

    if f"sl_salary_{selected_emp_id}" not in st.session_state:
        st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
        st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
        st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
        st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
        st.session_state[f"sel_emp_type_{selected_emp_id}"] = base_emp_type

    def reset_levers():
        st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
        st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
        st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
        st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
        st.session_state[f"sel_emp_type_{selected_emp_id}"] = base_emp_type

    with col_rst:
        st.button("🔄 Reset Levers", on_click=reset_levers)

    # -------------------------------------------------------------------------
    # TOP SECTION: Employee demographics / profile strip
    # -------------------------------------------------------------------------
    tenure = safe_float(latest_record.get("vetek_months", 0))
    age = safe_float(latest_record.get("age", 0))
    gender = safe_str(latest_record.get("gender", "—"))
    city = safe_str(latest_record.get("Yishuv", "—"))
    role = safe_str(latest_record.get("tafkidCode", "—"))
    emp_type = safe_str(latest_record.get("TeurGroupHscm", "—"))
    marital = safe_str(latest_record.get("EMP_Matzav_Mishpachti", "—"))

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

        # Salary: ±100% of baseline (symmetric center), floor at 1000
        _sal_base = base_salary if base_salary > 0 else 10000.0
        _sal_min = 1000.0
        _sal_max = max(_sal_base * 2, _sal_base + 5000.0)
        new_salary = st.slider(
            f"Salary (NIS) — baseline: ₪{int(_sal_base):,}",
            min_value=float(_sal_min), max_value=float(_sal_max),
            step=500.0, key=f"sl_salary_{selected_emp_id}",
        )

        # Workload ratio (0=0%, 2=200% of contracted hours): ±100% of baseline, bounds [0, 2]
        _wl_base = base_workload if base_workload > 0 else 1.0
        _wl_max = min(2.0, _wl_base * 2)
        if _wl_max < 0.1: _wl_max = 2.0
        new_workload = st.slider(
            f"Workload (ratio of contracted hrs) — baseline: {_wl_base*100:.0f}%",
            min_value=0.0, max_value=_wl_max,
            step=0.01, format="%.2f", key=f"sl_workload_{selected_emp_id}",
        )
        st.caption(f"Selected: {new_workload*100:.0f}% of contracted hours")

        # Sick days/month: ±100% of baseline, bounds [0, 30]
        _ill_base = base_illness if base_illness > 0 else 1.0
        _ill_max = min(30.0, max(_ill_base * 2, _ill_base + 5.0))
        new_illness = st.slider(
            f"Sick Days/Month — baseline: {_ill_base:.1f}",
            min_value=0.0, max_value=float(_ill_max),
            step=0.5, key=f"sl_ill_{selected_emp_id}",
        )

        maamad_opts = raw_df["Maamad"].dropna().unique().tolist()
        if base_maamad not in maamad_opts and pd.notna(base_maamad): maamad_opts.append(base_maamad)
        maamad_opts = sorted(list(set(maamad_opts)))
        new_maamad = st.selectbox("Job Rank (Maamad)", options=maamad_opts, key=f"sel_maamad_{selected_emp_id}")

        emp_type_opts = [str(x) for x in raw_df["TeurGroupHscm"].dropna().unique().tolist()]
        if base_emp_type and base_emp_type not in emp_type_opts: emp_type_opts.append(base_emp_type)
        emp_type_opts = sorted(list(set(emp_type_opts)))
        _emp_default = base_emp_type if base_emp_type in emp_type_opts else (emp_type_opts[0] if emp_type_opts else "")
        new_emp_type = st.selectbox(
            "Employment Type",
            options=emp_type_opts,
            index=emp_type_opts.index(_emp_default) if _emp_default in emp_type_opts else 0,
            key=f"sel_emp_type_{selected_emp_id}",
        )

    # Apply modifications & predict
    mod_records = emp_records.copy()
    latest_idx = mod_records.index[-1]
    mod_records.at[latest_idx, "avg_Payment"] = new_salary
    mod_records.at[latest_idx, "avg_omes"] = new_workload
    mod_records.at[latest_idx, "avg_illness"] = new_illness
    mod_records.at[latest_idx, "Maamad"] = new_maamad
    mod_records.at[latest_idx, "TeurGroupHscm"] = new_emp_type

    try:
        new_prob, new_category = api.predict_risk(mod_records)
    except Exception as e:
        st.error(f"Inference error: {e}")
        new_prob = current_prob
        new_category = current_category

    # Lever diffs
    orig_salary = safe_float(latest_record.get("avg_Payment", 0))
    orig_workload = safe_float(latest_record.get("avg_omes", 0))
    orig_illness = safe_float(latest_record.get("avg_illness", 0))
    diffs = {
        "Salary": ((new_salary - orig_salary) / max(1, orig_salary)) * 100,
        "Workload": ((new_workload - orig_workload) / max(0.01, orig_workload)) * 100,
        "Sick Days": ((new_illness - orig_illness) / max(0.1, orig_illness)) * 100,
    }

    # -------------------------------------------------------------------------
    # COMBINED CHART: current gauge | simulated gauge + diffs bar (one figure)
    # -------------------------------------------------------------------------
    with col_chart:
        st.markdown("### Risk: Current vs. Simulated")
        st.caption("Left gauge is fixed baseline. Right gauge reflects your lever settings.")

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
                f"Current — {current_category}",
                f"Simulated — {new_category}",
                "Lever Changes vs. Baseline",
            ],
            vertical_spacing=0.18,
            row_heights=[0.60, 0.40],
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

        diff_colors = []
        for feat, val in diffs.items():
            if val == 0:
                diff_colors.append("lightgray")
            elif feat in ["Workload", "Sick Days"]:
                diff_colors.append("#e74c3c" if val > 0 else "#2ecc71")
            else:
                diff_colors.append("#2ecc71" if val > 0 else "#e74c3c")

        fig_combined.add_trace(
            go.Bar(
                x=list(diffs.values()),
                y=list(diffs.keys()),
                orientation="h",
                marker_color=diff_colors,
                text=[f"+{v:.1f}%" if v > 0 else f"{v:.1f}%" for v in diffs.values()],
                textposition="auto",
                showlegend=False,
            ),
            row=2, col=1,
        )

        fig_combined.update_layout(
            dragmode=False,
            height=540,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        fig_combined.update_xaxes(
            row=2, zeroline=True, zerolinewidth=2, zerolinecolor="black",
            title_text="% Change vs. Baseline",
        )
        fig_combined.update_yaxes(row=2, title_text="")
        st.plotly_chart(fig_combined, width="stretch", config={"displayModeBar": False})


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
