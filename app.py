import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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
    /* Floating chat launcher + hint bubble */
    div[data-testid="stPopover"] {
        position: fixed;
        right: 1.25rem;
        bottom: 1.25rem;
        z-index: 1000;
    }
    div[data-testid="stPopover"] button {
        width: 56px;
        height: 56px;
        border-radius: 999px;
        border: none;
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(2, 132, 199, 0.35);
    }
    div[data-testid="stPopover"] button:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1);
    }
    div[data-testid="stPopoverContent"] {
        width: min(92vw, 390px) !important;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.15);
    }
    .chat-launcher-hint {
        position: fixed;
        right: 5.4rem;
        bottom: 2.1rem;
        z-index: 999;
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.82rem;
        color: #0f172a;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.18);
        white-space: nowrap;
    }
    .chat-launcher-hint::after {
        content: "";
        position: absolute;
        right: -7px;
        top: 50%;
        transform: translateY(-50%);
        border-left: 8px solid #ffffff;
        border-top: 6px solid transparent;
        border-bottom: 6px solid transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("Executive Turnover Dashboard")

# ---------------------------------------------------------------------------
# Dynamic Dataset Selection & Caching
# ---------------------------------------------------------------------------

st.sidebar.title("??? Configuration")
dataset_specs, skipped_specs = discover_dataset_specs(include_root_excels=False)
spec_by_tag = {spec.tag: spec for spec in dataset_specs}
legacy_datasets = [
    spec.tag
    for spec in dataset_specs
    if os.path.exists(f"output/predictions_{spec.tag}.xlsx")
]

FINAL_DATASET_LABEL = "Final model (file3 external test)"
dataset_options = []
if FINAL_ARTIFACT_PATH.exists():
    dataset_options.append(FINAL_DATASET_LABEL)
dataset_options.extend(legacy_datasets)

if not dataset_options:
    st.error("No trained predictions found. Run `python scripts/train_final_models.py` for the final model or `python main.py --all` for legacy datasets.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
is_final_dataset = selected_dataset == FINAL_DATASET_LABEL

if is_final_dataset:
    DATA_PATH = "output/final/file3_predictions.xlsx"
    RAW_DATA_PATH = "prepared final file3 rows"
    API_PATH = str(FINAL_ARTIFACT_PATH)
else:
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
        st.error(f"Predictions file **{DATA_PATH}** not found. Please run `python main.py --dataset {selected_dataset}` first.")
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

    st.markdown('<div class="chat-launcher-hint">What do you want to know?</div>', unsafe_allow_html=True)

    with st.popover("💬 ", help="Open HR Assistant"):
        st.markdown("#### HR Assistant")

        selected_dept_option = st.selectbox(
            "Department Scope",
            options=valid_dept_options,
            key="chat_filter_dept",
        )
        selected_dept = None if selected_dept_option == dept_default else selected_dept_option

        scoped_df = dashboard_df
        if selected_dept:
            scoped_df = dashboard_df[dashboard_df["Budget Section"].astype(str) == selected_dept]

        employee_options = sorted(scoped_df["Employee ID"].astype(str).unique().tolist())
        valid_emp_options = [emp_default] + employee_options

        if "chat_filter_emp" not in st.session_state:
            st.session_state["chat_filter_emp"] = emp_default
        if st.session_state.get("chat_filter_emp") not in valid_emp_options:
            st.session_state["chat_filter_emp"] = emp_default

        selected_emp_option = st.selectbox(
            "Employee Scope",
            options=valid_emp_options,
            key="chat_filter_emp",
        )
        selected_emp = None if selected_emp_option == emp_default else selected_emp_option

        context_bits = []
        if selected_dept:
            context_bits.append(f"Department: **{selected_dept}**")
        if selected_emp:
            context_bits.append(f"Employee: **{selected_emp}**")
        if context_bits:
            st.caption(" | ".join(context_bits))
        else:
            st.caption("Query scope: company-wide")

        st.markdown("**Example Questions**")
        examples = [
            "Which department has the highest average risk?",
            "Who are the 5 highest-risk employees?",
            "How does risk vary by job rank?",
            "Show employees with risk above 70%.",
            "Give me the overall risk statistics.",
        ]

        selected_example = None
        for i, ex in enumerate(examples):
            if st.button(ex, use_container_width=True, key=f"floating_ex_{i}"):
                selected_example = ex

        st.markdown("---")

        for msg in st.session_state.chat_messages[-4:]:
            role_label = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role_label}:** {msg['content']}")

        st.markdown("---")
        with st.form("floating_chat_form", clear_on_submit=True):
            typed_prompt = st.text_input(
                "Ask about turnover risk",
                placeholder="Type your question...",
                label_visibility="collapsed",
            )
            send_clicked = st.form_submit_button("Send", use_container_width=True)

        if selected_example:
            process_chat_prompt(selected_example, dashboard_df, raw_df, api, selected_emp, selected_dept)
            st.rerun()

        if send_clicked and typed_prompt:
            process_chat_prompt(typed_prompt, dashboard_df, raw_df, api, selected_emp, selected_dept)
            st.rerun()

        if st.button("Clear Chat", use_container_width=True, key="floating_clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------------------------
tab_macro, tab_meso, tab_micro = st.tabs([
    "Macro View (Company)",
    "Meso View (Team/Department)",
    "Micro View (Individual Employee)",
])

# ===========================================================================
# MACRO VIEW
# ===========================================================================
with tab_macro:
    st.markdown("Monitor company-wide turnover risk and evaluate retention strategies at a glance.")
    
    st.markdown("### Top Indicators")
    col1, col2, col3 = st.columns(3)
    
    total_employees = len(dashboard_df)
    avg_turnover_prob = dashboard_df["Turnover Probability"].mean()
    high_risk_count = int(
        dashboard_df[
            dashboard_df["Risk Category"].isin(["High Risk", "Very High Risk"])
        ].shape[0]
    )
    
    with col1:
        st.metric("Total Active Employees", f"{total_employees:,}")
    with col2:
        st.metric("Company-Wide Turnover Risk", f"{avg_turnover_prob * 100:.1f}%")
    with col3:
        st.metric("High-Risk Headcount", f"{high_risk_count:,}")
        
    st.markdown("---")
    st.markdown("### Risk Analysis Insights")
    
    left_col, right_col = st.columns(2)
    
    # Donut Chart
    with left_col:
        st.markdown("#### Risk Distribution")
        risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
        color_map = {
            "Low Risk": "#2ecc71",
            "Medium Risk": "#f1c40f",
            "High Risk": "#e67e22",
            "Very High Risk": "#e74c3c",
        }
        
        risk_counts = (
            dashboard_df["Risk Category"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )
        risk_counts.columns = ["Risk Category", "Count"]
        
        fig_donut = go.Figure(data=[
            go.Pie(
                labels=risk_counts["Risk Category"],
                values=risk_counts["Count"],
                hole=0.55,
                marker=dict(colors=[color_map.get(r, "#000") for r in risk_counts["Risk Category"]]),
                textinfo="percent+label",
                textposition="auto",
            )
        ])
        fig_donut.update_layout(dragmode=False, showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig_donut, width="stretch", key="donut_chart", config={"displayModeBar": False})
        
    # Bar Chart
    with right_col:
        st.markdown("#### Avg Turnover Risk by Budget Section")
        dept_risk = dashboard_df.groupby("Budget Section")["Turnover Probability"].mean().reset_index()
        dept_risk["Turnover Probability (%)"] = dept_risk["Turnover Probability"] * 100
        dept_risk = dept_risk.sort_values("Turnover Probability (%)", ascending=False).head(15)
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=dept_risk["Budget Section"].astype(str),
                y=dept_risk["Turnover Probability (%)"],
                marker_color="#e74c3c",
            )
        ])
        fig_bar.update_layout(dragmode=False, xaxis_title="Budget Section", yaxis_title="Avg Turnover Risk (%)", xaxis_type="category", margin=dict(t=20, b=20, l=40, r=20), height=400)
        st.plotly_chart(fig_bar, width="stretch", key="bar_chart", config={"displayModeBar": False})
        
    st.markdown("---")
    st.markdown("#### Turnover Probability vs. Tenure")
    
    fig_scatter = go.Figure()
    for cat, color in color_map.items():
        subset = dashboard_df[dashboard_df["Risk Category"] == cat]
        if subset.empty:
            continue
        fig_scatter.add_trace(
            go.Scatter(
                x=subset["Tenure (Months)"],
                y=subset["Turnover Probability"],
                mode="markers",
                name=cat,
                marker=dict(color=color, opacity=0.35, size=5, line=dict(width=0.5, color='white')),
                text=subset["Employee ID"].astype(str),
                hovertemplate="Employee: %{text}<br>Tenure: %{x} months<br>Risk: %{y:.2%}<extra></extra>",
            )
        )
        
    fig_scatter.update_layout(dragmode=False, xaxis_title="Tenure (Months)", yaxis_title="Turnover Probability", legend_title="Risk Category", margin=dict(t=20, b=40, l=40, r=20), height=450)
    st.plotly_chart(fig_scatter, width="stretch", key="scatter_chart", config={"displayModeBar": False})

# ===========================================================================
# MESO VIEW (Department Level)
# ===========================================================================
with tab_meso:
    st.markdown("Analyze turnover risk across specific teams or departments to identify localized retention challenges.")
    
    if dashboard_df.empty:
        st.warning(f"Data not properly loaded. Missing `{DATA_PATH}`.")
    else:
        # Get unique budget sections (departments)
        departments = sorted([str(x) for x in dashboard_df["Budget Section"].dropna().unique() if str(x).strip() != ""])
        
        selected_dept = st.selectbox("Select Team / Budget Section", options=departments, index=0)
        st.session_state["chat_selected_dept"] = selected_dept

        if selected_dept:
            dept_df = dashboard_df[dashboard_df["Budget Section"].astype(str) == selected_dept]
            
            st.markdown(f"### Team Overview: {selected_dept}")
            
            # --- KPI Row ---
            col1, col2, col3 = st.columns(3)
            
            dept_total = len(dept_df)
            dept_avg_prob = dept_df["Turnover Probability"].mean()
            dept_high_risk = int(dept_df[dept_df["Risk Category"].isin(["High Risk", "Very High Risk"])].shape[0])
            
            with col1:
                st.metric("Team Members", f"{dept_total:,}")
            with col2:
                # Add a delta comparing this team to the company average
                overall_avg = dashboard_df["Turnover Probability"].mean()
                delta_prob = (dept_avg_prob - overall_avg) * 100
                st.metric("Avg Team Risk", f"{dept_avg_prob * 100:.1f}%", f"{delta_prob:+.1f}% vs Company", delta_color="inverse")
            with col3:
                st.metric("High-Risk Members", f"{dept_high_risk:,}")
                
            st.markdown("---")
            
            # --- Visualizations ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### Team Risk Distribution")
                risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
                color_map = {
                    "Low Risk": "#2ecc71",
                    "Medium Risk": "#f1c40f",
                    "High Risk": "#e67e22",
                    "Very High Risk": "#e74c3c",
                }
                
                dept_risk_counts = (
                    dept_df["Risk Category"]
                    .value_counts()
                    .reindex(risk_order, fill_value=0)
                    .reset_index()
                )
                dept_risk_counts.columns = ["Risk Category", "Count"]
                
                fig_dept_donut = go.Figure(data=[
                    go.Pie(
                        labels=dept_risk_counts["Risk Category"],
                        values=dept_risk_counts["Count"],
                        hole=0.55,
                        marker=dict(colors=[color_map.get(r, "#000") for r in dept_risk_counts["Risk Category"]]),
                        textinfo="value+label",
                        textposition="auto",
                    )
                ])
                fig_dept_donut.update_layout(dragmode=False, showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_dept_donut, width="stretch", key="dept_donut", config={"displayModeBar": False})
                
            with col_chart2:
                st.markdown("#### Average Risk by Job Rank (Maamad)")
                
                if "Job Rank" in dept_df.columns or "Maamad" in dept_df.columns:
                    rank_col = "Job Rank" if "Job Rank" in dept_df.columns else "Maamad"
                    
                    rank_risk = dept_df.groupby(rank_col)["Turnover Probability"].mean().reset_index()
                    rank_risk["Turnover Probability (%)"] = rank_risk["Turnover Probability"] * 100
                    rank_risk = rank_risk.sort_values("Turnover Probability (%)", ascending=False)
                    
                    fig_dept_bar = go.Figure(data=[
                        go.Bar(
                            x=rank_risk[rank_col].astype(str),
                            y=rank_risk["Turnover Probability (%)"],
                            marker_color="#3498db",
                        )
                    ])
                    fig_dept_bar.update_layout(dragmode=False, xaxis_title="Job Rank", yaxis_title="Avg Turnover Risk (%)", xaxis_type="category", margin=dict(t=20, b=20, l=40, r=20), height=350)
                    st.plotly_chart(fig_dept_bar, width="stretch", key="dept_bar", config={"displayModeBar": False})
                else:
                    st.info("Job Rank dimension not available in the output data.")
                    
            st.markdown("---")
            st.markdown(f"#### Act Now: Highest Risk Employees in {selected_dept}")
            
            # Show top 10 highest risk employees in this department
            top_risk_df = dept_df.sort_values(by="Turnover Probability", ascending=False).head(10)
            
            display_cols = ["Employee ID", "Risk Category", "Turnover Probability"]
            # Add secondary informative columns if they exist
            for c in ["Job Rank", "Maamad", "Role Code", "tafkidCode", "Tenure (Months)"]:
                if c in top_risk_df.columns:
                    display_cols.append(c)
            
            styled_df = top_risk_df[display_cols].copy()
            styled_df["Turnover Probability"] = styled_df["Turnover Probability"].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)


# ===========================================================================
# MICRO VIEW
# ===========================================================================
with tab_micro:
    st.markdown("Select an employee to view their current profile and simulate how changes to key factors affect their turnover risk.")
    
    if raw_df.empty or api is None:
        st.warning(f"Raw data or model not properly loaded for `{selected_dataset}`. Missing `{RAW_DATA_PATH}` or `{API_PATH}`. Please run `python main.py` first.")
    else:
        if is_final_dataset:
            employee_id_col = getattr(api, 'employee_id_col', 'fictive_employee')
            time_col = getattr(api, 'time_col', 'calc_month')
        else:
            dataset_config = api.pipeline.get('dataset_config', {})
            employee_id_col = dataset_config.get('employee_id_col', 'fictive2')
            time_col = dataset_config.get('time_col')
        
        valid_employees = sorted(dashboard_df["Employee ID"].unique())
        
        selected_emp_id = st.selectbox("Search / Select Employee ID", options=[""] + valid_employees)
        st.session_state["chat_selected_emp_id"] = selected_emp_id or None

        if selected_emp_id:
            # 1. Employee Data Setup
            emp_records = raw_df[raw_df[employee_id_col] == selected_emp_id].copy()
            if time_col and time_col in emp_records.columns:
                emp_records = emp_records.sort_values(time_col)
                
            latest_record = emp_records.iloc[-1].copy()
            
            # Compute current_prob organically via the API so it perfectly aligns with the What-If simulation engine baselines.
            try:
                current_prob, current_category = api.predict_risk(emp_records)
            except Exception as e:
                st.error(f"Error predicting current risk: {e}")
                # Fallback to the dashboard precomputed batch if direct API fails
                dashboard_match = dashboard_df[dashboard_df["Employee ID"] == selected_emp_id]
                if not dashboard_match.empty:
                    current_prob = dashboard_match.iloc[0]["Turnover Probability"]
                    current_category = dashboard_match.iloc[0]["Risk Category"]
                else:
                    current_prob, current_category = 0.0, "Unknown"
                
            # 2. Lever State Initialization
            session_key = f"levers_{selected_emp_id}"
            
            def safe_float(val, default=0.0):
                try:
                    return float(val) if pd.notna(val) else default
                except (ValueError, TypeError):
                    return default
            
            def safe_val(val, default=0.0):
                return val if pd.notna(val) else default
                
            def safe_str(val, default=""):
                return str(val) if pd.notna(val) else default
            
            # Grab latest baseline directly from the record to feed the initial widget states
            # Grab latest baseline directly from the record to feed the initial widget states
            base_salary = safe_float(latest_record.get('avg_Payment', 4000.0))
            base_workload = safe_float(latest_record.get('avg_omes', 0.0))
            base_illness = safe_float(latest_record.get('avg_illness', 0.0))
            base_maamad = safe_val(latest_record.get('Maamad', np.nan))
            base_seif = safe_val(latest_record.get('Seif', np.nan))
            base_contract = safe_val(latest_record.get('contract_type', np.nan))
            
            # Initialize Session State ONLY if it doesn't exist
            if f"sl_salary_{selected_emp_id}" not in st.session_state:
                st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
                st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
                st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
                st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
                st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
                st.session_state[f"sel_contract_{selected_emp_id}"] = base_contract
            
            # Reset Callback Logic
            def reset_levers():
                st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
                st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
                st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
                st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
                st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
                st.session_state[f"sel_contract_{selected_emp_id}"] = base_contract
                
            col_reset1, col_reset2 = st.columns([8, 2])
            with col_reset2:
                st.button("🔄 Reset to Original Levers", on_click=reset_levers)
            
            st.markdown("---")
            
            # 3. Two-Column UI (Profile vs Levers)
            col_prof, col_levers = st.columns([1, 1], gap="large")
            
            with col_prof:
                st.markdown("### Current Profile Analysis")
                
                fig_gauge_curr = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_prob * 100,
                    number={'suffix': "%", 'valueformat': ".1f"},
                    title={'text': "Current Risk", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 30], 'color': "#2ecc71"},
                            {'range': [30, 50], 'color': "#f1c40f"},
                            {'range': [50, 70], 'color': "#e67e22"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': current_prob * 100
                        }
                    }
                ))
                fig_gauge_curr.update_layout(dragmode=False, height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge_curr, width="stretch", config={"displayModeBar": False})
                
                # Format Snapshot neatly
                tenure = safe_float(latest_record.get('vetek_months', 0))
                age = safe_float(latest_record.get('age', 0))
                gender = safe_str(latest_record.get('gender', 'Unknown'))
                city = safe_str(latest_record.get('Yishuv', 'Unknown'))
                role = safe_str(latest_record.get('tafkidCode', 'Unknown'))
                
                st.info(f"""
                **Employee Snapshot**
                - **Tenure:** {tenure:.0f} months
                - **Age:** {age:.0f} years
                - **Gender:** {gender}
                - **City:** {city}
                - **Role:** {role}
                """)
                
            with col_levers:
                st.markdown("### What-If Scenario Levers")
                
                # Handle edge cases for max/min
                max_salary = max(40000.0, float(raw_df['avg_Payment'].max()) if pd.api.types.is_numeric_dtype(raw_df['avg_Payment']) else 40000.0)
                min_salary = min(4000.0, float(raw_df['avg_Payment'].min()) if pd.api.types.is_numeric_dtype(raw_df['avg_Payment']) else 4000.0)
                if base_salary < min_salary: base_salary = min_salary
                if base_salary > max_salary: base_salary = max_salary
                
                new_salary = st.slider("Average Salary (NIS)", 
                                      min_value=min_salary, max_value=max_salary, 
                                      step=500.0, key=f"sl_salary_{selected_emp_id}")
                
                max_workload = max(200.0, float(raw_df['avg_omes'].max()) if pd.api.types.is_numeric_dtype(raw_df['avg_omes']) else 200.0)
                if base_workload < 0.0: base_workload = 0.0
                if base_workload > max_workload: base_workload = max_workload
                new_workload = st.slider("Average Workload", 
                                        min_value=0.0, max_value=max_workload, 
                                        step=1.0, key=f"sl_workload_{selected_emp_id}")
                                        
                max_illness = max(50.0, float(raw_df['avg_illness'].max()) if pd.api.types.is_numeric_dtype(raw_df['avg_illness']) else 50.0)
                if base_illness < 0.0: base_illness = 0.0
                if base_illness > max_illness: base_illness = max_illness
                new_illness = st.slider("Average Sick Days", 
                                       min_value=0.0, max_value=max_illness, 
                                       step=1.0, key=f"sl_ill_{selected_emp_id}")
                
                if is_final_dataset:
                    contract_opts = available_options(raw_df, 'contract_type', base_contract)
                    new_contract = st.selectbox("Contract Type", options=contract_opts, key=f"sel_contract_{selected_emp_id}") if contract_opts else base_contract
                else:
                    new_contract = base_contract

                maamad_opts = available_options(raw_df, 'Maamad', base_maamad)
                new_maamad = st.selectbox("Job Rank (Maamad)", options=maamad_opts, key=f"sel_maamad_{selected_emp_id}") if maamad_opts else base_maamad
                                         
                seif_opts = available_options(raw_df, 'Seif', base_seif)
                new_seif = st.selectbox("Budget Section (Seif)", options=seif_opts, key=f"sel_seif_{selected_emp_id}") if seif_opts else base_seif
                                       
            if is_final_dataset:
                mod_records = apply_final_what_if(
                    emp_records,
                    salary=new_salary,
                    workload=new_workload,
                    illness=new_illness,
                    contract_type=new_contract,
                    maamad=new_maamad,
                    seif=new_seif,
                )
            else:
                mod_records = emp_records.copy()
                latest_idx = mod_records.index[-1]
                mod_records.at[latest_idx, 'avg_Payment'] = new_salary
                mod_records.at[latest_idx, 'avg_omes'] = new_workload
                mod_records.at[latest_idx, 'avg_illness'] = new_illness
                mod_records.at[latest_idx, 'Maamad'] = new_maamad
                mod_records.at[latest_idx, 'Seif'] = new_seif
            
            try:
                new_prob, new_category = api.predict_risk(mod_records)
            except Exception as e:
                st.error(f"Inference error: {e}")
                new_prob = current_prob
                
            st.markdown("---")
            st.markdown("### Simulated Risk Outlook")
            
            col_sim1, col_sim2 = st.columns([1, 1], gap="large")
            
            with col_sim1:
                fig_gauge_new = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=new_prob * 100,
                    number={'suffix': "%", 'valueformat': ".1f"},
                    delta={
                        'reference': current_prob * 100, 
                        'position': "bottom", 
                        'valueformat': ".1f", 
                        'suffix': "%",
                        'increasing': {'color': "#e74c3c"}, 
                        'decreasing': {'color': "#2ecc71"}  
                    },
                    title={'text': "New Simulated Risk", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 30], 'color': "#2ecc71"},
                            {'range': [30, 50], 'color': "#f1c40f"},
                            {'range': [50, 70], 'color': "#e67e22"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': new_prob * 100
                        }
                    }
                ))
                fig_gauge_new.update_layout(dragmode=False, height=280, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge_new, width="stretch", config={"displayModeBar": False})
                
            with col_sim2:
                st.markdown("#### Explainability: Lever Adjustments Impact")
                
                orig_salary = safe_float(latest_record.get('avg_Payment', 0))
                orig_workload = safe_float(latest_record.get('avg_omes', 0))
                orig_illness = safe_float(latest_record.get('avg_illness', 0))

                if is_final_dataset:
                    lever_specs = [
                        ("Salary", dict(salary=new_salary, workload=orig_workload, illness=orig_illness, contract_type=base_contract, maamad=base_maamad, seif=base_seif)),
                        ("Workload", dict(salary=orig_salary, workload=new_workload, illness=orig_illness, contract_type=base_contract, maamad=base_maamad, seif=base_seif)),
                        ("Sick Days", dict(salary=orig_salary, workload=orig_workload, illness=new_illness, contract_type=base_contract, maamad=base_maamad, seif=base_seif)),
                        ("Contract", dict(salary=orig_salary, workload=orig_workload, illness=orig_illness, contract_type=new_contract, maamad=base_maamad, seif=base_seif)),
                        ("Job Rank", dict(salary=orig_salary, workload=orig_workload, illness=orig_illness, contract_type=base_contract, maamad=new_maamad, seif=base_seif)),
                        ("Budget Section", dict(salary=orig_salary, workload=orig_workload, illness=orig_illness, contract_type=base_contract, maamad=base_maamad, seif=new_seif)),
                    ]
                    impact_rows = []
                    for label, kwargs in lever_specs:
                        try:
                            one_change_records = apply_final_what_if(emp_records, **kwargs)
                            one_prob, _ = api.predict_risk(one_change_records)
                            impact_rows.append({"Feature": label, "Risk Delta (pp)": (one_prob - current_prob) * 100})
                        except Exception:
                            impact_rows.append({"Feature": label, "Risk Delta (pp)": 0.0})

                    df_diffs = pd.DataFrame(impact_rows)
                    df_diffs = df_diffs.loc[df_diffs["Risk Delta (pp)"].abs().sort_values(ascending=False).index]
                    colors = ["#e74c3c" if val > 0 else "#2ecc71" if val < 0 else "lightgray" for val in df_diffs["Risk Delta (pp)"]]
                    fig_diff = go.Figure(go.Bar(
                        x=df_diffs["Risk Delta (pp)"],
                        y=df_diffs["Feature"],
                        orientation='h',
                        marker_color=colors,
                        text=df_diffs["Risk Delta (pp)"].apply(lambda x: f"+{x:.2f} pp" if x > 0 else f"{x:.2f} pp"),
                        textposition="auto"
                    ))
                    fig_diff.update_layout(dragmode=False,
                        xaxis_title="Estimated risk change from each lever",
                        yaxis_title="",
                        height=260,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                else:
                    diffs = {
                        "Salary": ((new_salary - orig_salary) / max(1, orig_salary)) * 100,
                        "Workload": ((new_workload - orig_workload) / max(1, orig_workload)) * 100,
                        "Sick Days": ((new_illness - orig_illness) / max(1, orig_illness)) * 100,
                    }
                    
                    df_diffs = pd.DataFrame(list(diffs.items()), columns=["Feature", "% Change"])
                    
                    colors = []
                    for idx, row in df_diffs.iterrows():
                        val = row["% Change"]
                        feat = row["Feature"]
                        if val == 0:
                            colors.append("lightgray")
                        elif feat in ["Workload", "Sick Days"]:
                            colors.append("#e74c3c" if val > 0 else "#2ecc71")
                        else: 
                            colors.append("#2ecc71" if val > 0 else "#e74c3c")
                    
                    fig_diff = go.Figure(go.Bar(
                        x=df_diffs["% Change"],
                        y=df_diffs["Feature"],
                        orientation='h',
                        marker_color=colors,
                        text=df_diffs["% Change"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"),
                        textposition="auto"
                    ))
                    fig_diff.update_layout(dragmode=False, 
                        xaxis_title="Simulation % Change vs Original",
                        yaxis_title="",
                        height=200,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
                    )
                st.plotly_chart(fig_diff, width="stretch", config={"displayModeBar": False})

render_floating_chat_widget(dashboard_df, raw_df, api)
