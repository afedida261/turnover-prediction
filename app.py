import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path for internal imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.inference import TurnoverInferenceAPI
    from src.config import FEATURE_DESCRIPTIONS
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
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Executive Turnover Dashboard")

# ---------------------------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------------------------
DATA_PATH = "predictions_output.xlsx"
RAW_DATA_PATH = "first_file.xlsx"

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

# ---------------------------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------------------------
tab_macro, tab_meso, tab_micro = st.tabs(["Macro View (Company)", "Meso View (Team/Department)", "Micro View (Individual Employee)"])

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
        st.warning("Data not properly loaded. Missing `predictions_output.xlsx`.")
    else:
        # Get unique budget sections (departments)
        departments = sorted([str(x) for x in dashboard_df["Budget Section"].dropna().unique() if str(x).strip() != ""])
        
        selected_dept = st.selectbox("Select Team / Budget Section", options=departments, index=0)
        
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
        st.warning("Raw data or model not properly loaded. Missing `first_file.xlsx` or `artifacts/model_pipeline.pkl`. Please run `python main.py` first.")
    else:
        employee_id_col = 'fictive2'
        time_col = 'fictive-ovedmiun'
        
        valid_employees = sorted(dashboard_df["Employee ID"].unique())
        
        selected_emp_id = st.selectbox("Search / Select Employee ID", options=[""] + valid_employees)
        
        if selected_emp_id:
            # 1. Employee Data Setup
            emp_records = raw_df[raw_df[employee_id_col] == selected_emp_id].copy()
            if time_col in emp_records.columns:
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
            
            # Initialize Session State ONLY if it doesn't exist
            if f"sl_salary_{selected_emp_id}" not in st.session_state:
                st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
                st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
                st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
                st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
                st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
            
            # Reset Callback Logic
            def reset_levers():
                st.session_state[f"sl_salary_{selected_emp_id}"] = base_salary
                st.session_state[f"sl_workload_{selected_emp_id}"] = base_workload
                st.session_state[f"sl_ill_{selected_emp_id}"] = base_illness
                st.session_state[f"sel_maamad_{selected_emp_id}"] = base_maamad
                st.session_state[f"sel_seif_{selected_emp_id}"] = base_seif
                
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
                
                maamad_opts = raw_df['Maamad'].dropna().unique().tolist()
                if base_maamad not in maamad_opts and pd.notna(base_maamad): maamad_opts.append(base_maamad)
                maamad_opts = sorted(list(set(maamad_opts)))
                new_maamad = st.selectbox("Job Rank (Maamad)", options=maamad_opts, key=f"sel_maamad_{selected_emp_id}")
                                         
                seif_opts = raw_df['Seif'].dropna().unique().tolist()
                if base_seif not in seif_opts and pd.notna(base_seif): seif_opts.append(base_seif)
                seif_opts = sorted(list(set(seif_opts)))
                new_seif = st.selectbox("Budget Section (Seif)", options=seif_opts, key=f"sel_seif_{selected_emp_id}")
                                       
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
