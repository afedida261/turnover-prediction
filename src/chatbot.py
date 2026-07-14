from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv, dotenv_values, find_dotenv
from google import genai
from google.genai import types

from src.analysis.local_explain import (
    explain_employee_prediction,
    global_feature_importance,
)

load_dotenv()


def _resolve_provider_keys() -> tuple[Optional[str], Optional[str]]:
    """Resolve the Gemini and OpenAI keys, treating the .env FILE as the source
    of truth. If a .env file exists, only keys actually present (uncommented) in
    it are used — so commenting out GEMINI_API_KEY reliably switches to OpenAI,
    even if a stale key lingers in the OS environment. If no .env file is found,
    fall back to the process environment (useful for deployments)."""
    file_env: dict[str, str] | None = None
    try:
        path = find_dotenv(usecwd=True)
        if path:
            file_env = dotenv_values(path)
    except Exception:
        file_env = None

    if file_env is not None:
        gemini_key = (
            file_env.get("GEMINI_API_KEY")
            or file_env.get("gimini_api_key")
            or file_env.get("GIMINI_API_KEY")
        )
        openai_key = (
            file_env.get("OPENAI_API_KEY")
            or file_env.get("OPEN_AI_API_KEY")
            or file_env.get("OPEN_AI_KEY")
            or file_env.get("OPENAI_KEY")
        )
    else:
        gemini_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("gimini_api_key")
            or os.getenv("GIMINI_API_KEY")
        )
        openai_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY")
            or os.getenv("OPEN_AI_KEY")
            or os.getenv("OPENAI_KEY")
        )

    return (gemini_key or None), (openai_key or None)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an HR analytics assistant embedded in an Executive Turnover Dashboard.
You help HR managers and executives understand employee turnover risk across the organization.

## Data Context
- You have access to batch predictions for all employees: turnover probability (0–1), risk category, department, tenure, and more.
- Risk tiers: Low Risk (≤30%), Medium Risk (31–50%), High Risk (51–70%), Very High Risk (>70%)
- You can look up live risk scores, run what-if simulations, and explain WHY a score is high or low.

## Your Role
- Answer questions about company-wide, department-level, or individual employee turnover risk.
- Explain the REASONS behind a prediction using SHAP feature importance — call `explain_employee` for a specific
  person, or `top_risk_drivers` for the organisation-wide drivers.
- Surface trends and correlations (call `feature_correlations`) to explain what tends to move risk up or down.
- Identify at-risk employees and suggest concrete, actionable retention strategies grounded in the drivers you find.
- Simulate how changes to salary, workload, or sick days affect an individual's risk (call `what_if`).
- Act as a data analyst: rank departments, segment risk by any column, compute statistics, and filter employees.
- For anything not covered by a dedicated tool, use `run_sql` to query the predictions table directly.

## Tool Strategy
- Prefer the dedicated tools (explain_employee, top_risk_drivers, feature_correlations, what_if, rank_departments,
  risk_by_group, company_stats, find_employees, department_summary, high_risk_employees, employee_risk).
- Use `run_sql` only for bespoke slicing the dedicated tools don't cover.
- When a user asks "why" a score is high/low, ALWAYS use explain_employee (individual) or top_risk_drivers (global)
  rather than guessing.
- You may call multiple tools before answering. Combine their results into one clear response.

## Guidelines
- Always ground answers in tool output — never fabricate employee IDs, probabilities, headcounts, or drivers.
- Be concise and actionable. Lead with the answer, then 1–3 supporting points.
- If the user is currently viewing a specific employee or department (given in context), prioritise it.
- Never search the web or use outside knowledge. If a tool can't answer, say so plainly.
"""

# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_high_risk_employees",
            "description": (
                "Returns the top-N highest turnover-risk employees, optionally filtered by department. "
                "Use this when the user asks who is most at risk, or wants a ranked list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Filter by Budget Section / department name. Omit to search company-wide."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of employees to return (default 5, max 20).",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_risk",
            "description": (
                "Returns the current risk score and category for a specific employee by ID. "
                "Use this when the user asks about a particular employee."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "The Employee ID to look up."
                    }
                },
                "required": ["employee_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_summary",
            "description": (
                "Returns a summary of turnover risk statistics for a given department: "
                "headcount, average risk, high-risk count, and risk distribution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "The Budget Section / department name."
                    }
                },
                "required": ["department"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_what_if",
            "description": (
                "Simulates how changing salary, workload, or sick days for a specific employee "
                "would affect their turnover risk. Returns both current and simulated probabilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "The Employee ID to simulate."
                    },
                    "salary_delta_pct": {
                        "type": "number",
                        "description": "Percentage change to apply to average salary (e.g. 10 for +10%, -5 for -5%). Default 0."
                    },
                    "workload_delta_pct": {
                        "type": "number",
                        "description": "Percentage change to apply to average workload. Default 0."
                    },
                    "illness_delta_pct": {
                        "type": "number",
                        "description": "Percentage change to apply to average sick days. Default 0."
                    }
                },
                "required": ["employee_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_departments_by_risk",
            "description": (
                "Ranks all departments by average turnover risk. Use this when the user asks which "
                "department is riskiest, safest, or wants a league-table of departments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many departments to return (default 10).",
                        "default": 10
                    },
                    "order": {
                        "type": "string",
                        "enum": ["desc", "asc"],
                        "description": "Sort by highest risk first ('desc') or lowest risk first ('asc'). Default 'desc'."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_risk_by_group",
            "description": (
                "Groups employees by any available column and computes avg risk per group. "
                "Use this to answer questions like 'which job rank is riskiest?' or 'how does risk vary by city?'. "
                "Available group columns include: 'Job Rank', 'Budget Section', 'Tenure (Months)' (binned), "
                "and any other column present in the predictions data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_column": {
                        "type": "string",
                        "description": "Column name to group by. Call list_available_columns first if unsure."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of groups to return, sorted by avg risk descending (default 10).",
                        "default": 10
                    }
                },
                "required": ["group_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_risk_stats",
            "description": (
                "Returns company-wide turnover risk statistics: mean, median, std deviation, "
                "percentile breakdown (25/50/75/90/95), and count per risk tier. "
                "Use this for broad analytical questions about the overall risk landscape."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_employees_by_criteria",
            "description": (
                "Filters employees by multiple criteria and returns a ranked list. "
                "Use this for questions like 'show me high-risk employees with long tenure' "
                "or 'find employees in dept X with risk above 60%'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_risk_pct": {
                        "type": "number",
                        "description": "Minimum turnover probability % (e.g. 60 means ≥60%). Default 0."
                    },
                    "max_risk_pct": {
                        "type": "number",
                        "description": "Maximum turnover probability % (e.g. 80 means ≤80%). Default 100."
                    },
                    "department": {
                        "type": "string",
                        "description": "Filter by department name. Omit for company-wide."
                    },
                    "min_tenure_months": {
                        "type": "number",
                        "description": "Minimum tenure in months. Omit to skip filter."
                    },
                    "max_tenure_months": {
                        "type": "number",
                        "description": "Maximum tenure in months. Omit to skip filter."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max number of employees to return (default 10, max 30).",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_columns",
            "description": (
                "Returns the list of column names available in the predictions dataset. "
                "Call this first when you need to know valid column names for analyze_risk_by_group."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def list_high_risk_employees(dashboard_df: pd.DataFrame, department: Optional[str] = None, top_n: int = 5) -> str:
    top_n = min(int(top_n), 20)
    df = dashboard_df.copy()
    if department:
        df = df[df["Budget Section"].astype(str).str.lower() == department.lower()]
        if df.empty:
            return f"No employees found in department '{department}'."

    top = df.sort_values("Turnover Probability", ascending=False).head(top_n)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            f"  Employee {r['Employee ID']} — {r['Turnover Probability']*100:.1f}% ({r['Risk Category']})"
            + (f", Dept: {r['Budget Section']}" if not department else "")
        )
    scope = f"department '{department}'" if department else "company-wide"
    return f"Top {top_n} highest-risk employees ({scope}):\n" + "\n".join(rows)


def get_employee_risk(dashboard_df: pd.DataFrame, employee_id: str) -> str:
    match = dashboard_df[dashboard_df["Employee ID"].astype(str) == str(employee_id)]
    if match.empty:
        return f"No employee found with ID '{employee_id}'."
    r = match.iloc[0]
    tenure = f"{r['Tenure (Months)']:.0f} months" if "Tenure (Months)" in r and pd.notna(r.get("Tenure (Months)")) else "N/A"
    dept = r.get("Budget Section", "N/A")
    return (
        f"Employee {employee_id}:\n"
        f"  Risk: {r['Turnover Probability']*100:.1f}% — {r['Risk Category']}\n"
        f"  Department: {dept}\n"
        f"  Tenure: {tenure}"
    )


def get_department_summary(dashboard_df: pd.DataFrame, department: str) -> str:
    df = dashboard_df[dashboard_df["Budget Section"].astype(str).str.lower() == department.lower()]
    if df.empty:
        available = sorted(dashboard_df["Budget Section"].dropna().unique().tolist())
        return f"Department '{department}' not found. Available departments: {', '.join(str(d) for d in available[:10])}..."

    total = len(df)
    avg_risk = df["Turnover Probability"].mean() * 100
    high_risk = int(df["Risk Category"].isin(["High Risk", "Very High Risk"]).sum())
    dist = df["Risk Category"].value_counts().to_dict()
    dist_str = ", ".join(f"{k}: {v}" for k, v in dist.items())
    company_avg = dashboard_df["Turnover Probability"].mean() * 100
    delta = avg_risk - company_avg

    return (
        f"Department '{department}':\n"
        f"  Headcount: {total}\n"
        f"  Avg Risk: {avg_risk:.1f}% ({delta:+.1f}% vs company avg of {company_avg:.1f}%)\n"
        f"  High-Risk Members: {high_risk}\n"
        f"  Risk Distribution: {dist_str}"
    )


def rank_departments_by_risk(dashboard_df: pd.DataFrame, top_n: int = 10, order: str = "desc") -> str:
    top_n = min(int(top_n), 30)
    ascending = order == "asc"
    company_avg = dashboard_df["Turnover Probability"].mean() * 100

    grouped = (
        dashboard_df.groupby("Budget Section")
        .agg(
            headcount=("Turnover Probability", "count"),
            avg_risk=("Turnover Probability", "mean"),
            high_risk_count=("Risk Category", lambda x: x.isin(["High Risk", "Very High Risk"]).sum()),
        )
        .reset_index()
    )
    grouped["avg_risk_pct"] = grouped["avg_risk"] * 100
    grouped = grouped.sort_values("avg_risk_pct", ascending=ascending).head(top_n)

    label = "lowest" if ascending else "highest"
    rows = []
    for i, (_, r) in enumerate(grouped.iterrows(), 1):
        delta = r["avg_risk_pct"] - company_avg
        rows.append(
            f"  {i}. {r['Budget Section']} — {r['avg_risk_pct']:.1f}% avg risk ({delta:+.1f}% vs company), "
            f"{int(r['headcount'])} employees, {int(r['high_risk_count'])} high-risk"
        )
    return f"Departments ranked by {label} turnover risk (company avg: {company_avg:.1f}%):\n" + "\n".join(rows)


def analyze_risk_by_group(dashboard_df: pd.DataFrame, group_column: str, top_n: int = 10) -> str:
    top_n = min(int(top_n), 30)

    # Fuzzy-match column name (case-insensitive)
    col_map = {c.lower(): c for c in dashboard_df.columns}
    matched_col = col_map.get(group_column.lower())
    if matched_col is None:
        available = ", ".join(sorted(dashboard_df.columns.tolist()))
        return f"Column '{group_column}' not found. Available columns: {available}"

    # Bin numeric columns into quartiles for readability
    series = dashboard_df[matched_col]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        binned_col = f"{matched_col}_bin"
        dashboard_df = dashboard_df.copy()
        dashboard_df[binned_col] = pd.qcut(series, q=4, duplicates="drop").astype(str)
        group_col = binned_col
    else:
        group_col = matched_col

    grouped = (
        dashboard_df.groupby(group_col)
        .agg(
            headcount=("Turnover Probability", "count"),
            avg_risk=("Turnover Probability", "mean"),
            high_risk_count=("Risk Category", lambda x: x.isin(["High Risk", "Very High Risk"]).sum()),
        )
        .reset_index()
    )
    grouped["avg_risk_pct"] = grouped["avg_risk"] * 100
    grouped = grouped.sort_values("avg_risk_pct", ascending=False).head(top_n)

    rows = []
    for _, r in grouped.iterrows():
        pct_high = (r["high_risk_count"] / r["headcount"] * 100) if r["headcount"] > 0 else 0
        rows.append(
            f"  {r[group_col]}: {r['avg_risk_pct']:.1f}% avg risk, "
            f"{int(r['headcount'])} employees, {pct_high:.0f}% are high/very-high risk"
        )
    return f"Risk breakdown by '{matched_col}' (top {top_n} groups by avg risk):\n" + "\n".join(rows)


def get_company_risk_stats(dashboard_df: pd.DataFrame) -> str:
    prob = dashboard_df["Turnover Probability"] * 100
    percentiles = prob.quantile([0.25, 0.50, 0.75, 0.90, 0.95])
    tier_counts = dashboard_df["Risk Category"].value_counts().to_dict()
    total = len(dashboard_df)

    tier_lines = "\n".join(
        f"    {cat}: {cnt} ({cnt/total*100:.1f}%)"
        for cat, cnt in sorted(tier_counts.items(), key=lambda x: -x[1])
    )
    return (
        f"Company-Wide Risk Statistics ({total} employees):\n"
        f"  Mean:   {prob.mean():.1f}%\n"
        f"  Median: {prob.median():.1f}%\n"
        f"  Std:    {prob.std():.1f}%\n"
        f"  Min:    {prob.min():.1f}%  |  Max: {prob.max():.1f}%\n"
        f"  Percentiles:\n"
        f"    25th: {percentiles[0.25]:.1f}%\n"
        f"    50th: {percentiles[0.50]:.1f}%\n"
        f"    75th: {percentiles[0.75]:.1f}%\n"
        f"    90th: {percentiles[0.90]:.1f}%\n"
        f"    95th: {percentiles[0.95]:.1f}%\n"
        f"  Risk Tier Breakdown:\n{tier_lines}"
    )


def find_employees_by_criteria(
    dashboard_df: pd.DataFrame,
    min_risk_pct: float = 0,
    max_risk_pct: float = 100,
    department: Optional[str] = None,
    min_tenure_months: Optional[float] = None,
    max_tenure_months: Optional[float] = None,
    top_n: int = 10,
) -> str:
    top_n = min(int(top_n), 30)
    df = dashboard_df.copy()

    df = df[(df["Turnover Probability"] * 100 >= min_risk_pct) & (df["Turnover Probability"] * 100 <= max_risk_pct)]

    if department:
        df = df[df["Budget Section"].astype(str).str.lower() == department.lower()]

    if min_tenure_months is not None and "Tenure (Months)" in df.columns:
        df = df[df["Tenure (Months)"] >= min_tenure_months]
    if max_tenure_months is not None and "Tenure (Months)" in df.columns:
        df = df[df["Tenure (Months)"] <= max_tenure_months]

    if df.empty:
        return "No employees match the specified criteria."

    df = df.sort_values("Turnover Probability", ascending=False).head(top_n)
    rows = []
    for _, r in df.iterrows():
        tenure_str = f", tenure {r['Tenure (Months)']:.0f}m" if "Tenure (Months)" in r and pd.notna(r.get("Tenure (Months)")) else ""
        rows.append(
            f"  Employee {r['Employee ID']} — {r['Turnover Probability']*100:.1f}% "
            f"({r['Risk Category']}), dept: {r.get('Budget Section', 'N/A')}{tenure_str}"
        )

    filters = []
    if department: filters.append(f"dept={department}")
    if min_risk_pct > 0: filters.append(f"risk≥{min_risk_pct}%")
    if max_risk_pct < 100: filters.append(f"risk≤{max_risk_pct}%")
    if min_tenure_months is not None: filters.append(f"tenure≥{min_tenure_months}m")
    if max_tenure_months is not None: filters.append(f"tenure≤{max_tenure_months}m")
    filter_desc = f" [{', '.join(filters)}]" if filters else ""

    return f"Employees matching criteria{filter_desc} ({len(df)} shown, sorted by risk):\n" + "\n".join(rows)


def list_available_columns(dashboard_df: pd.DataFrame) -> str:
    cols = sorted(dashboard_df.columns.tolist())
    return "Available columns in predictions data:\n" + "\n".join(f"  - {c}" for c in cols)


def simulate_what_if(
    dashboard_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    api,
    employee_id: str,
    salary_delta_pct: float = 0,
    workload_delta_pct: float = 0,
    illness_delta_pct: float = 0,
) -> str:
    if api is None:
        return "Live inference API is not available. Cannot run simulation."
    if raw_df.empty:
        return "Raw employee data is not loaded. Cannot run simulation."

    employee_id_col = getattr(api, "employee_id_col", "fictive_employee")
    time_col = getattr(api, "time_col", "calc_month")
    if employee_id_col not in raw_df.columns:
        # Fall back to a legacy dataset config if present.
        dataset_config = getattr(api, "pipeline", {}) or {}
        dataset_config = dataset_config.get("dataset_config", {}) if isinstance(dataset_config, dict) else {}
        employee_id_col = dataset_config.get("employee_id_col", employee_id_col)
        time_col = dataset_config.get("time_col", time_col)

    emp_records = raw_df[raw_df[employee_id_col].astype(str) == str(employee_id)].copy()
    if emp_records.empty:
        return f"No raw records found for employee '{employee_id}'."

    if time_col and time_col in emp_records.columns:
        emp_records = emp_records.sort_values(time_col)

    try:
        current_prob, current_cat = api.predict_risk(emp_records)
    except Exception as e:
        match = dashboard_df[dashboard_df["Employee ID"].astype(str) == str(employee_id)]
        if match.empty:
            return f"Could not compute risk for employee '{employee_id}': {e}"
        current_prob = match.iloc[0]["Turnover Probability"]
        current_cat = match.iloc[0]["Risk Category"]

    latest_idx = emp_records.index[-1]
    latest = emp_records.loc[latest_idx]

    def _delta(col: str, pct: float):
        try:
            val = float(latest.get(col, 0) or 0)
            return val * (1 + pct / 100.0)
        except Exception:
            return None

    new_salary = _delta("avg_Payment", salary_delta_pct)
    new_workload = _delta("avg_omes", workload_delta_pct)
    new_illness = _delta("avg_illness", illness_delta_pct)

    # Use the final-model-aligned what-if transform when available so derived
    # features stay consistent; otherwise edit the latest row directly.
    is_final = employee_id_col == "fictive_employee"
    try:
        if is_final:
            from src.final_dashboard import apply_final_what_if
            baseline_records = apply_final_what_if(
                emp_records,
                salary=float(latest.get("avg_Payment", 0) or 0),
                workload=float(latest.get("avg_omes", 0) or 0),
                illness=float(latest.get("avg_illness", 0) or 0),
            )
            mod_records = apply_final_what_if(
                emp_records,
                salary=new_salary,
                workload=new_workload,
                illness=new_illness,
            )
            base_prob, _ = api.predict_risk(baseline_records)
            mod_prob, _ = api.predict_risk(mod_records)
            new_prob = float(min(max(current_prob + (mod_prob - base_prob), 0.0), 1.0))
            new_cat = api.risk_category(new_prob)
        else:
            mod_records = emp_records.copy()
            if new_salary is not None:
                mod_records.at[latest_idx, "avg_Payment"] = new_salary
            if new_workload is not None:
                mod_records.at[latest_idx, "avg_omes"] = new_workload
            if new_illness is not None:
                mod_records.at[latest_idx, "avg_illness"] = new_illness
            new_prob, new_cat = api.predict_risk(mod_records)
    except Exception as e:
        return f"Simulation failed: {e}"

    delta_prob = (new_prob - current_prob) * 100
    changes = []
    if salary_delta_pct:
        changes.append(f"salary {salary_delta_pct:+.0f}%")
    if workload_delta_pct:
        changes.append(f"workload {workload_delta_pct:+.0f}%")
    if illness_delta_pct:
        changes.append(f"sick days {illness_delta_pct:+.0f}%")
    change_desc = ", ".join(changes) if changes else "no changes"

    direction = "increase" if delta_prob > 0 else ("decrease" if delta_prob < 0 else "no change")
    return (
        f"What-If Simulation for Employee {employee_id} ({change_desc}):\n"
        f"  Current Risk:   {current_prob*100:.1f}% ({current_cat})\n"
        f"  Simulated Risk: {new_prob*100:.1f}% ({new_cat})\n"
        f"  Change: {delta_prob:+.1f}pp {direction}"
    )


def analyze_feature_correlations(dashboard_df: pd.DataFrame, top_n: int = 12) -> str:
    """Correlate numeric features with turnover probability and show which
    categories carry the most risk spread — a data-driven 'trends' view."""
    from src.config import FEATURE_DESCRIPTIONS

    if "Turnover Probability" not in dashboard_df.columns:
        return "Turnover Probability column is missing; cannot compute correlations."

    y = pd.to_numeric(dashboard_df["Turnover Probability"], errors="coerce")
    skip = {
        "Turnover Probability", "turnover_probability", "turnover_prediction",
        "risk_rank", "Employee ID", "fictive_employee",
    }

    numeric_rows: list[tuple[str, float]] = []
    categorical_rows: list[tuple[str, float, float]] = []
    for col in dashboard_df.columns:
        if col in skip:
            continue
        series = dashboard_df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > 10 and numeric.nunique(dropna=True) > 8:
            corr = numeric.corr(y)
            if pd.notna(corr):
                numeric_rows.append((col, float(corr)))
        else:
            valid = series.dropna()
            if 2 <= valid.nunique() <= 30 and len(valid) > 10:
                grp = y.groupby(series).mean()
                counts = series.value_counts()
                grp = grp[counts.reindex(grp.index).fillna(0) >= 5]
                if len(grp) >= 2:
                    spread = float((grp.max() - grp.min()) * 100)
                    categorical_rows.append((col, spread, float(grp.mean() * 100)))

    numeric_rows.sort(key=lambda kv: abs(kv[1]), reverse=True)
    categorical_rows.sort(key=lambda kv: kv[1], reverse=True)

    def _name(c: str) -> str:
        return FEATURE_DESCRIPTIONS.get(c, c)

    lines = ["Trends & correlations with turnover risk:"]
    if numeric_rows:
        lines.append("Numeric features (Pearson correlation with risk; + = higher value → higher risk):")
        for col, corr in numeric_rows[:top_n]:
            arrow = "↑" if corr > 0 else "↓"
            lines.append(f"  - {_name(col)}: r = {corr:+.2f} {arrow}")
    if categorical_rows:
        lines.append("Categorical features (risk spread between best and worst category):")
        for col, spread, avg in categorical_rows[: max(4, top_n // 2)]:
            lines.append(f"  - {_name(col)}: {spread:.0f} pp spread (avg {avg:.0f}%)")
    if len(lines) == 1:
        return "Not enough variation in the data to compute meaningful correlations."
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def build_context(
    dashboard_df: pd.DataFrame,
    selected_employee_id=None,
    selected_dept=None,
) -> str:
    total = len(dashboard_df)
    avg_risk = dashboard_df["Turnover Probability"].mean() * 100
    high_risk = int(dashboard_df["Risk Category"].isin(["High Risk", "Very High Risk"]).sum())
    dist = dashboard_df["Risk Category"].value_counts().to_dict()

    lines = [
        "## Live Dashboard Context",
        f"- Total employees: {total}",
        f"- Company-wide avg risk: {avg_risk:.1f}%",
        f"- High/Very High risk headcount: {high_risk}",
        f"- Risk distribution: {dist}",
    ]

    if selected_dept:
        dept_df = dashboard_df[dashboard_df["Budget Section"].astype(str) == str(selected_dept)]
        if not dept_df.empty:
            lines.append(f"\n## Currently Viewed Department: {selected_dept}")
            lines.append(f"- Team size: {len(dept_df)}")
            lines.append(f"- Team avg risk: {dept_df['Turnover Probability'].mean()*100:.1f}%")
            lines.append(f"- Team high-risk count: {int(dept_df['Risk Category'].isin(['High Risk','Very High Risk']).sum())}")

    if selected_employee_id:
        match = dashboard_df[dashboard_df["Employee ID"].astype(str) == str(selected_employee_id)]
        if not match.empty:
            r = match.iloc[0]
            lines.append(f"\n## Currently Viewed Employee: {selected_employee_id}")
            lines.append(f"- Risk: {r['Turnover Probability']*100:.1f}% ({r['Risk Category']})")
            lines.append(f"- Department: {r.get('Budget Section', 'N/A')}")
            if "Tenure (Months)" in r:
                lines.append(f"- Tenure: {r['Tenure (Months)']:.0f} months")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Chat Function
# ---------------------------------------------------------------------------
def run_sql_query(dashboard_df: pd.DataFrame, query: str) -> str:
    """Execute a read-only SQLite SELECT against the predictions table and
    return a compact text payload (row count + up to 25 rows)."""
    if not query or not query.strip():
        return "No SQL query was provided."
    normalized = query.strip().lstrip().lower()
    if not normalized.startswith("select") and not normalized.startswith("with"):
        return "Rejected: only read-only SELECT queries are allowed."
    forbidden = ("insert", "update", "delete", "drop", "alter", "pragma", "attach", "create", ";--")
    if any(tok in normalized for tok in (" insert ", " update ", " delete ", " drop ", " alter ", " pragma ")):
        return "Rejected: the query contains a non-read-only statement."

    df_for_sql = dashboard_df.copy()
    for col in df_for_sql.columns:
        df_for_sql[col] = df_for_sql[col].map(lambda x: x.item() if hasattr(x, "item") else x)

    try:
        with sqlite3.connect(":memory:") as conn:
            df_for_sql.to_sql("turnover_data", conn, index=False, if_exists="replace")
            result_df = pd.read_sql_query(query, conn)
    except Exception as e:
        return f"SQL execution failed: {e}\nQuery: {query}"

    preview = result_df.head(25).to_dict(orient="records")
    return (
        f"SQL returned {len(result_df)} row(s). "
        f"Columns: {list(result_df.columns)}. "
        f"Rows (first 25): {preview}"
        + ("" if len(result_df) <= 25 else " [truncated]")
    )


# ---------------------------------------------------------------------------
# OpenAI tool schemas + agent loop (fallback provider)
# ---------------------------------------------------------------------------
def _openai_tool_schemas() -> list[dict[str, Any]]:
    """JSON-schema declarations mirroring the chatbot tools, for OpenAI function calling."""

    def f(name: str, desc: str, properties: dict, required: list | None = None) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or [],
                },
            },
        }

    return [
        f("explain_employee",
          "Explain WHY a specific employee's turnover risk is high or low using SHAP feature contributions.",
          {"employee_id": {"type": "string", "description": "The Employee ID."}},
          ["employee_id"]),
        f("top_risk_drivers",
          "Rank the features that drive turnover risk across the whole organisation (SHAP global importance).",
          {"top_n": {"type": "integer", "description": "How many drivers to return (default 15)."}}),
        f("feature_correlations",
          "Show which features correlate with turnover risk (numeric correlations + categorical risk spreads).",
          {"top_n": {"type": "integer", "description": "How many features to return (default 12)."}}),
        f("high_risk_employees",
          "List the highest-risk employees, optionally within a department.",
          {"department": {"type": "string", "description": "Department name; empty for company-wide."},
           "top_n": {"type": "integer", "description": "How many to return (default 5)."}}),
        f("employee_risk",
          "Return the current risk score and category for one employee by ID.",
          {"employee_id": {"type": "string", "description": "The Employee ID."}},
          ["employee_id"]),
        f("department_summary",
          "Summarise turnover risk for a department (headcount, avg risk, distribution).",
          {"department": {"type": "string", "description": "Department name."}},
          ["department"]),
        f("rank_departments",
          "Rank departments by average turnover risk.",
          {"top_n": {"type": "integer", "description": "How many departments (default 10)."},
           "order": {"type": "string", "enum": ["desc", "asc"], "description": "'desc' = riskiest first."}}),
        f("risk_by_group",
          "Average turnover risk grouped by any column (e.g. 'Job Rank', 'Contract Type', 'City of Residence').",
          {"group_column": {"type": "string", "description": "Column to group by."},
           "top_n": {"type": "integer", "description": "How many groups (default 10)."}},
          ["group_column"]),
        f("company_stats",
          "Company-wide risk statistics: mean, median, std, percentiles, and risk-tier breakdown.",
          {}),
        f("find_employees",
          "Filter employees by risk %, department and tenure range; returns a ranked list.",
          {"min_risk_pct": {"type": "number"}, "max_risk_pct": {"type": "number"},
           "department": {"type": "string", "description": "Empty for no department filter."},
           "min_tenure_months": {"type": "number", "description": "Use -1 to skip."},
           "max_tenure_months": {"type": "number", "description": "Use -1 to skip."},
           "top_n": {"type": "integer"}}),
        f("what_if",
          "Simulate how % changes to salary, workload or sick days change an employee's turnover risk.",
          {"employee_id": {"type": "string"},
           "salary_delta_pct": {"type": "number"},
           "workload_delta_pct": {"type": "number"},
           "illness_delta_pct": {"type": "number"}},
          ["employee_id"]),
        f("run_sql",
          "Run a read-only SQLite SELECT against table `turnover_data` (predictions). SELECT only.",
          {"query": {"type": "string", "description": "A single SELECT query."}},
          ["query"]),
    ]


def _run_openai_agent(
    api_key: str,
    callables: dict[str, Any],
    system_prompt: str,
    context_block: str,
    history: list,
    user_message: str,
    model: str | None = None,
) -> str:
    """Run an OpenAI chat-completions tool-calling loop using the same tool implementations."""
    try:
        from openai import OpenAI
    except Exception:
        return "The 'openai' package is not installed. Run: pip install openai"

    model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{system_prompt}\n\n{context_block}"}
    ]
    for item in (history or [])[-10:]:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    tools = _openai_tool_schemas()

    try:
        for _ in range(6):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = response.choices[0].message
            if not getattr(msg, "tool_calls", None):
                return (msg.content or "").strip() or "I wasn't able to produce an answer for that."

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                fn = callables.get(name)
                try:
                    result = fn(**args) if fn else f"Unknown tool: {name}"
                except Exception as exc:
                    result = f"Tool '{name}' failed: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        # Tool budget exhausted — ask for a final answer without more tools.
        final = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        return (final.choices[0].message.content or "").strip() or "I couldn't complete that request."
    except Exception as exc:
        return f"Assistant error (OpenAI): {exc}"


# ---------------------------------------------------------------------------
# Main Chat Function
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: list,
    dashboard_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    api,
    selected_employee_id=None,
    selected_dept=None,
) -> str:
    """
    Send a message to the AI assistant and return its response.

    Uses Gemini function calling: the model chooses among analytical, SHAP
    explanation, correlation, what-if and SQL tools, then writes a grounded
    answer.

    Args:
        user_message:        The user's latest message.
        history:             List of {"role": ..., "content": ...} dicts (prior turns).
        dashboard_df:        Batch predictions DataFrame.
        raw_df:              Raw employee records DataFrame.
        api:                 TurnoverInferenceAPI instance (or None).
        selected_employee_id: Employee currently viewed in the Micro tab.
        selected_dept:       Department currently viewed in the Meso tab.

    Returns:
        The assistant's response string.
    """
    gemini_key, openai_key = _resolve_provider_keys()
    if not gemini_key and not openai_key:
        return (
            "No API key found. Add GEMINI_API_KEY (preferred) or an OpenAI key "
            "(OPENAI_API_KEY / OPEN_AI_API_KEY / OPEN_AI_KEY) to your .env file."
        )

    context_block = build_context(dashboard_df, selected_employee_id, selected_dept)

    emp_id_col = getattr(api, "employee_id_col", "fictive_employee") if api is not None else "fictive_employee"
    time_col = getattr(api, "time_col", "calc_month") if api is not None else "calc_month"

    def _emp_records(employee_id: str) -> pd.DataFrame:
        if raw_df is None or raw_df.empty or emp_id_col not in raw_df.columns:
            return pd.DataFrame()
        recs = raw_df[raw_df[emp_id_col].astype(str) == str(employee_id)].copy()
        if time_col and time_col in recs.columns:
            recs = recs.sort_values(time_col)
        return recs

    background_df = raw_df if (raw_df is not None and not raw_df.empty) else dashboard_df

    # ---- Tool closures (schema is inferred from type hints + docstrings) ----
    def explain_employee(employee_id: str) -> str:
        """Explain WHY a specific employee's turnover risk is high or low using SHAP
        feature contributions (with a risk-sensitivity fallback). Returns the factors
        pushing risk up and down, with the employee's values vs typical values."""
        if api is None:
            return "The live model is not loaded, so individual predictions can't be explained."
        recs = _emp_records(employee_id)
        if recs.empty:
            return f"No records found for employee '{employee_id}'."
        return explain_employee_prediction(api, recs, background_df, employee_id=str(employee_id))

    def top_risk_drivers(top_n: int = 15) -> str:
        """Rank the features that drive turnover risk across the whole organisation using
        SHAP global importance. Use for 'what drives turnover?' and trend questions."""
        if api is None:
            return "The live model is not loaded, so global SHAP importance is unavailable."
        return global_feature_importance(api, background_df, top_n=int(top_n))

    def feature_correlations(top_n: int = 12) -> str:
        """Show which features correlate with turnover risk (numeric Pearson correlations and
        categorical risk spreads). Use for trend / correlation questions."""
        return analyze_feature_correlations(dashboard_df, top_n=int(top_n))

    def high_risk_employees(department: str = "", top_n: int = 5) -> str:
        """List the highest-risk employees, optionally within a department (empty = company-wide)."""
        return list_high_risk_employees(dashboard_df, department or None, int(top_n))

    def employee_risk(employee_id: str) -> str:
        """Return the current risk score and category for one employee by ID."""
        return get_employee_risk(dashboard_df, employee_id)

    def department_summary(department: str) -> str:
        """Summarise turnover risk for a department (headcount, avg risk, distribution)."""
        return get_department_summary(dashboard_df, department)

    def rank_departments(top_n: int = 10, order: str = "desc") -> str:
        """Rank departments by average turnover risk. order='desc' = riskiest first, 'asc' = safest."""
        return rank_departments_by_risk(dashboard_df, int(top_n), order or "desc")

    def risk_by_group(group_column: str, top_n: int = 10) -> str:
        """Average turnover risk grouped by any column (e.g. 'Job Rank', 'Contract Type', 'City of Residence')."""
        return analyze_risk_by_group(dashboard_df, group_column, int(top_n))

    def company_stats() -> str:
        """Company-wide risk statistics: mean, median, std, percentiles, and risk-tier breakdown."""
        return get_company_risk_stats(dashboard_df)

    def find_employees(
        min_risk_pct: float = 0,
        max_risk_pct: float = 100,
        department: str = "",
        min_tenure_months: float = -1,
        max_tenure_months: float = -1,
        top_n: int = 10,
    ) -> str:
        """Filter employees by risk %, department and tenure range; returns a ranked list.
        Use -1 for a tenure bound to skip it, and an empty string for no department filter."""
        return find_employees_by_criteria(
            dashboard_df,
            min_risk_pct=min_risk_pct,
            max_risk_pct=max_risk_pct,
            department=department or None,
            min_tenure_months=None if min_tenure_months is None or min_tenure_months < 0 else min_tenure_months,
            max_tenure_months=None if max_tenure_months is None or max_tenure_months < 0 else max_tenure_months,
            top_n=int(top_n),
        )

    def what_if(
        employee_id: str,
        salary_delta_pct: float = 0,
        workload_delta_pct: float = 0,
        illness_delta_pct: float = 0,
    ) -> str:
        """Simulate how percentage changes to salary, workload or sick days would change a
        specific employee's turnover risk. Returns current vs simulated risk."""
        return simulate_what_if(
            dashboard_df, raw_df, api, employee_id,
            salary_delta_pct, workload_delta_pct, illness_delta_pct,
        )

    def run_sql(query: str) -> str:
        """Run a read-only SQLite SELECT against table `turnover_data` (the predictions).
        Use only for bespoke slicing not covered by the other tools. SELECT statements only."""
        return run_sql_query(dashboard_df, query)

    tools = [
        explain_employee, top_risk_drivers, feature_correlations,
        high_risk_employees, employee_risk, department_summary,
        rank_departments, risk_by_group, company_stats, find_employees,
        what_if, run_sql,
    ]
    callables = {fn.__name__: fn for fn in tools}

    history_lines: list[str] = []
    for item in history:
        role = str(item.get("role", "user")).strip()
        content = str(item.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    transcript = "\n".join(history_lines[-10:])

    # ---- Provider 1: Gemini (preferred) ----
    if gemini_key:
        contents = (
            f"{context_block}\n\n"
            + (f"Conversation history:\n{transcript}\n\n" if transcript else "")
            + f"Latest user message:\n{user_message}"
        )
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=tools,
                    temperature=0.2,
                ),
            )
        except Exception as e:
            return f"Assistant error (Gemini): {e}"

        answer_text = (getattr(response, "text", "") or "").strip()
        if not answer_text:
            return "I wasn't able to produce an answer for that. Try rephrasing or asking about a specific employee, department, or driver."
        return answer_text

    # ---- Provider 2: OpenAI (fallback) ----
    return _run_openai_agent(
        openai_key, callables, SYSTEM_PROMPT, context_block, history, user_message
    )
