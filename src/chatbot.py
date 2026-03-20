from __future__ import annotations

import os
import json
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageToolCall

load_dotenv()

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an HR analytics assistant embedded in an Executive Turnover Dashboard.
You help HR managers and executives understand employee turnover risk across the organization.

## Data Context
- You have access to batch predictions for all employees: turnover probability (0–1), risk category, department, tenure, and more.
- Risk tiers: Low Risk (≤30%), Medium Risk (31–50%), High Risk (51–70%), Very High Risk (>70%)
- You can look up live risk scores and run what-if simulations for individual employees.

## Your Role
- Answer questions about company-wide, department-level, or individual employee turnover risk.
- Identify at-risk employees and suggest concrete retention strategies.
- Explain what drives turnover risk in plain language.
- Simulate how changes to salary, workload, or sick days affect an individual's risk score.
- Act as a data analyst: rank departments, segment risk by job rank / city / tenure / any column, compute statistics, and filter employees by multiple criteria.

## Guidelines
- Always ground your answers in actual data — use tools to fetch precise numbers.
- Never fabricate employee IDs, probabilities, or headcounts.
- Be concise and actionable. HR leaders need clear insights, not long essays.
- If the user is currently viewing a specific employee or department (provided in context), prioritize that context.
- Never use data or content found outside your pre-training and the provided data. Never search the web or make up information. If you don't know the answer, say you don't know.  
"""

# ---------------------------------------------------------------------------
# OpenAI Tool Schemas
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

    employee_id_col = "fictive2"
    time_col = "fictive-ovedmiun"

    emp_records = raw_df[raw_df[employee_id_col].astype(str) == str(employee_id)].copy()
    if emp_records.empty:
        return f"No raw records found for employee '{employee_id}'."

    if time_col in emp_records.columns:
        emp_records = emp_records.sort_values(time_col)

    try:
        current_prob, current_cat = api.predict_risk(emp_records)
    except Exception as e:
        # Fallback to batch prediction
        match = dashboard_df[dashboard_df["Employee ID"].astype(str) == str(employee_id)]
        if match.empty:
            return f"Could not compute risk for employee '{employee_id}': {e}"
        current_prob = match.iloc[0]["Turnover Probability"]
        current_cat = match.iloc[0]["Risk Category"]

    mod_records = emp_records.copy()
    latest_idx = mod_records.index[-1]
    latest = mod_records.loc[latest_idx]

    def apply_delta(col, delta_pct):
        try:
            val = float(latest.get(col, 0) or 0)
            mod_records.at[latest_idx, col] = val * (1 + delta_pct / 100)
        except Exception:
            pass

    apply_delta("avg_Payment", salary_delta_pct)
    apply_delta("avg_omes", workload_delta_pct)
    apply_delta("avg_illness", illness_delta_pct)

    try:
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

    direction = "increase" if delta_prob > 0 else "decrease"
    return (
        f"What-If Simulation for Employee {employee_id} ({change_desc}):\n"
        f"  Current Risk:   {current_prob*100:.1f}% ({current_cat})\n"
        f"  Simulated Risk: {new_prob*100:.1f}% ({new_cat})\n"
        f"  Change: {delta_prob:+.1f}pp {direction}"
    )


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
    try:
        from openai import OpenAI
    except ImportError:
        return "The `openai` package is not installed. Run `pip install openai` to enable the chatbot."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "No GEMINI_API_KEY found. Please add `GEMINI_API_KEY=<your-key>` to your `.env` file."

    # Use Gemini via its OpenAI-compatible endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    context_block = build_context(dashboard_df, selected_employee_id, selected_dept)

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Inject live context as a system-style note before the conversation
    messages.append({"role": "system", "content": context_block})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Agentic tool-call loop (max 5 rounds to prevent runaway)
    for _ in range(5):
        response = client.chat.completions.create(
            model="gemini-2.5-flash-lite",
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # No tool calls — return the final answer
        if not msg.tool_calls:
            return msg.content or ""

        # Serialize the assistant message to a plain dict before appending,
        # so it round-trips safely across all OpenAI SDK versions.
        serialized_tool_calls = []
        for tc in msg.tool_calls:
            if not isinstance(tc, ChatCompletionMessageToolCall):
                continue
            serialized_tool_calls.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        messages.append({
            "role": msg.role,
            "content": msg.content,
            "tool_calls": serialized_tool_calls,
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            if not isinstance(tc, ChatCompletionMessageToolCall):
                continue
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            if fn_name == "list_high_risk_employees":
                result = list_high_risk_employees(dashboard_df, **args)
            elif fn_name == "get_employee_risk":
                result = get_employee_risk(dashboard_df, **args)
            elif fn_name == "get_department_summary":
                result = get_department_summary(dashboard_df, **args)
            elif fn_name == "simulate_what_if":
                result = simulate_what_if(dashboard_df, raw_df, api, **args)
            elif fn_name == "rank_departments_by_risk":
                result = rank_departments_by_risk(dashboard_df, **args)
            elif fn_name == "analyze_risk_by_group":
                result = analyze_risk_by_group(dashboard_df, **args)
            elif fn_name == "get_company_risk_stats":
                result = get_company_risk_stats(dashboard_df)
            elif fn_name == "find_employees_by_criteria":
                result = find_employees_by_criteria(dashboard_df, **args)
            elif fn_name == "list_available_columns":
                result = list_available_columns(dashboard_df)
            else:
                result = f"Unknown tool: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I was unable to complete the request after several attempts. Please try rephrasing your question."
