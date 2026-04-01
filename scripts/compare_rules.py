"""
compare_rules.py
----------------
Parses WizWhy's if-then rules and compares feature importance with SHAP.

For each feature appearing in the rules, computes:
  - Rule Count       : how many rules mention it
  - Weighted Records : sum of (records in rule) across all rules that mention it
  - Avg Probability  : average rule probability weighted by records
  - Avg IFF Criterion: average IFF Criterion (statistical strength) weighted by records
  - Avg I.F.         : average Information Factor weighted by records

Then aligns against the SHAP top-20 from comparison_results.xlsx.

Outputs:
  WizWhy/rule_feature_comparison.xlsx  — two sheets:
    "Feature Comparison"   : SHAP rank | WizWhy rank | all metrics side-by-side
    "Rule Detail"          : every parsed rule with its conditions and stats

Run:
    python compare_rules.py
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Resolve paths relative to project root (one level up from scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

IFTHEN_PATH      = os.path.join("WizWhy", "if-then.txt")
SHAP_RESULTS     = os.path.join("WizWhy", "comparison_results.xlsx")
OUTPUT_PATH      = os.path.join("WizWhy", "rule_feature_comparison.txt")

try:
    from src.config import FEATURE_DESCRIPTIONS
except ImportError:
    FEATURE_DESCRIPTIONS = {}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_rules(filepath: str) -> list[dict]:
    """
    Parse WizWhy if-then rules file into a list of rule dicts:
      {
        rule_num: int,
        conditions: [{'feature': str, 'condition': str}],
        probability: float,
        records: int,
        iff_criterion: float,
        if_factor: float,
      }
    """
    with open(filepath, encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Split on rule boundaries: lines like " 1)\t" or " 1050)\t"
    rule_blocks = re.split(r'\n\s+(\d+)\)\t', text)

    # First element is the header — skip it
    # After split: [header, rule_num, rule_body, rule_num, rule_body, ...]
    rules = []
    i = 1
    while i + 1 < len(rule_blocks):
        rule_num = int(rule_blocks[i])
        body = rule_blocks[i + 1]
        rule = _parse_rule_body(rule_num, body)
        if rule:
            rules.append(rule)
        i += 2

    return rules


def _parse_rule_body(rule_num: int, body: str) -> dict | None:
    """Parse a single rule body string into a structured dict."""
    conditions = []

    # Extract conditions: lines starting with "If  " or "and "
    cond_lines = re.findall(
        r'(?:If|and)\s+(\S+)\s+(?:is|=)\s+(.+?)(?=\n|$)',
        body, re.IGNORECASE
    )
    for feat, cond_text in cond_lines:
        feat = feat.strip()
        # Skip fictive2 (employee ID — not a real feature)
        if feat.lower() == 'fictive2':
            continue
        conditions.append({'feature': feat, 'condition': cond_text.strip()})

    if not conditions:
        return None

    # Rule probability
    prob_match = re.search(r"Rule's probability:\s+([\d.]+)", body)
    probability = float(prob_match.group(1)) if prob_match else None

    # Records
    rec_match = re.search(r"rule exists in\s+([\d,]+)\s+records", body)
    records = int(rec_match.group(1).replace(',', '')) if rec_match else None

    # IFF Criterion
    iff_match = re.search(r'IFF Criterion \(([\d.]+)\)', body)
    iff_criterion = float(iff_match.group(1)) if iff_match else None

    # I.F.
    if_match = re.search(r'I\.F\. \(([\d.]+)\)', body)
    if_factor = float(if_match.group(1)) if if_match else None

    # Then clause: determine if rule predicts turnover (leave) or retention (stay)
    then_match = re.search(r'Then\s+leave_ind\s+is\s+(not\s+)?more than', body, re.IGNORECASE)
    if then_match:
        is_turnover_rule = then_match.group(1) is None  # True if "more than" (no "not")
    else:
        is_turnover_rule = True  # Fallback: treat as turnover if Then clause not found

    return {
        'rule_num': rule_num,
        'conditions': conditions,
        'n_conditions': len(conditions),
        'probability': probability,
        'records': records,
        'iff_criterion': iff_criterion,
        'if_factor': if_factor,
        'is_turnover_rule': is_turnover_rule,
    }


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------

def aggregate_feature_stats(rules: list[dict]) -> pd.DataFrame:
    """
    For each feature, aggregate stats across all rules mentioning it.
    Weighted by number of records in each rule.

    Separates turnover rules (predicting leave) from retention rules (predicting stay)
    to avoid conflating drivers of churn with drivers of loyalty.
    """
    feature_data = defaultdict(lambda: {
        'rule_count_leave': 0,
        'rule_count_stay': 0,
        'total_records': 0,
        'prob_x_rec': 0.0,
        'iff_x_rec': 0.0,
        'if_x_rec': 0.0,
    })

    for rule in rules:
        rec = rule['records'] or 0
        prob = rule['probability'] or 0.0
        iff = rule['iff_criterion'] or 0.0
        ifac = rule['if_factor'] or 0.0
        is_leave = rule.get('is_turnover_rule', True)

        for cond in rule['conditions']:
            feat = cond['feature']
            fd = feature_data[feat]
            if is_leave:
                fd['rule_count_leave'] += 1
            else:
                fd['rule_count_stay'] += 1
            fd['total_records'] += rec
            fd['prob_x_rec'] += prob * rec
            fd['iff_x_rec'] += iff * rec
            fd['if_x_rec'] += ifac * rec

    rows = []
    for feat, fd in feature_data.items():
        total_rec = fd['total_records']
        rule_count = fd['rule_count_leave'] + fd['rule_count_stay']
        rows.append({
            'Feature (Raw)': feat,
            'Feature (Readable)': FEATURE_DESCRIPTIONS.get(feat, feat),
            'Rule Count': rule_count,
            'Rule Count (Leave)': fd['rule_count_leave'],
            'Rule Count (Stay)': fd['rule_count_stay'],
            'Total Records Covered': total_rec,
            'Avg Probability (weighted)': round(fd['prob_x_rec'] / total_rec, 4) if total_rec else 0,
            'Avg IFF Criterion (weighted)': round(fd['iff_x_rec'] / total_rec, 4) if total_rec else 0,
            'Avg I.F. (weighted)': round(fd['if_x_rec'] / total_rec, 4) if total_rec else 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Rule Count (Leave)', ascending=False).reset_index(drop=True)
    df.insert(0, 'WizWhy Rank', df.index + 1)
    return df


# ---------------------------------------------------------------------------
# Rule detail table
# ---------------------------------------------------------------------------

def build_rule_detail(rules: list[dict]) -> pd.DataFrame:
    """Flat table: one row per rule, conditions as a readable string."""
    rows = []
    for rule in rules:
        cond_str = ' AND '.join(
            f"{c['feature']} {c['condition']}" for c in rule['conditions']
        )
        rows.append({
            'Rule #': rule['rule_num'],
            'Predicts': 'Leave' if rule.get('is_turnover_rule', True) else 'Stay',
            'Conditions': cond_str,
            '# Conditions': rule['n_conditions'],
            'Probability': rule['probability'],
            'Records': rule['records'],
            'IFF Criterion': rule['iff_criterion'],
            'I.F.': rule['if_factor'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge with SHAP
# ---------------------------------------------------------------------------

def _base_name(feature: str) -> str:
    """
    Strip time-series suffixes so aggregated SHAP names match raw WizWhy names.
    e.g. 'vetek_months_std' → 'vetek_months'
         'manager_Code_mean' → 'manager_Code'
         'avg_Payment_trend' → 'avg_Payment'
    Engineered names without suffixes are returned as-is.
    """
    for suffix in ('_std', '_mean', '_trend'):
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def merge_with_shap(ww_df: pd.DataFrame, shap_path: str) -> pd.DataFrame:
    """
    Load SHAP sheet from comparison_results.xlsx.

    SHAP features use aggregated names (e.g. vetek_months_std) while WizWhy
    uses raw names (vetek_months).  We group SHAP features by their base name
    (stripping _mean/_std/_trend), sum their |SHAP| values, then join on the
    base name against WizWhy's raw feature names.
    """
    try:
        shap_df = pd.read_excel(shap_path, sheet_name='SHAP Feature Importance')
    except Exception as e:
        print(f"  Warning: Could not load SHAP sheet: {e}")
        return ww_df

    # Group SHAP by base feature name.
    # Use max (not sum) to avoid inflating importance of correlated aggregations
    # (e.g. vetek_months_std and vetek_months_mean are correlated — summing would
    # double-count their shared signal; max takes the strongest single signal).
    shap_df['Base Feature'] = shap_df['Feature'].apply(_base_name)
    shap_grouped = (
        shap_df.groupby('Base Feature')
        .agg(
            SHAP_Max=('Mean_Abs_SHAP', 'max'),
            SHAP_Components=('Feature', lambda x: ', '.join(x)),
        )
        .reset_index()
        .rename(columns={'Base Feature': 'Feature (Raw)'})
        .sort_values('SHAP_Max', ascending=False)
        .reset_index(drop=True)
    )
    shap_grouped.insert(0, 'SHAP Rank', shap_grouped.index + 1)

    # Outer merge on raw feature name
    merged = pd.merge(ww_df, shap_grouped, on='Feature (Raw)', how='outer')

    # Fill missing ranks/values
    merged['WizWhy Rank'] = merged['WizWhy Rank'].fillna('—')
    merged['SHAP Rank'] = merged['SHAP Rank'].fillna('—')
    merged['Feature (Readable)'] = merged['Feature (Readable)'].fillna(
        merged['Feature (Raw)'].apply(lambda x: FEATURE_DESCRIPTIONS.get(x, x))
    )

    # Sort by SHAP rank first, then WizWhy rank
    merged = merged.sort_values(
        by=['SHAP Rank', 'WizWhy Rank'],
        key=lambda col: col.map(lambda x: x if isinstance(x, (int, float)) else 9999)
    ).reset_index(drop=True)

    cols = [
        'SHAP Rank', 'WizWhy Rank',
        'Feature (Raw)', 'Feature (Readable)',
        'SHAP_Max', 'SHAP_Components',
        'Rule Count', 'Rule Count (Leave)', 'Rule Count (Stay)',
        'Total Records Covered',
        'Avg Probability (weighted)', 'Avg IFF Criterion (weighted)', 'Avg I.F. (weighted)',
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("WizWhy Rules + SHAP Feature Comparison")
    print("=" * 65)

    if not os.path.exists(IFTHEN_PATH):
        print(f"Error: {IFTHEN_PATH} not found.")
        return

    # 1. Parse rules
    print(f"\nParsing {IFTHEN_PATH}...")
    rules = parse_rules(IFTHEN_PATH)
    print(f"  Parsed {len(rules)} rules")

    total_features = sum(len(r['conditions']) for r in rules)
    unique_features = len({c['feature'] for r in rules for c in r['conditions']})
    print(f"  {unique_features} unique features appear across {total_features} condition slots")

    # 2. Aggregate feature stats
    ww_df = aggregate_feature_stats(rules)

    # 3. Build rule detail
    detail_df = build_rule_detail(rules)

    # 4. Merge with SHAP
    comparison_df = merge_with_shap(ww_df, SHAP_RESULTS)

    # 5. Print summary
    n_leave = sum(1 for r in rules if r.get('is_turnover_rule', True))
    n_stay = len(rules) - n_leave
    print(f"\n  Rule breakdown: {n_leave} turnover rules (Leave), {n_stay} retention rules (Stay)")

    print("\n  Top features by WizWhy turnover rule frequency:")
    print(f"  {'WizWhy Rank':<12} {'Feature':<35} {'Leave':<8} {'Stay':<8} {'Rec Covered':<14} {'Avg Prob'}")
    print("  " + "-" * 88)
    for _, row in ww_df.head(20).iterrows():
        name = row['Feature (Readable)']
        if len(name) > 33: name = name[:30] + "..."
        leave_c = int(row['Rule Count (Leave)'])
        stay_c = int(row['Rule Count (Stay)'])
        print(f"  {int(row['WizWhy Rank']):<12} {name:<35} {leave_c:<8} {stay_c:<8} "
              f"{int(row['Total Records Covered']):<14} {row['Avg Probability (weighted)']:.3f}")

    print("\n  SHAP vs WizWhy alignment (SHAP top features):")
    print(f"  {'SHAP Rank':<10} {'WizWhy Rank':<12} {'Feature':<35} {'SHAP Max':<14} {'Leave Rules':<13} {'Stay Rules'}")
    print("  " + "-" * 95)

    shap_rows = comparison_df[comparison_df['SHAP Rank'] != '—'].head(20)
    for _, row in shap_rows.iterrows():
        name = str(row.get('Feature (Readable)', row['Feature (Raw)']))
        if len(name) > 33: name = name[:30] + "..."
        shap_rank = row['SHAP Rank']
        ww_rank = row['WizWhy Rank']
        shap_val = row.get('SHAP_Max', '')
        shap_str = f"{shap_val:.4f}" if isinstance(shap_val, float) else "—"
        leave_r = row.get('Rule Count (Leave)', float('nan'))
        stay_r = row.get('Rule Count (Stay)', float('nan'))
        leave_str = str(int(leave_r)) if (isinstance(leave_r, (int, float)) and not pd.isna(leave_r)) else '—'
        stay_str = str(int(stay_r)) if (isinstance(stay_r, (int, float)) and not pd.isna(stay_r)) else '—'
        print(f"  {str(shap_rank):<10} {str(ww_rank):<12} {name:<35} {shap_str:<14} {leave_str:<13} {stay_str}")

    # 6. Save as text
    os.makedirs("WizWhy", exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("WIZWHY RULES vs SHAP FEATURE COMPARISON\n")
        f.write("=" * 100 + "\n\n")

        # Feature Comparison Table
        f.write("FEATURE COMPARISON (SHAP rank vs WizWhy rule frequency)\n")
        f.write("NOTE: WizWhy Rank based on Leave rules only. SHAP_Max = max |SHAP| across correlated aggregations.\n")
        f.write("-" * 115 + "\n")
        f.write(f"{'SHAP':<8} {'WizWhy':<8} {'Feature':<30} {'SHAP Max':<12} {'Leave':<8} {'Stay':<8} ")
        f.write(f"{'Records':<12} {'Avg Prob':<10}\n")
        f.write("-" * 115 + "\n")

        for _, row in comparison_df.iterrows():
            shap_r = str(row['SHAP Rank']) if row['SHAP Rank'] != '—' else '—'
            ww_r = str(row['WizWhy Rank']) if row['WizWhy Rank'] != '—' else '—'
            feat = str(row['Feature (Readable)'])[:28]
            shap_t = f"{row['SHAP_Max']:.4f}" if 'SHAP_Max' in row and pd.notna(row['SHAP_Max']) else '—'
            leave_r = str(int(row['Rule Count (Leave)'])) if 'Rule Count (Leave)' in row and pd.notna(row['Rule Count (Leave)']) else '—'
            stay_r = str(int(row['Rule Count (Stay)'])) if 'Rule Count (Stay)' in row and pd.notna(row['Rule Count (Stay)']) else '—'
            recs = str(int(row['Total Records Covered'])) if 'Total Records Covered' in row and pd.notna(row['Total Records Covered']) else '—'
            prob = f"{row['Avg Probability (weighted)']:.3f}" if 'Avg Probability (weighted)' in row and pd.notna(row['Avg Probability (weighted)']) else '—'

            f.write(f"{shap_r:<8} {ww_r:<8} {feat:<30} {shap_t:<12} {leave_r:<8} {stay_r:<8} {recs:<12} {prob:<10}\n")

        f.write("\n" + "=" * 100 + "\n\n")

        # Rule Details
        f.write(f"RULE DETAILS ({len(detail_df)} if-then rules)\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Rule':<6} {'Predicts':<10} {'# Cond':<8} {'Conditions (If-And)':<50} {'Prob':<8} {'Records':<10}\n")
        f.write("-" * 110 + "\n")

        for _, row in detail_df.iterrows():
            rule_num = str(int(row['Rule #']))
            predicts = row['Predicts']
            n_cond = str(int(row['# Conditions']))
            conds = row['Conditions'][:48] + "..." if len(row['Conditions']) > 48 else row['Conditions']
            prob = f"{row['Probability']:.3f}" if pd.notna(row['Probability']) else '—'
            recs = str(int(row['Records'])) if pd.notna(row['Records']) else '—'

            f.write(f"{rule_num:<6} {predicts:<10} {n_cond:<8} {conds:<50} {prob:<8} {recs:<10}\n")

    print(f"\nSaved to {OUTPUT_PATH}")
    print("  Feature Comparison: SHAP rank vs WizWhy rule frequency")
    print("  Rule Details: all 1050 if-then rules with conditions and statistics")


if __name__ == "__main__":
    main()
