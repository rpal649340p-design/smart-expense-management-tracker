"""
expense_utils.py
----------------
Utility functions for calculating financial metrics, aggregations,
formatting Indian Rupee amounts, and generating rule-based insights.
"""

from datetime import datetime
import pandas as pd
import numpy as np


def format_inr(number):
    """
    Formats a numeric amount into Indian Rupee (₹) format with comma grouping.
    Example: 125000 -> ₹1,25,000.00
    """
    if number is None or pd.isna(number):
        return "₹0.00"
    
    val = float(number)
    sign = "-" if val < 0 else ""
    val = abs(val)

    parts = f"{val:.2f}".split(".")
    integer_part = parts[0]
    decimal_part = parts[1]

    # Standard Indian numbering formatting
    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        # Group in pairs of 2 from right
        chunks = []
        while len(remaining) > 2:
            chunks.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            chunks.insert(0, remaining)
        formatted_int = ",".join(chunks) + "," + last_three
    else:
        formatted_int = integer_part

    return f"{sign}₹{formatted_int}.{decimal_part}"


def format_inr_short(number):
    """Returns a short formatted INR string, e.g. ₹12.5k or ₹1.2L."""
    if number is None or pd.isna(number):
        return "₹0"
    val = float(number)
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    
    if abs_val >= 10000000:
        return f"{sign}₹{abs_val / 10000000:.2f} Cr"
    elif abs_val >= 100000:
        return f"{sign}₹{abs_val / 100000:.2f} Lakh"
    elif abs_val >= 1000:
        return f"{sign}₹{abs_val / 1000:.1f}k"
    else:
        return f"{sign}₹{abs_val:.0f}"


def get_monthly_summary(df):
    """
    Aggregates transactions into monthly totals for both Income and Expenses.
    Returns a sorted DataFrame with columns:
    ['month_year', 'year_month_dt', 'Income', 'Expense', 'Savings', 'Savings_Rate']
    """
    if df.empty:
        return pd.DataFrame(columns=['month_year', 'year_month_dt', 'Income', 'Expense', 'Savings', 'Savings_Rate'])

    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['month_year'] = data['date'].dt.strftime('%Y-%m')
    data['year_month_dt'] = data['date'].dt.to_period('M').dt.to_timestamp()

    # Pivot table to aggregate Income and Expense per month
    pivot = data.pivot_table(
        index=['month_year', 'year_month_dt'],
        columns='type',
        values='amount',
        aggfunc='sum',
        fill_value=0.0
    ).reset_index()

    # Ensure both columns exist even if one type hasn't occurred yet
    if 'Income' not in pivot.columns:
        pivot['Income'] = 0.0
    if 'Expense' not in pivot.columns:
        pivot['Expense'] = 0.0

    pivot['Savings'] = pivot['Income'] - pivot['Expense']
    pivot['Savings_Rate'] = pivot.apply(
        lambda row: (row['Savings'] / row['Income'] * 100.0) if row['Income'] > 0 else 0.0,
        axis=1
    )

    pivot = pivot.sort_values(by='year_month_dt').reset_index(drop=True)
    return pivot


def get_category_summary(df, month_year=None):
    """
    Computes total expenses per category and their percentage share.
    Optionally filters for a specific 'YYYY-MM' month.
    """
    if df.empty:
        return pd.DataFrame(columns=['category', 'amount', 'percentage'])

    data = df[df['type'] == 'Expense'].copy()
    if data.empty:
        return pd.DataFrame(columns=['category', 'amount', 'percentage'])

    if month_year:
        data['date'] = pd.to_datetime(data['date'])
        data['my'] = data['date'].dt.strftime('%Y-%m')
        data = data[data['my'] == month_year]

    if data.empty:
        return pd.DataFrame(columns=['category', 'amount', 'percentage'])

    cat_df = data.groupby('category')['amount'].sum().reset_index()
    total = cat_df['amount'].sum()
    cat_df['percentage'] = (cat_df['amount'] / total * 100.0).round(1)
    cat_df = cat_df.sort_values(by='amount', ascending=False).reset_index(drop=True)
    return cat_df


def calculate_dashboard_metrics(df, current_month=None, total_budget=None):
    """
    Computes all core dashboard metrics:
    - All-time totals: Income, Expenses, Remaining Balance, Overall Savings Rate
    - Current month totals: Income, Expenses, Savings, Budget utilization %
    """
    if current_month is None:
        current_month = datetime.now().strftime('%Y-%m')

    if df.empty:
        return {
            'total_income': 0.0,
            'total_expenses': 0.0,
            'remaining_balance': 0.0,
            'savings_rate': 0.0,
            'curr_income': 0.0,
            'curr_expenses': 0.0,
            'curr_savings': 0.0,
            'curr_savings_rate': 0.0,
            'budget': total_budget or 0.0,
            'budget_used_pct': 0.0,
            'budget_remaining': total_budget or 0.0,
            'status': 'Empty Database'
        }

    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['my'] = data['date'].dt.strftime('%Y-%m')

    # All-time metrics
    total_income = data[data['type'] == 'Income']['amount'].sum()
    total_expenses = data[data['type'] == 'Expense']['amount'].sum()
    remaining_balance = total_income - total_expenses
    savings_rate = (remaining_balance / total_income * 100.0) if total_income > 0 else 0.0

    # Current month metrics
    curr_data = data[data['my'] == current_month]
    curr_income = curr_data[curr_data['type'] == 'Income']['amount'].sum()
    curr_expenses = curr_data[curr_data['type'] == 'Expense']['amount'].sum()
    curr_savings = curr_income - curr_expenses
    curr_savings_rate = (curr_savings / curr_income * 100.0) if curr_income > 0 else 0.0

    # Budget metrics
    budget = total_budget if total_budget is not None else 0.0
    if budget > 0:
        budget_used_pct = (curr_expenses / budget) * 100.0
        budget_remaining = budget - curr_expenses
    else:
        budget_used_pct = 0.0
        budget_remaining = 0.0

    return {
        'total_income': float(total_income),
        'total_expenses': float(total_expenses),
        'remaining_balance': float(remaining_balance),
        'savings_rate': float(savings_rate),
        'curr_income': float(curr_income),
        'curr_expenses': float(curr_expenses),
        'curr_savings': float(curr_savings),
        'curr_savings_rate': float(curr_savings_rate),
        'budget': float(budget),
        'budget_used_pct': float(budget_used_pct),
        'budget_remaining': float(budget_remaining),
        'status': 'Active'
    }


def generate_spending_insights(df, current_month=None, total_budget=None):
    """
    Generates rule-based personal finance insights:
    - Highest spending category
    - Average monthly expenses
    - Month with peak spending
    - Rising expense categories (comparing current month to previous month)
    - Actionable suggestions
    """
    insights = {
        'top_category': None,
        'top_category_amount': 0.0,
        'top_category_pct': 0.0,
        'avg_monthly_expense': 0.0,
        'peak_month': None,
        'peak_month_expense': 0.0,
        'rising_categories': [],
        'recommendations': []
    }

    if df.empty:
        insights['recommendations'].append(
            "Welcome! No transactions recorded yet. Add your first income or expense to unlock real-time financial analytics."
        )
        return insights

    expense_df = df[df['type'] == 'Expense'].copy()
    if expense_df.empty:
        insights['recommendations'].append("No expenses recorded yet. Your income is 100% saved!")
        return insights

    expense_df['date'] = pd.to_datetime(expense_df['date'])
    expense_df['my'] = expense_df['date'].dt.strftime('%Y-%m')

    # 1. Top Category overall
    cat_summary = expense_df.groupby('category')['amount'].sum().reset_index()
    if not cat_summary.empty:
        top_cat_row = cat_summary.sort_values(by='amount', ascending=False).iloc[0]
        insights['top_category'] = top_cat_row['category']
        insights['top_category_amount'] = float(top_cat_row['amount'])
        total_exp = float(cat_summary['amount'].sum())
        insights['top_category_pct'] = (insights['top_category_amount'] / total_exp * 100.0) if total_exp > 0 else 0.0

    # 2. Monthly aggregations
    monthly_exp = expense_df.groupby('my')['amount'].sum().reset_index()
    if not monthly_exp.empty:
        insights['avg_monthly_expense'] = float(monthly_exp['amount'].mean())
        peak_row = monthly_exp.sort_values(by='amount', ascending=False).iloc[0]
        insights['peak_month'] = peak_row['my']
        insights['peak_month_expense'] = float(peak_row['amount'])

    # 3. Rising categories (Current month vs Previous month)
    months = sorted(expense_df['my'].unique())
    if len(months) >= 2:
        curr_m = current_month if current_month in months else months[-1]
        prev_idx = months.index(curr_m) - 1
        if prev_idx >= 0:
            prev_m = months[prev_idx]
            curr_cat = expense_df[expense_df['my'] == curr_m].groupby('category')['amount'].sum()
            prev_cat = expense_df[expense_df['my'] == prev_m].groupby('category')['amount'].sum()

            for cat in curr_cat.index:
                if cat in prev_cat:
                    diff = curr_cat[cat] - prev_cat[cat]
                    pct_increase = (diff / prev_cat[cat]) * 100.0
                    if pct_increase > 15 and diff > 300:
                        insights['rising_categories'].append({
                            'category': cat,
                            'increase_amount': float(diff),
                            'pct_increase': float(pct_increase),
                            'prev_amount': float(prev_cat[cat]),
                            'curr_amount': float(curr_cat[cat])
                        })

    # 4. Actionable rule-based suggestions
    recs = []

    # Category proportion checks
    if insights['top_category_pct'] > 35:
        recs.append(
            f"⚠️ **Concentrated Outflow**: **{insights['top_category']}** accounts for **{insights['top_category_pct']:.1f}%** of your total expenses ({format_inr(insights['top_category_amount'])}). Setting a stricter ceiling on this category will yield the biggest savings."
        )

    # Budget alert checks
    if total_budget and total_budget > 0:
        curr_m_total = expense_df[expense_df['my'] == current_month]['amount'].sum() if current_month else 0.0
        used_pct = (curr_m_total / total_budget) * 100.0
        if used_pct >= 100:
            recs.append(
                f"🚨 **Budget Exceeded**: You have exhausted {used_pct:.1f}% of your monthly budget ({format_inr(curr_m_total)} / {format_inr(total_budget)}). Pause non-essential purchases immediately."
            )
        elif used_pct >= 80:
            recs.append(
                f"🔔 **Budget Warning**: You have used {used_pct:.1f}% of your monthly budget. Only {format_inr(total_budget - curr_m_total)} remains for the rest of the month."
            )
        else:
            recs.append(
                f"✅ **Healthy Budget Pace**: You are currently within budget ({used_pct:.1f}% utilized). You have {format_inr(total_budget - curr_m_total)} available."
            )

    # Food and dining specific student tip
    food_exp = expense_df[expense_df['category'] == 'Food']['amount'].sum()
    all_exp = expense_df['amount'].sum()
    if all_exp > 0 and (food_exp / all_exp) > 0.28:
        recs.append(
            f"💡 **Student Tip - Dining**: Food spending is {((food_exp / all_exp) * 100):.1f}% of your outgoings ({format_inr(food_exp)}). Replacing 2 takeout meals per week with college canteen or hostel mess can save around ₹1,500 - ₹2,500 monthly."
        )

    # Rising categories advice
    if insights['rising_categories']:
        top_rising = sorted(insights['rising_categories'], key=lambda x: x['increase_amount'], reverse=True)[0]
        recs.append(
            f"📈 **Spike Detected**: Spending on **{top_rising['category']}** surged by **+{top_rising['pct_increase']:.0f}%** (+{format_inr(top_rising['increase_amount'])}) compared to last month."
        )

    # 50/30/20 guideline tip
    recs.append(
        "🎯 **Recommended 50/30/20 Framework**: Allocate 50% of income to Needs (Groceries, Bills, Travel), 30% to Wants (Entertainment, Shopping), and 20% to Savings/Investments."
    )

    insights['recommendations'] = recs
    return insights
