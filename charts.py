"""
charts.py
---------
Interactive Plotly visualizations for Smart Expense Management Tracker.
Modern styling with clean colors, Indian Rupee tooltips, and responsive layouts.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Palette for consistent, professional presentation
COLOR_INCOME = "#10B981"      # Emerald green
COLOR_EXPENSE = "#EF4444"     # Rose red
COLOR_SAVINGS = "#3B82F6"     # Bright blue
COLOR_BUDGET = "#8B5CF6"      # Violet
COLOR_PREDICTION = "#F59E0B"  # Amber / Gold
CHART_THEME = "plotly_white"


def plot_income_vs_expense(monthly_df):
    """Grouped Bar Chart showing Income vs Expenses per month."""
    if monthly_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Income vs Expenses (No Data)",
            template=CHART_THEME,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No monthly data available yet.", "xref": "paper", "yref": "paper", "showarrow": False}]
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly_df['month_year'],
        y=monthly_df['Income'],
        name="Income",
        marker_color=COLOR_INCOME,
        hovertemplate="<b>%{x}</b><br>Income: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=monthly_df['month_year'],
        y=monthly_df['Expense'],
        name="Expenses",
        marker_color=COLOR_EXPENSE,
        hovertemplate="<b>%{x}</b><br>Expenses: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Monthly Cash Flow: Income vs Expenses</b>",
        barmode='group',
        template=CHART_THEME,
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_category_pie(category_df, title="Category-Wise Expense Breakdown"):
    """Donut chart illustrating spending distribution by category."""
    if category_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (No Data)",
            template=CHART_THEME,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No expenses recorded.", "xref": "paper", "yref": "paper", "showarrow": False}]
        )
        return fig

    fig = go.Figure(data=[go.Pie(
        labels=category_df['category'],
        values=category_df['amount'],
        hole=0.45,
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
        marker=dict(colors=px.colors.qualitative.Prism)
    )])

    fig.update_layout(
        title=f"<b>{title}</b>",
        template=CHART_THEME,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_monthly_trend_with_ml(monthly_df, ml_result=None):
    """
    Plots the monthly historical expense line, fits the Machine Learning
    Linear Regression trend line, and projects the predicted point for next month.
    """
    if monthly_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Expense Trend & ML Forecast (No Data)",
            template=CHART_THEME,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No data available.", "xref": "paper", "yref": "paper", "showarrow": False}]
        )
        return fig

    fig = go.Figure()

    # 1. Historical Actual Expenses
    fig.add_trace(go.Scatter(
        x=monthly_df['month_year'],
        y=monthly_df['total_expense'],
        mode='lines+markers',
        name='Actual Monthly Spending',
        line=dict(color='#2563EB', width=3),
        marker=dict(size=8, color='#1D4ED8'),
        hovertemplate="<b>%{x}</b><br>Actual: ₹%{y:,.2f}<extra></extra>"
    ))

    # 2. If ML trained successfully, overlay Regression Trendline & Next Month Projection
    if ml_result and ml_result.get('success'):
        slope = ml_result['slope']
        intercept = ml_result['intercept']
        pred_val = ml_result['predicted_expense']
        next_month_str = ml_result['next_month_str']

        # Historical fitted points
        x_indices = monthly_df['month_index'].values
        y_fitted = [slope * i + intercept for i in x_indices]

        fig.add_trace(go.Scatter(
            x=monthly_df['month_year'],
            y=y_fitted,
            mode='lines',
            name='ML Linear Trendline (y = mx + c)',
            line=dict(color='#9333EA', width=2, dash='dash'),
            hovertemplate="<b>%{x}</b><br>Trendline: ₹%{y:,.2f}<extra></extra>"
        ))

        # Add connection between last point and prediction
        forecast_x = [monthly_df['month_year'].iloc[-1], f"{next_month_str} (Pred)"]
        forecast_y = [y_fitted[-1], pred_val]

        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines',
            name='Forecast Path',
            line=dict(color=COLOR_PREDICTION, width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Single forecast marker
        fig.add_trace(go.Scatter(
            x=[f"{next_month_str} (Pred)"],
            y=[pred_val],
            mode='markers+text',
            name=f'Predicted Expense: ₹{pred_val:,.0f}',
            marker=dict(size=14, color=COLOR_PREDICTION, symbol='star'),
            text=[f"₹{pred_val:,.0f}"],
            textposition="top center",
            hovertemplate=f"<b>Predicted for {next_month_str}</b><br>Estimated: ₹{pred_val:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="<b>Historical Expense Trend & Machine Learning Forecast</b>",
        template=CHART_THEME,
        hovermode="x unified",
        xaxis_title="Month Timeline",
        yaxis_title="Expense Amount (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_budget_vs_actual(categories, actuals, budgets):
    """
    Comparison Bar Chart: Actual Spending vs Category Budget Limits.
    Helps identify which categories have exceeded or are approaching limits.
    """
    if not categories:
        fig = go.Figure()
        fig.update_layout(
            title="Budget vs Actual Spending (No Budgets Configured)",
            template=CHART_THEME,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No category budgets configured yet.", "xref": "paper", "yref": "paper", "showarrow": False}]
        )
        return fig

    fig = go.Figure()

    # Category Budget Limit Bars
    fig.add_trace(go.Bar(
        x=categories,
        y=budgets,
        name="Budget Limit",
        marker_color="#94A3B8",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Budget: ₹%{y:,.2f}<extra></extra>"
    ))

    # Actual Spending Bars (color coded: red if exceeded, green/blue if within)
    bar_colors = [
        COLOR_EXPENSE if act > bud and bud > 0 else (COLOR_PREDICTION if act >= 0.8 * bud and bud > 0 else COLOR_INCOME)
        for act, bud in zip(actuals, budgets)
    ]

    fig.add_trace(go.Bar(
        x=categories,
        y=actuals,
        name="Actual Spent",
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>Actual Spent: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Category Budget vs Actual Spending Comparison</b>",
        barmode='group',
        template=CHART_THEME,
        xaxis_title="Expense Category",
        yaxis_title="Amount (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_savings_trend(monthly_df):
    """Area / Line chart depicting net monthly savings and savings rate."""
    if monthly_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Savings Trend (No Data)",
            template=CHART_THEME,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": "No savings records found.", "xref": "paper", "yref": "paper", "showarrow": False}]
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_df['month_year'],
        y=monthly_df['Savings'],
        mode='lines+markers',
        name='Net Savings (₹)',
        line=dict(color=COLOR_SAVINGS, width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.15)',
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>Savings: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Monthly Net Savings Trajectory</b>",
        template=CHART_THEME,
        xaxis_title="Month",
        yaxis_title="Savings Amount (₹)",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_payment_methods(df):
    """Donut chart depicting transaction volume per payment method."""
    if df.empty or 'payment_method' not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Payment Methods (No Data)", template=CHART_THEME)
        return fig

    pm_df = df.groupby('payment_method')['amount'].sum().reset_index()
    fig = go.Figure(data=[go.Pie(
        labels=pm_df['payment_method'],
        values=pm_df['amount'],
        hole=0.45,
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Pastel)
    )])
    fig.update_layout(
        title="<b>Spending by Payment Method</b>",
        template=CHART_THEME,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
