"""
ml_model.py
-----------
Real Machine Learning Module for Monthly Expense Prediction using Scikit-learn LinearRegression.

Beginner-Friendly Explanation for 1st Year B.Tech CSE:
-----------------------------------------------------
1. What is Machine Learning here?
   We use Supervised Learning -> Linear Regression.
   - Independent Variable (Feature X): The sequential month number (1, 2, 3, ... N).
   - Dependent Variable (Target y): Total expenses incurred in that month (₹).

2. Mathematical Equation:
   y = m * X + c
   - m (Slope / Coefficient): Represents the monthly rate of change in your spending.
     If m > 0, expenses are trending upwards by ₹m per month.
     If m < 0, expenses are decreasing by ₹|m| per month.
   - c (Intercept): The estimated baseline expense level.

3. Prediction:
   For the upcoming month (N + 1), the model calculates:
   Predicted_Expense = m * (N + 1) + c

4. Evaluation Metrics:
   - R² Score (Coefficient of Determination): Measures how well the linear line fits historical points (max 1.0).
   - MAE (Mean Absolute Error): Average error in Rupees between historical actuals and fitted values.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "linear_regression_expense_model.joblib")


def prepare_monthly_expense_features(df):
    """
    Extracts chronological monthly expense totals from transaction records.
    Returns:
    - monthly_df: DataFrame with ['month_year', 'year_month_dt', 'month_index', 'total_expense']
    """
    if df.empty:
        return pd.DataFrame(columns=['month_year', 'year_month_dt', 'month_index', 'total_expense'])

    # Filter for expenses only
    expense_data = df[df['type'] == 'Expense'].copy()
    if expense_data.empty:
        return pd.DataFrame(columns=['month_year', 'year_month_dt', 'month_index', 'total_expense'])

    expense_data['date'] = pd.to_datetime(expense_data['date'])
    expense_data['month_year'] = expense_data['date'].dt.strftime('%Y-%m')
    expense_data['year_month_dt'] = expense_data['date'].dt.to_period('M').dt.to_timestamp()

    # Group by month
    monthly = expense_data.groupby(['month_year', 'year_month_dt'])['amount'].sum().reset_index()
    monthly = monthly.sort_values(by='year_month_dt').reset_index(drop=True)
    monthly.rename(columns={'amount': 'total_expense'}, inplace=True)

    # 1-indexed sequential month feature: 1, 2, 3, ...
    monthly['month_index'] = np.arange(1, len(monthly) + 1)
    return monthly


def train_and_predict_expenses(df, target_budget=None):
    """
    Trains a Scikit-learn LinearRegression model on historical monthly expenses
    and predicts next month's spending.

    Returns a comprehensive result dictionary:
    - success (bool)
    - status_code: 'INSUFFICIENT_DATA', 'NO_DATA', or 'SUCCESS'
    - message (str)
    - monthly_data (DataFrame)
    - model (LinearRegression instance or None)
    - slope (m)
    - intercept (c)
    - r2 (float)
    - mae (float)
    - predicted_expense (float)
    - next_month_str (str)
    - budget_comparison (dict)
    - trend_interpretation (str)
    - formula_display (str)
    """
    monthly_df = prepare_monthly_expense_features(df)
    total_months = len(monthly_df)

    # Edge Case 1: No expense data at all
    if total_months == 0:
        return {
            'success': False,
            'status_code': 'NO_DATA',
            'message': 'No expense transactions available to train the Machine Learning model.',
            'monthly_data': monthly_df,
            'model': None,
            'predicted_expense': None,
            'historical_count': 0
        }

    # Edge Case 2: Insufficient historical months for a valid machine learning regression
    # Linear Regression strictly requires at least 3 distinct time steps for meaningful trend estimation.
    if total_months < 3:
        avg_val = float(monthly_df['total_expense'].mean())
        return {
            'success': False,
            'status_code': 'INSUFFICIENT_DATA',
            'message': (
                f"Insufficient historical data: The model requires at least 3 distinct months of expenses "
                f"to discover spending patterns and calculate a regression slope. Currently, only {total_months} "
                f"month(s) exist."
            ),
            'monthly_data': monthly_df,
            'model': None,
            'historical_count': total_months,
            'fallback_avg': avg_val,
            'fallback_label': "Simple Historical Average Fallback (Not ML)",
            'predicted_expense': avg_val
        }

    # Step 1: Feature Matrix X and Target Vector y
    # X must be 2D array: shape (n_samples, n_features) -> e.g. [[1], [2], [3], ...]
    X = monthly_df[['month_index']].values
    y = monthly_df['total_expense'].values

    # Step 2: Initialize and Train Scikit-Learn LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Step 3: Compute Fitted Values and Evaluation Metrics
    y_pred_hist = model.predict(X)
    r2 = r2_score(y, y_pred_hist)
    mae = mean_absolute_error(y, y_pred_hist)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Step 4: Determine Next Month Index and Predict
    next_month_idx = total_months + 1
    predicted_val = float(model.predict([[next_month_idx]])[0])

    # Ensure prediction doesn't fall below zero in extreme downward trends
    predicted_val = max(0.0, round(predicted_val, 2))

    # Calculate next month label (e.g. if last was 2026-10, next is 2026-11)
    last_dt = monthly_df['year_month_dt'].iloc[-1]
    # Add approx 32 days and snap to beginning of month
    next_dt = (last_dt + pd.DateOffset(months=1))
    next_month_str = next_dt.strftime('%B %Y')

    # Step 5: Save trained model artifact to models/
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        joblib.dump(model, MODEL_FILE_PATH)
    except Exception as e:
        print(f"Notice: Model persistence skipped: {e}")

    # Step 6: Trend Interpretation in Plain English
    if slope > 50:
        trend_interpretation = f"Spending is trending upward by an average of ₹{abs(slope):.2f} per month."
        trend_direction = "UPWARD"
    elif slope < -50:
        trend_interpretation = f"Spending is trending downward by an average of ₹{abs(slope):.2f} per month. Great job cutting costs!"
        trend_direction = "DOWNWARD"
    else:
        trend_interpretation = "Spending has remained relatively stable across the observed months."
        trend_direction = "STABLE"

    # Step 7: Budget Comparison
    budget_comparison = {}
    if target_budget and target_budget > 0:
        diff = predicted_val - target_budget
        budget_comparison['budget'] = float(target_budget)
        budget_comparison['diff'] = float(diff)
        budget_comparison['exceeds'] = diff > 0
        if diff > 0:
            budget_comparison['status'] = 'EXCEEDS_BUDGET'
            budget_comparison['alert'] = (
                f"⚠️ Projected spending (₹{predicted_val:,.2f}) is likely to exceed your monthly budget "
                f"(₹{target_budget:,.2f}) by ₹{diff:,.2f}."
            )
        else:
            budget_comparison['status'] = 'WITHIN_BUDGET'
            budget_comparison['alert'] = (
                f"✅ Projected spending (₹{predicted_val:,.2f}) is expected to stay comfortably within your "
                f"monthly budget (₹{target_budget:,.2f}) with ₹{abs(diff):,.2f} to spare."
            )
    else:
        budget_comparison['status'] = 'NO_BUDGET_SET'
        budget_comparison['alert'] = "No monthly budget set. Set a budget in the Budget Planner to compare against predictions."

    # Formula string representation for educational viva
    sign_str = "+" if slope >= 0 else "-"
    formula_display = f"y = ({sign_str}₹{abs(slope):.2f} × X) + ₹{intercept:.2f}"

    return {
        'success': True,
        'status_code': 'SUCCESS',
        'message': f"Model successfully trained on {total_months} historical months using Scikit-learn LinearRegression.",
        'monthly_data': monthly_df,
        'model': model,
        'historical_count': total_months,
        'slope': slope,
        'intercept': intercept,
        'r2': max(0.0, float(r2)),
        'mae': float(mae),
        'predicted_expense': predicted_val,
        'next_month_str': next_month_str,
        'next_month_idx': next_month_idx,
        'budget_comparison': budget_comparison,
        'trend_interpretation': trend_interpretation,
        'trend_direction': trend_direction,
        'formula_display': formula_display
    }
