"""
app.py
------
Smart Expense Management Tracker using Machine Learning
Frontend Application built with Streamlit, Plotly, SQLite, and Scikit-Learn.
Designed for 1st-Year B.Tech CSE Project Demonstrations.
"""

import os
from datetime import datetime, date
import streamlit as st
import pandas as pd
import numpy as np

# Internal modules
import database as db
import expense_utils as utils
import ml_model as ml
import charts

# Page Configuration
st.set_page_config(
    page_title="Smart Expense Tracker with ML",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Finance UI
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    /* Banner tags */
    .sample-data-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .budget-warning {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #EF4444;
    }
    .budget-caution {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #F59E0B;
    }
    .budget-safe {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Ensure Database is Initialized
db.init_db()

# Expense and Income Categories
EXPENSE_CATEGORIES = [
    "Food",
    "Groceries",
    "Travel",
    "Education",
    "Entertainment",
    "Shopping",
    "Bills",
    "Healthcare",
    "Other"
]

INCOME_CATEGORIES = [
    "Allowance / Pocket Money",
    "Freelance / Internship",
    "Salary",
    "Investment / Interest",
    "Gift",
    "Other"
]

PAYMENT_METHODS = ["UPI", "Cash", "Credit Card", "Debit Card", "Net Banking", "Other"]

# Sidebar Navigation
st.sidebar.markdown("## 💰 Smart Expense Tracker")
st.sidebar.caption("Machine Learning Assisted Personal Finance")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "➕ Add Transaction",
        "📋 Transaction History",
        "🎯 Budget Planner",
        "🤖 ML Expense Prediction",
        "📈 Reports & Insights",
        "⚙️ Settings & Demo Data"
    ]
)

# Sidebar Quick Sample Data Status & Trigger
has_sample = db.has_sample_data()
has_user = db.has_user_data()

st.sidebar.markdown("---")
if has_sample:
    st.sidebar.markdown('<span class="sample-data-badge">⚠️ Demo Sample Data Active</span>', unsafe_allow_html=True)
    if st.sidebar.button("🧹 Clear Sample Data", key="side_clear_sample"):
        db.clear_sample_data()
        st.toast("Sample transactions removed successfully!", icon="✅")
        st.rerun()
else:
    if st.sidebar.button("🚀 Load 6-Month Demo Data", key="side_load_sample"):
        count = db.load_sample_data()
        st.toast(f"Loaded {count} sample transactions!", icon="🎉")
        st.rerun()

# Helper: Generate project zip in memory for direct download
import io
import zipfile

def get_project_zip_bytes():
    """Generates an in-memory zip archive of the entire project source code."""
    zip_buffer = io.BytesIO()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    excluded_dirs = {'.git', '__pycache__', '.venv', 'env', 'venv'}
    excluded_exts = {'.pyc', '.pyo'}
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in excluded_exts or file.endswith('.zip'):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.join("expense-management-tracker", os.path.relpath(full_path, project_dir))
                zf.write(full_path, rel_path)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Sidebar Download Option
st.sidebar.download_button(
    label="📥 Download Project (.zip)",
    data=get_project_zip_bytes(),
    file_name="smart-expense-management-tracker.zip",
    mime="application/zip",
    use_container_width=True,
    help="Download full project source code as a ZIP file"
)

# Helper: Current Month String
CURRENT_MONTH_STR = datetime.now().strftime('%Y-%m')


# ==============================================================================
# 1. DASHBOARD
# ==============================================================================
if nav_choice == "📊 Dashboard":
    st.title("📊 Financial Overview & Dashboard")
    st.markdown("Track your real-time cash flow, budget usage, and Machine Learning projections at a glance.")

    # Load all transactions
    df_all = db.get_transactions()
    total_budget = db.get_monthly_budget(CURRENT_MONTH_STR)
    metrics = utils.calculate_dashboard_metrics(df_all, current_month=CURRENT_MONTH_STR, total_budget=total_budget)

    # Empty State Guide
    if df_all.empty:
        st.info(
            "👋 **No transactions recorded yet!**\n\n"
            "You can either **add your first transaction** from the sidebar or click **'Load 6-Month Demo Data'** "
            "to preview the dashboard with realistic student expense records and trained ML forecasts."
        )
        if st.button("🚀 Load Demo Data Now", type="primary"):
            db.load_sample_data()
            st.rerun()
    else:
        # Budget Warning Banners
        if total_budget and total_budget > 0:
            used_pct = metrics['budget_used_pct']
            if used_pct >= 100:
                st.markdown(
                    f'<div class="budget-warning">🚨 <b>Budget Exceeded!</b> You have spent <b>{utils.format_inr(metrics["curr_expenses"])}</b> '
                    f'against your monthly limit of <b>{utils.format_inr(total_budget)}</b> ({used_pct:.1f}% utilized).</div>',
                    unsafe_allow_html=True
                )
            elif used_pct >= 80:
                st.markdown(
                    f'<div class="budget-caution">🔔 <b>Budget Warning:</b> You have used <b>{used_pct:.1f}%</b> of your monthly budget. '
                    f'Only <b>{utils.format_inr(metrics["budget_remaining"])}</b> remaining for {datetime.now().strftime("%B")}.</div>',
                    unsafe_allow_html=True
                )

        # Row 1: KPI Metric Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Income (All-Time)</div>
                <div class="metric-value" style="color: #10B981;">{utils.format_inr(metrics['total_income'])}</div>
                <div class="metric-sub">This Month: {utils.format_inr(metrics['curr_income'])}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Expenses (All-Time)</div>
                <div class="metric-value" style="color: #EF4444;">{utils.format_inr(metrics['total_expenses'])}</div>
                <div class="metric-sub">This Month: {utils.format_inr(metrics['curr_expenses'])}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Net Balance (Savings)</div>
                <div class="metric-value" style="color: {'#10B981' if metrics['remaining_balance'] >= 0 else '#EF4444'};">
                    {utils.format_inr(metrics['remaining_balance'])}
                </div>
                <div class="metric-sub">Savings Rate: {metrics['savings_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            budget_display = utils.format_inr(total_budget) if total_budget else "Not Set"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{datetime.now().strftime('%B')} Budget Usage</div>
                <div class="metric-value" style="color: #6366F1;">{metrics['budget_used_pct']:.1f}%</div>
                <div class="metric-sub">Cap: {budget_display}</div>
            </div>
            """, unsafe_allow_html=True)

        # Budget Progress Bar
        if total_budget and total_budget > 0:
            prog_val = min(1.0, metrics['budget_used_pct'] / 100.0)
            st.progress(prog_val, text=f"Budget Consumed: {metrics['budget_used_pct']:.1f}% ({utils.format_inr(metrics['curr_expenses'])} / {utils.format_inr(total_budget)})")

        # Row 2: ML Quick Forecast Banner
        ml_res = ml.train_and_predict_expenses(df_all, target_budget=total_budget)
        if ml_res.get('success'):
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #F5F3FF 0%, #EDE9FE 100%); border: 1px solid #C4B5FD; border-radius: 10px; padding: 14px 18px; margin: 15px 0;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #5B21B6;">🤖 Machine Learning Expense Forecast for {ml_res['next_month_str']}:</span>
                <span style="font-size: 1.3rem; font-weight: 800; color: #6D28D9; margin-left: 10px;">{utils.format_inr(ml_res['predicted_expense'])}</span>
                <span style="font-size: 0.85rem; color: #4C1D95; margin-left: 15px;">({ml_res['trend_interpretation']})</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Cash Flow & Category Distribution")
        chart_col1, chart_col2 = st.columns([1.2, 1.0])

        monthly_summary = utils.get_monthly_summary(df_all)
        category_summary = utils.get_category_summary(df_all)

        with chart_col1:
            fig_bar = charts.plot_income_vs_expense(monthly_summary)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            fig_pie = charts.plot_category_pie(category_summary, title="Overall Expense Share")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Row 3: Recent Transactions
        st.markdown("### 🕒 Recent Transactions")
        recent_df = df_all.head(8).copy()
        if not recent_df.empty:
            recent_display = recent_df[['date', 'type', 'category', 'amount', 'payment_method', 'description', 'is_sample']].copy()
            recent_display['date'] = recent_display['date'].dt.strftime('%d %b %Y')
            recent_display['amount'] = recent_display['amount'].apply(utils.format_inr)
            recent_display['is_sample'] = recent_display['is_sample'].apply(lambda x: "Demo Sample" if x == 1 else "User Record")
            recent_display.rename(columns={
                'date': 'Date',
                'type': 'Type',
                'category': 'Category',
                'amount': 'Amount',
                'payment_method': 'Payment Mode',
                'description': 'Description',
                'is_sample': 'Source'
            }, inplace=True)
            st.dataframe(recent_display, use_container_width=True, hide_index=True)


# ==============================================================================
# 2. ADD TRANSACTION
# ==============================================================================
elif nav_choice == "➕ Add Transaction":
    st.title("➕ Record Income or Expense")
    st.markdown("Add a validated financial transaction to update your budget and database immediately.")

    with st.form("add_transaction_form", clear_on_submit=True):
        col_type, col_amt = st.columns(2)
        with col_type:
            tx_type = st.radio("Transaction Type", ["Expense", "Income"], horizontal=True)
        with col_amt:
            amount = st.number_input("Amount (₹) *", min_value=0.0, step=50.0, format="%.2f", help="Must be strictly greater than 0")

        col_cat, col_pay, col_date = st.columns(3)
        with col_cat:
            cat_options = EXPENSE_CATEGORIES if tx_type == "Expense" else INCOME_CATEGORIES
            category = st.selectbox("Category *", cat_options)
        with col_pay:
            payment_method = st.selectbox("Payment Method *", PAYMENT_METHODS)
        with col_date:
            tx_date = st.date_input("Transaction Date *", value=date.today())

        description = st.text_input("Description / Notes (Optional)", placeholder="e.g. Swiggy dinner, Metro recharge, Freelance UI work")

        submitted = st.form_submit_button("💾 Save Transaction", type="primary", use_container_width=True)

        if submitted:
            if amount <= 0:
                st.error("⚠️ Invalid Amount: Transaction amount must be strictly greater than ₹0.00.")
            else:
                try:
                    new_id = db.add_transaction(
                        tx_date=tx_date,
                        tx_type=tx_type,
                        category=category,
                        amount=amount,
                        payment_method=payment_method,
                        description=description,
                        is_sample=0
                    )
                    st.success(f"✅ Successfully recorded **{tx_type}** of **{utils.format_inr(amount)}** under **{category}**! (Ref ID: #{new_id})")
                except Exception as e:
                    st.error(f"Failed to record transaction: {str(e)}")


# ==============================================================================
# 3. TRANSACTION HISTORY
# ==============================================================================
elif nav_choice == "📋 Transaction History":
    st.title("📋 Transaction History & Records")
    st.markdown("Search, filter, edit, or delete transactions. Export your data to CSV anytime.")

    # Search and Filter Controls
    with st.expander("🔍 Search and Filter Options", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            filter_type = st.selectbox("Filter Type", ["All", "Expense", "Income"])
        with f_col2:
            all_cats = ["All"] + sorted(list(set(EXPENSE_CATEGORIES + INCOME_CATEGORIES)))
            filter_cat = st.selectbox("Category", all_cats)
        with f_col3:
            start_d = st.date_input("From Date", value=None)
        with f_col4:
            end_d = st.date_input("To Date", value=None)

        search_query = st.text_input("Search (keyword in description, category, or payment method)", placeholder="Type to search...")

    # Fetch matching transactions
    df_filtered = db.get_transactions(
        start_date=start_d,
        end_date=end_d,
        category=filter_cat,
        tx_type=filter_type,
        search_query=search_query
    )

    col_count, col_exp = st.columns([2, 1])
    with col_count:
        st.caption(f"Showing **{len(df_filtered)}** transaction(s).")
    with col_exp:
        if not df_filtered.empty:
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export as CSV",
                data=csv_data,
                file_name=f"expense_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    if df_filtered.empty:
        st.warning("No transactions match the selected filters.")
    else:
        # Display Table
        display_df = df_filtered.copy()
        display_df['Formatted Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['Formatted Amount'] = display_df['amount'].apply(utils.format_inr)
        display_df['Source Tag'] = display_df['is_sample'].apply(lambda x: "Demo Sample" if x == 1 else "User Record")

        st.dataframe(
            display_df[['id', 'Formatted Date', 'type', 'category', 'Formatted Amount', 'payment_method', 'description', 'Source Tag']].rename(
                columns={
                    'id': 'ID',
                    'Formatted Date': 'Date',
                    'type': 'Type',
                    'category': 'Category',
                    'Formatted Amount': 'Amount',
                    'payment_method': 'Payment Mode',
                    'description': 'Description',
                    'Source Tag': 'Source'
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("✏️ Edit or Delete a Transaction")
        edit_col1, edit_col2 = st.columns([1, 2])

        with edit_col1:
            tx_id_list = df_filtered['id'].tolist()
            selected_id = st.selectbox("Select Transaction ID", tx_id_list)

        if selected_id:
            tx_data = db.get_transaction_by_id(selected_id)
            if tx_data:
                with st.form(f"edit_form_{selected_id}"):
                    st.write(f"Editing Transaction **#{selected_id}**")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        new_type = st.radio("Type", ["Expense", "Income"], index=0 if tx_data['type'] == 'Expense' else 1, horizontal=True)
                        new_amount = st.number_input("Amount (₹)", min_value=0.01, value=float(tx_data['amount']), step=50.0, format="%.2f")
                        cat_list = EXPENSE_CATEGORIES if new_type == 'Expense' else INCOME_CATEGORIES
                        curr_cat_idx = cat_list.index(tx_data['category']) if tx_data['category'] in cat_list else 0
                        new_cat = st.selectbox("Category", cat_list, index=curr_cat_idx)
                    with ec2:
                        parsed_date = datetime.strptime(tx_data['date'], '%Y-%m-%d').date()
                        new_date = st.date_input("Date", value=parsed_date)
                        pay_idx = PAYMENT_METHODS.index(tx_data['payment_method']) if tx_data['payment_method'] in PAYMENT_METHODS else 0
                        new_pm = st.selectbox("Payment Method", PAYMENT_METHODS, index=pay_idx)
                        new_desc = st.text_input("Description", value=tx_data['description'] or "")

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        save_btn = st.form_submit_button("💾 Update Record", type="primary", use_container_width=True)
                    with btn_c2:
                        delete_btn = st.form_submit_button("🗑️ Delete Record", use_container_width=True)

                    if save_btn:
                        db.update_transaction(
                            tx_id=selected_id,
                            tx_date=new_date,
                            tx_type=new_type,
                            category=new_cat,
                            amount=new_amount,
                            payment_method=new_pm,
                            description=new_desc
                        )
                        st.success(f"Transaction #{selected_id} updated successfully!")
                        st.rerun()

                    if delete_btn:
                        db.delete_transaction(selected_id)
                        st.warning(f"Transaction #{selected_id} deleted successfully.")
                        st.rerun()


# ==============================================================================
# 4. BUDGET PLANNER
# ==============================================================================
elif nav_choice == "🎯 Budget Planner":
    st.title("🎯 Monthly Budget Management")
    st.markdown("Set overall monthly budgets and category caps. Monitor real-time spending with color-coded safety progress bars.")

    # Target Month Selector
    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        selected_month_dt = st.date_input("Select Month to Plan", value=date.today())
        month_key = selected_month_dt.strftime('%Y-%m')
        month_label = selected_month_dt.strftime('%B %Y')

    current_budget = db.get_monthly_budget(month_key) or 0.0

    with b_col2:
        with st.form("set_total_budget_form"):
            new_tot = st.number_input(
                f"Total Budget Cap for {month_label} (₹)",
                min_value=0.0,
                value=float(current_budget),
                step=1000.0,
                format="%.2f"
            )
            if st.form_submit_button("Save Total Budget", type="primary"):
                db.set_monthly_budget(month_key, new_tot)
                st.success(f"Updated {month_label} total budget to {utils.format_inr(new_tot)}!")
                st.rerun()

    st.markdown("---")
    st.subheader(f"📊 Budget Consumption for {month_label}")

    # Fetch expenses for selected month
    df_all = db.get_transactions()
    if not df_all.empty:
        df_all['date'] = pd.to_datetime(df_all['date'])
        df_month = df_all[(df_all['date'].dt.strftime('%Y-%m') == month_key) & (df_all['type'] == 'Expense')]
    else:
        df_month = pd.DataFrame()

    total_month_spent = df_month['amount'].sum() if not df_month.empty else 0.0
    saved_category_budgets = db.get_category_budgets(month_key)

    # Overview Metrics
    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        st.metric("Total Monthly Budget", utils.format_inr(current_budget))
    with ov2:
        st.metric("Total Spent", utils.format_inr(total_month_spent))
    with ov3:
        rem = current_budget - total_month_spent if current_budget > 0 else 0.0
        st.metric("Remaining Buffer", utils.format_inr(rem), delta=f"{rem:,.2f}")

    if current_budget > 0:
        total_used_pct = (total_month_spent / current_budget) * 100.0
        st.progress(min(1.0, total_used_pct / 100.0), text=f"Total Budget Utilization: {total_used_pct:.1f}%")
        if total_used_pct >= 100:
            st.error(f"🚨 ALERT: You have exceeded the total monthly budget by {utils.format_inr(total_month_spent - current_budget)}!")
        elif total_used_pct >= 80:
            st.warning(f"⚠️ WARNING: You have consumed {total_used_pct:.1f}% of your budget. Caution is advised for the rest of the month.")
        else:
            st.success("✅ Spending is well within safe monthly budget limits.")

    st.markdown("---")
    st.subheader("🏷️ Category-Wise Budgets & Limits")

    cat_actuals = df_month.groupby('category')['amount'].sum().to_dict() if not df_month.empty else {}

    cat_cols = st.columns(2)
    categories_list = EXPENSE_CATEGORIES

    for i, cat in enumerate(categories_list):
        col_target = cat_cols[i % 2]
        with col_target:
            cat_cap = saved_category_budgets.get(cat, 0.0)
            actual_spent = cat_actuals.get(cat, 0.0)

            with st.container():
                st.markdown(f"**{cat}**")
                c_in1, c_in2 = st.columns([1.5, 1])
                with c_in1:
                    new_cap = st.number_input(
                        f"Limit (₹)",
                        min_value=0.0,
                        value=float(cat_cap),
                        step=500.0,
                        key=f"cap_in_{cat}_{month_key}",
                        label_visibility="collapsed"
                    )
                with c_in2:
                    if st.button("Set Cap", key=f"btn_cap_{cat}_{month_key}"):
                        db.set_category_budget(month_key, cat, new_cap)
                        st.toast(f"Saved {cat} limit: {utils.format_inr(new_cap)}")
                        st.rerun()

                # Progress Bar
                if cat_cap > 0:
                    pct = (actual_spent / cat_cap) * 100.0
                    prog_val = min(1.0, pct / 100.0)
                    st.progress(prog_val, text=f"Spent {utils.format_inr(actual_spent)} of {utils.format_inr(cat_cap)} ({pct:.1f}%)")
                    if pct >= 100:
                        st.markdown(f'<span style="color:#EF4444; font-size:0.8rem; font-weight:700;">🚨 Exceeded by {utils.format_inr(actual_spent - cat_cap)}</span>', unsafe_allow_html=True)
                    elif pct >= 80:
                        st.markdown(f'<span style="color:#F59E0B; font-size:0.8rem; font-weight:700;">⚠️ Near limit ({pct:.0f}%)</span>', unsafe_allow_html=True)
                else:
                    st.caption(f"Spent: {utils.format_inr(actual_spent)} (No limit set)")
                st.write("")

    # Visual Comparison Chart
    st.markdown("### 📊 Category Budget vs Actual Chart")
    plot_cats = []
    plot_acts = []
    plot_buds = []
    for cat in categories_list:
        plot_cats.append(cat)
        plot_acts.append(cat_actuals.get(cat, 0.0))
        plot_buds.append(saved_category_budgets.get(cat, 0.0))

    fig_bud = charts.plot_budget_vs_actual(plot_cats, plot_acts, plot_buds)
    st.plotly_chart(fig_bud, use_container_width=True)


# ==============================================================================
# 5. MACHINE LEARNING EXPENSE PREDICTION
# ==============================================================================
elif nav_choice == "🤖 ML Expense Prediction":
    st.title("🤖 Machine Learning Expense Prediction")
    st.markdown("Forecast next month's total expenses using **Scikit-learn Linear Regression** trained on real historical monthly data.")

    # Educational Expander explaining the algorithm
    with st.expander("📚 Beginner's Guide: How Linear Regression Works in This Project", expanded=False):
        st.markdown(r"""
        ### 🧠 Understanding Machine Learning for 1st-Year CSE
        
        **1. What Model Are We Using?**
        - Algorithm: **Linear Regression** (`sklearn.linear_model.LinearRegression`).
        - Paradigm: **Supervised Learning** (learning patterns from historical inputs to predict future continuous values).

        **2. The Mathematical Equation:**
        $$y = m \cdot X + c$$
        - **$X$ (Feature Matrix)**: The sequential timeline index of months ($1, 2, 3, \dots, N$).
        - **$y$ (Target Vector)**: The total monthly expense amount in Rupees ($\text{₹}$).
        - **$m$ (Slope / Coefficient)**: Represents the spending trend rate.
          - If $m > 0$: Spending is increasing by $₹m$ per month.
          - If $m < 0$: Spending is decreasing by $₹|m|$ per month.
        - **$c$ (Y-Intercept)**: The baseline expense when $X = 0$.

        **3. How We Predict Next Month:**
        If we have $N$ historical months, the next month has feature index $X = N + 1$.
        $$\hat{y}_{\text{next}} = m \cdot (N + 1) + c$$

        **4. Evaluation Metrics:**
        - **$R^2$ Score (Coefficient of Determination)**: Indicates the proportion of variance in spending explained by the linear model.
        - **MAE (Mean Absolute Error)**: The average error magnitude in Rupees ($\text{₹}$).
        """)

    df_all = db.get_transactions()
    total_budget = db.get_monthly_budget(CURRENT_MONTH_STR)

    # Train and Run Real ML Model
    ml_result = ml.train_and_predict_expenses(df_all, target_budget=total_budget)

    # Handle Edge Cases Gracefully
    if ml_result['status_code'] == 'NO_DATA':
        st.warning("⚠️ **No expense data found.** Add transactions or load demo data to train the ML model.")
        if st.button("🚀 Load 6-Month Demo Data"):
            db.load_sample_data()
            st.rerun()

    elif ml_result['status_code'] == 'INSUFFICIENT_DATA':
        st.warning(
            f"⚠️ **{ml_result['message']}**\n\n"
            f"**Why this limitation exists:** Linear Regression requires at least 3 distinct historical months "
            f"to establish a reliable mathematical trendline. With only {ml_result['historical_count']} month(s), "
            f"fitting a line would lead to severe overfitting."
        )

        st.info(
            f"💡 **Fallback Baseline (Simple Historical Average):** {utils.format_inr(ml_result['fallback_avg'])}\n"
            f"*(Note: This is an empirical average, not an ML regression model).* "
        )

        st.markdown("#### 🎯 College Viva Demonstration Option:")
        st.write("Click below to immediately load 6 realistic historical months (May - Oct 2026) to demonstrate the live Scikit-learn model to your professor:")
        if st.button("🚀 Load 6-Month Dataset for Live ML Demo", type="primary"):
            db.load_sample_data()
            st.rerun()

    elif ml_result['status_code'] == 'SUCCESS':
        # Success state: Full ML visualization and metrics
        st.success(f"✅ **{ml_result['message']}**")

        # Top Highlight: Prediction Card
        p_col1, p_col2 = st.columns([1.3, 1])

        with p_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); color: white; border-radius: 14px; padding: 24px; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);">
                <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Projected Spending for {ml_result['next_month_str']}</div>
                <div style="font-size: 2.5rem; font-weight: 800; margin: 8px 0;">{utils.format_inr(ml_result['predicted_expense'])}</div>
                <div style="font-size: 0.95rem; opacity: 0.95;">{ml_result['trend_interpretation']}</div>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown("#### 🎯 Budget Alignment Analysis")
            b_comp = ml_result['budget_comparison']
            if b_comp['status'] == 'EXCEEDS_BUDGET':
                st.error(b_comp['alert'])
            elif b_comp['status'] == 'WITHIN_BUDGET':
                st.success(b_comp['alert'])
            else:
                st.info(b_comp['alert'])

            st.caption(f"Monthly Budget Reference: {utils.format_inr(b_comp.get('budget', 0.0))}")

        st.markdown("---")

        # Row 2: Regression Metrics & Formula
        st.subheader("📐 Model Parameters & Mathematical Equation")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric("Regression Formula", "y = mx + c", help="Standard linear slope-intercept equation")
            st.caption(f"`{ml_result['formula_display']}`")

        with m_col2:
            st.metric("Slope (m)", f"{ml_result['slope']:+,.2f}", help="Monthly rate of expense change in ₹")
            st.caption("Rate of spending growth per month")

        with m_col3:
            st.metric("R² Fit Score", f"{ml_result['r2']:.3f}", help="Variance explained by trend (0.0 to 1.0)")
            st.caption("Closer to 1.0 indicates stronger linear trend")

        with m_col4:
            st.metric("Mean Absolute Error (MAE)", utils.format_inr(ml_result['mae']), help="Average deviation from actual historical expenses")
            st.caption("Average error in Rupees")

        # Row 3: Interactive Plotly Regression Chart
        st.markdown("---")
        st.subheader("📈 Historical Expenses, ML Trendline, and Future Forecast")
        fig_trend = charts.plot_monthly_trend_with_ml(ml_result['monthly_data'], ml_result)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Row 4: Historical Training Table
        st.markdown("#### 📋 Historical Training Points")
        hist_table = ml_result['monthly_data'][['month_year', 'month_index', 'total_expense']].copy()
        hist_table['Fitted Regression (₹)'] = hist_table['month_index'].apply(
            lambda idx: round(ml_result['slope'] * idx + ml_result['intercept'], 2)
        )
        hist_table['Residual / Error (₹)'] = hist_table['total_expense'] - hist_table['Fitted Regression (₹)']
        hist_table['total_expense'] = hist_table['total_expense'].apply(utils.format_inr)
        hist_table['Fitted Regression (₹)'] = hist_table['Fitted Regression (₹)'].apply(utils.format_inr)
        hist_table['Residual / Error (₹)'] = hist_table['Residual / Error (₹)'].apply(utils.format_inr)
        hist_table.rename(columns={
            'month_year': 'Month',
            'month_index': 'Feature X (Month Index)',
            'total_expense': 'Target y (Actual Expense)'
        }, inplace=True)
        st.dataframe(hist_table, use_container_width=True, hide_index=True)

        # Viva / Presentation Preparation Guide
        with st.expander("🎓 Questions Your Professor Might Ask in Viva (With Answers!)"):
            st.markdown("""
            **Q1: Why did you choose Linear Regression instead of complex Deep Learning?**
            *Answer:* For monthly personal expense forecasting with limited monthly data points (5-12 months), Linear Regression is optimal because it avoids severe overfitting, is mathematically interpretable ($y = mx + c$), and runs instantaneously without requiring massive neural network parameters.

            **Q2: What is the input Feature (X) and Target (y)?**
            *Answer:* The feature $X$ is the chronological monthly index ($1, 2, 3, \dots, N$) representing time progression. The target $y$ is the aggregated monthly expense in Rupees ($\text{₹}$).

            **Q3: What are the limitations of this model?**
            *Answer:* 
            1. Assumes a continuous linear trajectory, meaning it does not capture seasonal spikes (e.g. Diwali shopping, semester fee months).
            2. Sensitive to one-time outlier emergencies (e.g. laptop repair or hospital bills).
            3. In future work, we can incorporate category-level classification and ARIMA / Prophet time-series models.
            """)


# ==============================================================================
# 6. REPORTS AND INSIGHTS
# ==============================================================================
elif nav_choice == "📈 Reports & Insights":
    st.title("📈 Financial Reports & Actionable Insights")
    st.markdown("Comprehensive analytics, category breakdowns, savings rates, and automated financial health tips.")

    df_all = db.get_transactions()
    total_budget = db.get_monthly_budget(CURRENT_MONTH_STR)

    if df_all.empty:
        st.info("No transaction data available. Add transactions or load demo data to see analytical reports.")
    else:
        insights = utils.generate_spending_insights(df_all, current_month=CURRENT_MONTH_STR, total_budget=total_budget)

        # Highlight Metrics
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Top Spending Category", insights['top_category'] or "N/A")
            st.caption(f"{utils.format_inr(insights['top_category_amount'])} ({insights['top_category_pct']:.1f}% share)")
        with h2:
            st.metric("Avg Monthly Expense", utils.format_inr(insights['avg_monthly_expense']))
            st.caption("Across all recorded months")
        with h3:
            st.metric("Peak Expense Month", insights['peak_month'] or "N/A")
            st.caption(f"Outflow: {utils.format_inr(insights['peak_month_expense'])}")
        with h4:
            monthly_sum = utils.get_monthly_summary(df_all)
            avg_sav = monthly_sum['Savings_Rate'].mean() if not monthly_sum.empty else 0.0
            st.metric("Avg Savings Rate", f"{avg_sav:.1f}%")
            st.caption("Portion of income saved")

        st.markdown("---")

        # Rule-based suggestions section
        st.subheader("💡 Rule-Based Smart Financial Insights")
        st.caption("Automated suggestions generated from your actual transaction patterns:")
        for rec in insights['recommendations']:
            st.markdown(f"- {rec}")

        st.markdown("---")

        # Visual Analytics Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Category Breakdown", "💰 Savings Trajectory", "💳 Payment Methods"])

        with tab1:
            cat_df = utils.get_category_summary(df_all)
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.dataframe(
                    cat_df.rename(columns={'category': 'Category', 'amount': 'Total Spent (₹)', 'percentage': 'Share (%)'}).assign(
                        **{'Total Spent (₹)': cat_df['amount'].apply(utils.format_inr)}
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            with c2:
                fig_p = charts.plot_category_pie(cat_df, title="All-Time Category Distribution")
                st.plotly_chart(fig_p, use_container_width=True)

        with tab2:
            fig_sav = charts.plot_savings_trend(monthly_sum)
            st.plotly_chart(fig_sav, use_container_width=True)

        with tab3:
            fig_pm = charts.plot_payment_methods(df_all)
            st.plotly_chart(fig_pm, use_container_width=True)


# ==============================================================================
# 7. SETTINGS AND DEMO DATA
# ==============================================================================
elif nav_choice == "⚙️ Settings & Demo Data":
    st.title("⚙️ Application Settings & Demo Data")
    st.markdown("Manage sample datasets, database backups, and project settings.")

    st.subheader("🧪 Sample Data Demonstration")
    st.markdown(
        "For viva presentations and testing, you can populate the database with **6 consecutive months of realistic student transactions** "
        "(May 2026 to October 2026). All demo records are strictly tagged (`is_sample = 1`) so they can be removed cleanly at any time."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("🚀 Load 6-Month Demo Transactions & Budgets", type="primary", use_container_width=True):
            count = db.load_sample_data()
            st.success(f"Successfully loaded {count} realistic transactions across 6 months with pre-configured budgets!")
            st.rerun()

    with s_col2:
        if st.button("🧹 Clear Demo Sample Data Only", use_container_width=True):
            deleted = db.clear_sample_data()
            st.info(f"Removed {deleted} demo transaction(s). Any personal records you added were preserved.")
            st.rerun()

    st.markdown("---")
    st.subheader("⚠️ Complete Database Reset")
    st.markdown("Permanently erase all income, expense, and budget records to return to a clean state.")
    
    with st.expander("Danger Zone: Erase Everything"):
        st.write("This action cannot be undone.")
        if st.button("💥 Reset & Empty Entire Database", type="secondary"):
            db.clear_all_data()
            st.warning("Database has been completely cleared.")
            st.rerun()

    st.markdown("---")
    st.subheader("💾 Database Information")
    db_path = db.get_db_path()
    st.code(f"SQLite Database File Path:\n{db_path}", language="bash")

    all_tx = db.get_transactions()
    total_count = len(all_tx)
    sample_count = len(all_tx[all_tx['is_sample'] == 1]) if not all_tx.empty else 0
    user_count = total_count - sample_count

    st.write(f"- **Total Transactions:** {total_count}")
    st.write(f"- **User Entered Records:** {user_count}")
    st.write(f"- **Demo Sample Records:** {sample_count}")

    st.markdown("---")
    st.subheader("📦 Download Complete Project Code (.zip)")
    st.markdown(
        "Download the entire application package (including SQLite database scripts, ML models, "
        "Plotly charts, and test suite) as a portable ZIP file for college submission or offline backup."
    )
    st.download_button(
        label="📥 Download smart-expense-management-tracker.zip",
        data=get_project_zip_bytes(),
        file_name="smart-expense-management-tracker.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("👨‍💻 Academic Project Information")
    st.markdown("""
    - **Project Name:** Smart Expense Management Tracker using Machine Learning
    - **Academic Level:** 1st Year B.Tech Computer Science and Engineering
    - **Model Implemented:** Scikit-learn Linear Regression ($y = mx + c$)
    - **Frontend:** Streamlit
    - **Database:** SQLite3
    - **Visualization:** Plotly
    """)
