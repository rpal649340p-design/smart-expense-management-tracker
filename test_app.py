"""
test_app.py
-----------
Comprehensive automated test suite for Smart Expense Management Tracker.
Tests database operations, input validation, calculations, ML training, and visualizations.
"""

import os
import unittest
import tempfile
import pandas as pd
from datetime import date

# Import modules to test
import database as db
import expense_utils as utils
import ml_model as ml
import charts


class TestExpenseTracker(unittest.TestCase):

    def setUp(self):
        """Create a temporary database file for isolated testing."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        db.init_db(self.db_path)

    def tearDown(self):
        """Clean up the temporary database file after tests."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_database_init(self):
        """Verify tables are created in empty database."""
        df = db.get_transactions(db_path=self.db_path)
        self.assertTrue(df.empty)
        self.assertEqual(len(df), 0)

    def test_add_and_get_transaction(self):
        """Verify adding income and expense transactions."""
        tx1 = db.add_transaction(
            tx_date="2026-10-01",
            tx_type="Income",
            category="Allowance / Pocket Money",
            amount=20000.0,
            payment_method="Net Banking",
            description="Monthly allowance",
            db_path=self.db_path
        )
        self.assertGreater(tx1, 0)

        tx2 = db.add_transaction(
            tx_date="2026-10-05",
            tx_type="Expense",
            category="Food",
            amount=1500.0,
            payment_method="UPI",
            description="Hostel mess food",
            db_path=self.db_path
        )
        self.assertGreater(tx2, 0)

        df = db.get_transactions(db_path=self.db_path)
        self.assertEqual(len(df), 2)

        # Verify details
        t2_dict = db.get_transaction_by_id(tx2, db_path=self.db_path)
        self.assertEqual(t2_dict['category'], "Food")
        self.assertEqual(float(t2_dict['amount']), 1500.0)

    def test_invalid_transaction_validation(self):
        """Ensure negative or zero amounts are rejected."""
        with self.assertRaises(ValueError):
            db.add_transaction("2026-10-01", "Expense", "Food", -500.0, "UPI", db_path=self.db_path)

        with self.assertRaises(ValueError):
            db.add_transaction("2026-10-01", "Expense", "Food", 0.0, "UPI", db_path=self.db_path)

        with self.assertRaises(ValueError):
            db.add_transaction("2026-10-01", "InvalidType", "Food", 500.0, "UPI", db_path=self.db_path)

    def test_update_and_delete_transaction(self):
        """Verify editing and deleting records."""
        tx_id = db.add_transaction("2026-10-01", "Expense", "Travel", 200.0, "Cash", "Auto fare", db_path=self.db_path)
        
        # Update
        updated = db.update_transaction(tx_id, "2026-10-02", "Expense", "Travel", 250.0, "UPI", "Updated fare", db_path=self.db_path)
        self.assertTrue(updated)

        tx_mod = db.get_transaction_by_id(tx_id, db_path=self.db_path)
        self.assertEqual(float(tx_mod['amount']), 250.0)
        self.assertEqual(tx_mod['payment_method'], "UPI")

        # Delete
        deleted = db.delete_transaction(tx_id, db_path=self.db_path)
        self.assertTrue(deleted)
        self.assertIsNone(db.get_transaction_by_id(tx_id, db_path=self.db_path))

    def test_budget_operations(self):
        """Test setting and getting monthly & category budgets."""
        month = "2026-10"
        db.set_monthly_budget(month, 15000.0, db_path=self.db_path)
        retrieved_budget = db.get_monthly_budget(month, db_path=self.db_path)
        self.assertEqual(retrieved_budget, 15000.0)

        db.set_category_budget(month, "Food", 4000.0, db_path=self.db_path)
        db.set_category_budget(month, "Travel", 2000.0, db_path=self.db_path)

        cat_budgets = db.get_category_budgets(month, db_path=self.db_path)
        self.assertEqual(cat_budgets.get("Food"), 4000.0)
        self.assertEqual(cat_budgets.get("Travel"), 2000.0)

    def test_sample_data_workflow(self):
        """Verify loading and clearing sample demo data."""
        self.assertFalse(db.has_sample_data(db_path=self.db_path))

        count = db.load_sample_data(db_path=self.db_path)
        self.assertGreater(count, 30)
        self.assertTrue(db.has_sample_data(db_path=self.db_path))

        # Add a user record
        user_tx = db.add_transaction("2026-10-10", "Expense", "Bills", 500.0, "UPI", "User electricity bill", is_sample=0, db_path=self.db_path)

        # Clear sample data
        deleted_count = db.clear_sample_data(db_path=self.db_path)
        self.assertEqual(deleted_count, count)
        self.assertFalse(db.has_sample_data(db_path=self.db_path))

        # Ensure user transaction remains untouched
        user_record = db.get_transaction_by_id(user_tx, db_path=self.db_path)
        self.assertIsNotNone(user_record)

    def test_expense_utils_metrics(self):
        """Verify financial calculations and Indian Rupee formatting."""
        formatted = utils.format_inr(125450.50)
        self.assertEqual(formatted, "₹1,25,450.50")

        db.add_transaction("2026-10-01", "Income", "Salary", 50000.0, "Net Banking", db_path=self.db_path)
        db.add_transaction("2026-10-03", "Expense", "Food", 5000.0, "UPI", db_path=self.db_path)
        db.add_transaction("2026-10-05", "Expense", "Shopping", 10000.0, "Credit Card", db_path=self.db_path)

        df = db.get_transactions(db_path=self.db_path)
        metrics = utils.calculate_dashboard_metrics(df, current_month="2026-10", total_budget=20000.0)

        self.assertEqual(metrics['total_income'], 50000.0)
        self.assertEqual(metrics['total_expenses'], 15000.0)
        self.assertEqual(metrics['remaining_balance'], 35000.0)
        self.assertEqual(metrics['savings_rate'], 70.0)
        self.assertEqual(metrics['budget_used_pct'], 75.0)

    def test_ml_empty_and_insufficient_data(self):
        """Verify ML model handles 0 and < 3 months properly with transparent notices."""
        # Empty
        df_empty = pd.DataFrame()
        res_empty = ml.train_and_predict_expenses(df_empty)
        self.assertFalse(res_empty['success'])
        self.assertEqual(res_empty['status_code'], 'NO_DATA')

        # 1 Month
        db.add_transaction("2026-10-01", "Expense", "Food", 1000.0, "UPI", db_path=self.db_path)
        df_one = db.get_transactions(db_path=self.db_path)
        res_one = ml.train_and_predict_expenses(df_one)
        self.assertFalse(res_one['success'])
        self.assertEqual(res_one['status_code'], 'INSUFFICIENT_DATA')
        self.assertEqual(res_one['fallback_avg'], 1000.0)

    def test_ml_full_pipeline(self):
        """Verify real Scikit-learn LinearRegression training and prediction on multi-month dataset."""
        db.load_sample_data(db_path=self.db_path)
        df_sample = db.get_transactions(db_path=self.db_path)

        res = ml.train_and_predict_expenses(df_sample, target_budget=18000.0)
        self.assertTrue(res['success'])
        self.assertEqual(res['status_code'], 'SUCCESS')
        self.assertGreaterEqual(res['historical_count'], 5)
        self.assertIsNotNone(res['model'])
        self.assertIn("y = (", res['formula_display'])
        self.assertGreater(res['predicted_expense'], 0.0)
        self.assertIn('budget_comparison', res)

    def test_plotly_charts(self):
        """Verify Plotly charts build without crashing."""
        db.load_sample_data(db_path=self.db_path)
        df = db.get_transactions(db_path=self.db_path)

        monthly = utils.get_monthly_summary(df)
        cat_df = utils.get_category_summary(df)
        ml_res = ml.train_and_predict_expenses(df, target_budget=18000.0)

        fig1 = charts.plot_income_vs_expense(monthly)
        self.assertIsNotNone(fig1)

        fig2 = charts.plot_category_pie(cat_df)
        self.assertIsNotNone(fig2)

        fig3 = charts.plot_monthly_trend_with_ml(ml_res['monthly_data'], ml_res)
        self.assertIsNotNone(fig3)

        fig4 = charts.plot_budget_vs_actual(["Food", "Travel"], [2000, 1500], [3000, 2000])
        self.assertIsNotNone(fig4)


if __name__ == "__main__":
    unittest.main()
