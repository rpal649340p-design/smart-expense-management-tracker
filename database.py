"""
database.py
-----------
SQLite Database management module for Smart Expense Management Tracker.
Designed for beginner-friendly understanding with secure parameterized queries.
"""

import sqlite3
import os
from datetime import datetime, date, timedelta
import pandas as pd

# Default database path inside the project's data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "expenses.db")


def get_db_path(db_path=None):
    """Returns the verified database path, ensuring the parent directory exists."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_connection(db_path=None):
    """Establishes and returns an SQLite database connection."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """
    Initializes database tables if they do not exist:
    1. transactions: Stores all income and expense records.
    2. monthly_budgets: Stores overall monthly budgets.
    3. category_budgets: Stores category-specific budgets per month.
    4. settings: Stores key-value application settings.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Table 1: Transactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('Income', 'Expense')),
            category TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            payment_method TEXT NOT NULL,
            description TEXT,
            is_sample INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 2: Monthly Budgets (month_year format: 'YYYY-MM')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monthly_budgets (
            month_year TEXT PRIMARY KEY,
            total_budget REAL NOT NULL CHECK(total_budget >= 0),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table 3: Category Budgets (per month_year and category)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS category_budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month_year TEXT NOT NULL,
            category TEXT NOT NULL,
            budget_amount REAL NOT NULL CHECK(budget_amount >= 0),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(month_year, category)
        )
    """)

    # Table 4: Key-Value Settings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_transaction(tx_date, tx_type, category, amount, payment_method, description="", is_sample=0, db_path=None):
    """Adds a new transaction record to the database using parameterized queries."""
    if amount <= 0:
        raise ValueError("Transaction amount must be strictly greater than 0.")
    if tx_type not in ('Income', 'Expense'):
        raise ValueError("Transaction type must be 'Income' or 'Expense'.")

    # Format date string cleanly
    if isinstance(tx_date, (datetime, date)):
        date_str = tx_date.strftime("%Y-%m-%d")
    else:
        date_str = str(tx_date)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (date, type, category, amount, payment_method, description, is_sample)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (date_str, tx_type, category.strip(), float(amount), payment_method.strip(), description.strip(), int(is_sample)))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def update_transaction(tx_id, tx_date, tx_type, category, amount, payment_method, description="", db_path=None):
    """Updates an existing transaction record by ID."""
    if amount <= 0:
        raise ValueError("Transaction amount must be strictly greater than 0.")
    if tx_type not in ('Income', 'Expense'):
        raise ValueError("Transaction type must be 'Income' or 'Expense'.")

    if isinstance(tx_date, (datetime, date)):
        date_str = tx_date.strftime("%Y-%m-%d")
    else:
        date_str = str(tx_date)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE transactions
        SET date = ?, type = ?, category = ?, amount = ?, payment_method = ?, description = ?
        WHERE id = ?
    """, (date_str, tx_type, category.strip(), float(amount), payment_method.strip(), description.strip(), int(tx_id)))
    
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def delete_transaction(tx_id, db_path=None):
    """Deletes a transaction by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def get_transaction_by_id(tx_id, db_path=None):
    """Fetches a single transaction by its primary key ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE id = ?", (int(tx_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_transactions(start_date=None, end_date=None, category=None, tx_type=None, search_query=None, db_path=None):
    """
    Retrieves transactions based on optional filters.
    Returns a pandas DataFrame for seamless integration with Streamlit and Plotly.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    
    query = "SELECT id, date, type, category, amount, payment_method, description, is_sample, created_at FROM transactions WHERE 1=1"
    params = []

    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    if tx_type and tx_type != "All":
        query += " AND type = ?"
        params.append(tx_type)
    if search_query:
        query += " AND (description LIKE ? OR category LIKE ? OR payment_method LIKE ?)"
        like_str = f"%{search_query.strip()}%"
        params.extend([like_str, like_str, like_str])

    query += " ORDER BY date DESC, id DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def set_monthly_budget(month_year, total_budget, db_path=None):
    """Sets or updates the total monthly budget for a specific month (YYYY-MM)."""
    if total_budget < 0:
        raise ValueError("Budget cannot be negative.")
    
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO monthly_budgets (month_year, total_budget, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(month_year) DO UPDATE SET
            total_budget = excluded.total_budget,
            updated_at = CURRENT_TIMESTAMP
    """, (month_year, float(total_budget)))
    conn.commit()
    conn.close()


def get_monthly_budget(month_year, db_path=None):
    """Retrieves the total monthly budget for a specific month (YYYY-MM). Returns None if not set."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT total_budget FROM monthly_budgets WHERE month_year = ?", (month_year,))
    row = cursor.fetchone()
    conn.close()
    return float(row['total_budget']) if row else None


def set_category_budget(month_year, category, budget_amount, db_path=None):
    """Sets or updates a category budget for a specific month."""
    if budget_amount < 0:
        raise ValueError("Category budget cannot be negative.")
    
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO category_budgets (month_year, category, budget_amount, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(month_year, category) DO UPDATE SET
            budget_amount = excluded.budget_amount,
            updated_at = CURRENT_TIMESTAMP
    """, (month_year, category.strip(), float(budget_amount)))
    conn.commit()
    conn.close()


def get_category_budgets(month_year, db_path=None):
    """Returns a dictionary of category budgets for a given month: {category: budget_amount}."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT category, budget_amount FROM category_budgets WHERE month_year = ?", (month_year,))
    rows = cursor.fetchall()
    conn.close()
    return {row['category']: float(row['budget_amount']) for row in rows}


def has_sample_data(db_path=None):
    """Checks if any sample data exists in the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM transactions WHERE is_sample = 1")
    count = cursor.fetchone()['count']
    conn.close()
    return count > 0


def has_user_data(db_path=None):
    """Checks if any non-sample user data exists in the database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM transactions WHERE is_sample = 0")
    count = cursor.fetchone()['count']
    conn.close()
    return count > 0


def clear_sample_data(db_path=None):
    """Removes all transactions labeled as sample data."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE is_sample = 1")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count


def clear_all_data(db_path=None):
    """Completely resets all transactions and budget tables."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    cursor.execute("DELETE FROM monthly_budgets")
    cursor.execute("DELETE FROM category_budgets")
    conn.commit()
    conn.close()


def load_sample_data(db_path=None):
    """
    Populates the database with realistic student/individual personal finance records
    spanning 6 consecutive historical months (May 2026 to October 2026).
    This immediately provides:
    - Multiple months of income and expenses
    - Varied realistic spending categories (Food, Groceries, Travel, Education, etc.)
    - Realistic budget figures and category caps
    - Sufficient historical points for the Machine Learning model demonstration!
    All rows are strictly tagged with is_sample = 1.
    """
    init_db(db_path)
    # Clear any old sample records first to avoid duplicate multiplication
    clear_sample_data(db_path)

    # Realistic 6-month transaction series
    sample_records = [
        # --- Month 1: May 2026 ---
        ("2026-05-01", "Income", "Allowance / Pocket Money", 20000, "Net Banking", "Monthly allowance from parents"),
        ("2026-05-03", "Expense", "Food", 1250, "UPI", "Hostel canteen & evening snacks"),
        ("2026-05-06", "Expense", "Education", 3200, "Debit Card", "Semester reference books & lab coat"),
        ("2026-05-09", "Expense", "Groceries", 1800, "UPI", "Monthly toiletries & dry snacks"),
        ("2026-05-12", "Expense", "Travel", 950, "UPI", "Metro smart card recharge & bus pass"),
        ("2026-05-15", "Expense", "Bills", 699, "UPI", "Mobile recharge + JioFiber hostel share"),
        ("2026-05-18", "Expense", "Entertainment", 850, "UPI", "Weekend movie with classmates"),
        ("2026-05-22", "Expense", "Food", 1400, "UPI", "Weekend dinner at campus cafe"),
        ("2026-05-25", "Expense", "Shopping", 1600, "UPI", "College casual shirts"),
        ("2026-05-28", "Expense", "Healthcare", 450, "Cash", "Doctor consultation & medicines"),

        # --- Month 2: June 2026 ---
        ("2026-06-01", "Income", "Allowance / Pocket Money", 20000, "Net Banking", "Monthly allowance from parents"),
        ("2026-06-02", "Income", "Freelance / Internship", 4000, "UPI", "Web development mini project freelance"),
        ("2026-06-04", "Expense", "Food", 1650, "UPI", "Mess food & evening snacks"),
        ("2026-06-07", "Expense", "Education", 1200, "UPI", "Online Python/Data Science course subscription"),
        ("2026-06-11", "Expense", "Travel", 1100, "UPI", "Auto & cab fares for project meetups"),
        ("2026-06-14", "Expense", "Groceries", 2100, "Credit Card", "Grocery stockup for room"),
        ("2026-06-17", "Expense", "Bills", 750, "UPI", "Postpaid mobile bill & streaming service"),
        ("2026-06-20", "Expense", "Entertainment", 1200, "UPI", "College fest participation & snacks"),
        ("2026-06-24", "Expense", "Shopping", 2400, "Debit Card", "Running shoes & gym wear"),
        ("2026-06-27", "Expense", "Food", 1500, "Cash", "Street food & canteen with friends"),
        ("2026-06-29", "Expense", "Other", 600, "UPI", "Printouts and spiral binding for reports"),

        # --- Month 3: July 2026 ---
        ("2026-07-01", "Income", "Allowance / Pocket Money", 20000, "Net Banking", "Monthly allowance"),
        ("2026-07-04", "Expense", "Food", 1850, "UPI", "Swiggy orders during mid-term prep"),
        ("2026-07-08", "Expense", "Education", 2500, "Net Banking", "NPTEL & Hackathon registration fee"),
        ("2026-07-12", "Expense", "Travel", 1300, "UPI", "Train ticket to hometown for weekend"),
        ("2026-07-16", "Expense", "Groceries", 2250, "UPI", "Snacks, milk, fruits"),
        ("2026-07-19", "Expense", "Bills", 700, "UPI", "Recharge and Wi-Fi"),
        ("2026-07-23", "Expense", "Entertainment", 900, "UPI", "Bowling & cafe with study group"),
        ("2026-07-26", "Expense", "Food", 1700, "UPI", "Campus canteen & coffee"),
        ("2026-07-28", "Expense", "Healthcare", 800, "Cash", "Dental checkup & prescription"),
        ("2026-07-30", "Expense", "Shopping", 1100, "Credit Card", "Stationery, notebook bundle, backpack"),

        # --- Month 4: August 2026 ---
        ("2026-08-01", "Income", "Allowance / Pocket Money", 22000, "Net Banking", "Monthly allowance + festival bonus"),
        ("2026-08-03", "Income", "Freelance / Internship", 5500, "UPI", "Frontend UI bug fixes for client"),
        ("2026-08-05", "Expense", "Food", 2100, "UPI", "Celebration dinner & birthday treat"),
        ("2026-08-09", "Expense", "Shopping", 3200, "Credit Card", "Festive traditional wear"),
        ("2026-08-13", "Expense", "Travel", 1450, "UPI", "Metro & Ola rides"),
        ("2026-08-16", "Expense", "Groceries", 2400, "UPI", "Monthly essentials"),
        ("2026-08-20", "Expense", "Education", 1800, "Debit Card", "Cloud hosting & GitHub Copilot subscription"),
        ("2026-08-23", "Expense", "Bills", 850, "UPI", "Mobile + electricity share"),
        ("2026-08-26", "Expense", "Food", 1600, "UPI", "Canteen meals"),
        ("2026-08-28", "Expense", "Entertainment", 1100, "UPI", "Gaming lounge session"),

        # --- Month 5: September 2026 ---
        ("2026-09-01", "Income", "Allowance / Pocket Money", 21000, "Net Banking", "Monthly allowance"),
        ("2026-09-04", "Expense", "Food", 1950, "UPI", "Hostel mess & evening tea"),
        ("2026-09-07", "Expense", "Travel", 1600, "UPI", "Inter-college symposium travel"),
        ("2026-09-10", "Expense", "Education", 3500, "Net Banking", "Hardware components (Arduino, sensors for IoT project)"),
        ("2026-09-14", "Expense", "Groceries", 2300, "UPI", "Supermarket shopping"),
        ("2026-09-18", "Expense", "Bills", 750, "UPI", "Phone & cloud backup bill"),
        ("2026-09-22", "Expense", "Food", 1750, "UPI", "Weekend food delivery"),
        ("2026-09-25", "Expense", "Entertainment", 1300, "Debit Card", "Concert ticket in university"),
        ("2026-09-28", "Expense", "Healthcare", 600, "UPI", "Vitamin supplements & clinic visit"),
        ("2026-09-29", "Expense", "Shopping", 1500, "UPI", "Replacement earphones & laptop sleeve"),

        # --- Month 6: October 2026 (Current Month) ---
        ("2026-10-01", "Income", "Allowance / Pocket Money", 22000, "Net Banking", "Monthly allowance"),
        ("2026-10-03", "Income", "Freelance / Internship", 3500, "UPI", "Data analysis assignment support"),
        ("2026-10-04", "Expense", "Food", 2200, "UPI", "Department freshers welcome party & snacks"),
        ("2026-10-07", "Expense", "Education", 1500, "UPI", "IEEE membership student discount"),
        ("2026-10-09", "Expense", "Travel", 1200, "UPI", "Metro recharge & bus fare"),
        ("2026-10-11", "Expense", "Groceries", 2500, "Debit Card", "Hostel pantry groceries"),
        ("2026-10-13", "Expense", "Bills", 799, "UPI", "Jio postpaid & OTT share"),
        ("2026-10-14", "Expense", "Food", 1350, "Cash", "Canteen snacks and beverages")
    ]

    for rec in sample_records:
        add_transaction(
            tx_date=rec[0],
            tx_type=rec[1],
            category=rec[2],
            amount=rec[3],
            payment_method=rec[4],
            description=rec[5],
            is_sample=1,
            db_path=db_path
        )

    # Set sample budgets for October 2026
    current_month = "2026-10"
    set_monthly_budget(current_month, 18000.0, db_path=db_path)
    
    category_caps = {
        "Food": 5000.0,
        "Groceries": 3500.0,
        "Travel": 2500.0,
        "Education": 3000.0,
        "Entertainment": 2000.0,
        "Shopping": 3000.0,
        "Bills": 1200.0,
        "Healthcare": 1000.0,
        "Other": 1000.0
    }
    for cat, cap in category_caps.items():
        set_category_budget(current_month, cat, cap, db_path=db_path)

    return len(sample_records)
