# 💰 Smart Expense Management Tracker Using Machine Learning

> A beginner-friendly, full-stack personal finance web application built with **Python**, **Streamlit**, **SQLite**, **Plotly**, and **Scikit-learn** for college project demonstrations and everyday expense tracking.

---

## 📌 1. Project Overview

The **Smart Expense Management Tracker** empowers students and individuals to monitor their income, track daily expenditures, enforce monthly and category budgets, view rich interactive visualizations, and forecast upcoming monthly spending using a **real Machine Learning Linear Regression model**.

Designed specifically for 1st-year B.Tech Computer Science and Engineering students, this application combines clean software engineering principles with practical machine learning without unnecessary architectural complexity.

---

## 🎯 2. Project Objectives

1. **Systematic Record Keeping**: Provide a structured, validated way to log income and expenses into a persistent SQLite database.
2. **Budget Discipline**: Prevent overspending through real-time alerts and visual progress indicators when approaching or exceeding limits.
3. **Data-Driven Insights**: Deliver instant visibility into top spending categories, monthly savings rates, and cash flow trends.
4. **Practical Machine Learning**: Apply Supervised Learning (Linear Regression) to predict future monthly expenses and mathematically analyze spending trends.
5. **Zero-Setup Demonstration**: Include a realistic 6-month demo dataset so students can demonstrate the full application and ML model immediately during a viva or lab evaluation.

---

## 🛠️ 3. Technology Stack

| Component | Technology | Role & Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core application logic and data pipelines |
| **Frontend Framework** | Streamlit | Clean, reactive web interface with customizable cards and layouts |
| **Data Processing** | Pandas, NumPy | Data aggregation, group-bys, and numerical transformations |
| **Machine Learning** | Scikit-learn (`LinearRegression`) | Real linear trend modeling and next-month expense forecasting |
| **Visualizations** | Plotly (`plotly.graph_objects`, `px`) | Interactive charts with Indian Rupee (`₹`) hover tooltips |
| **Database** | SQLite3 | Embedded, lightweight relational database with parameterized queries |
| **Model Persistence** | Joblib | Serialization and storage of trained regression models |
| **Version Control** | Git & GitHub | Source code tracking and remote portfolio publishing |

---

## ✨ 4. Core Features

### A. Comprehensive Dashboard
- **All-Time Key Metrics**: Total Income, Total Expenses, Net Savings, Overall Savings Rate.
- **Current Month Overview**: Current month spending, income, and budget consumption percentage.
- **Dynamic Alerts**: Color-coded banners for safe budget pace, 80% caution warning, and budget exceeded alerts.
- **ML Forecast Preview**: Prominent badge displaying the predicted next-month expenditure.
- **Visuals**: Dual-column cash flow bar chart and category breakdown donut chart.
- **Recent Transactions Table**: Shows the latest records with date, source, and amount.

### B. Transaction Management (Income & Expense)
- Strict input validation: prevents negative amounts or zero transactions.
- Flexible categorization:
  - **Expenses**: Food, Groceries, Travel, Education, Entertainment, Shopping, Bills, Healthcare, Other.
  - **Income**: Allowance / Pocket Money, Freelance / Internship, Salary, Investment / Interest, Gift, Other.
- Payment modes: UPI, Cash, Credit Card, Debit Card, Net Banking.
- Full CRUD operations: Add, View, Filter, Edit, and Delete transactions with instant database persistence.
- One-click CSV export with timestamped filenames.

### C. Monthly Budget Planner
- Total monthly budget configuration (e.g. ₹18,000 / month).
- Category-specific spending limits.
- Color-coded progress bars:
  - 🟢 **Green (<80%)**: Safe spending trajectory.
  - 🟠 **Orange (80% - 100%)**: Caution — approaching limit.
  - 🔴 **Red (>100%)**: Over-budget alert with exact rupee excess.
- Interactive Plotly comparison chart: Budget vs Actual spending.

### D. Real Machine Learning Expense Prediction
- **No fake statistics**: Employs Scikit-learn's genuine `LinearRegression` algorithm.
- **Time-Series Feature Engineering**: Chronological month sequencing ($X = 1, 2, 3, \dots, N$).
- **Mathematical Transparency**:
  - Displays the exact fitted regression formula: $y = mx + c$.
  - Reports **$R^2$ Score** (goodness-of-fit) and **MAE** (Mean Absolute Error in ₹).
  - Interprets the slope ($m$) in plain English (e.g., *"Spending is trending upwards by ₹420.50/month"*).
- **Proactive Budget Comparison**: Compares the projected expenditure against the user's budget and advises whether spending will remain within limits.
- **Data Sufficiency Handling**: Transparently requires at least 3 distinct months; if fewer exist, it displays an informative message, empirical average fallback, and a one-click button to load sample data.

### E. Financial Reports & Actionable Insights
- All-time metrics: Top spending category with percentage share, average monthly expenditure, peak expense month, and savings rate.
- Automated rule-based tips: Identifies categories with over 35% concentration, rising expense categories, dining out optimizations, and the recommended 50/30/20 budgeting framework.
- Dedicated visualization tabs: Category distribution, savings trajectory over time, and payment method share.

### F. Sample Data Management & Academic Integrity
- **One-Click 6-Month Demo Data**: Pre-populates 50+ realistic student transactions from May 2026 to October 2026.
- **Strict Labeling**: Demo transactions are tagged with `is_sample = 1` and clearly badged in the UI.
- **Independent Reset**: Users can clear demo records without affecting personal records, or reset the entire database.

---

## 🧠 5. How Machine Learning Works in This Project

### 1. Supervised Learning Concept
We formulate expense prediction as a **univariate regression problem**:
- **Feature ($X$)**: Chronological month index ($1, 2, 3, \dots, N$) representing time.
- **Target ($y$)**: Total aggregated expenses incurred during that month in Rupees ($\text{₹}$).

```
Historical Month     Feature (X)     Target (y) Total Expense
May 2026             1               ₹ 11,899.00
June 2026            2               ₹ 13,000.00
July 2026            3               ₹ 13,800.00
August 2026          4               ₹ 14,400.00
September 2026       5               ₹ 15,150.00
October 2026         6               ₹ 10,799.00
--------------------------------------------------------------
November 2026        7 (Next Month)  ? (To be predicted by ML)
```

### 2. Mathematical Formulation
Linear Regression computes the best-fit line minimizing the Sum of Squared Residuals (Ordinary Least Squares):

$$y = m \cdot X + c$$

Where:
- $m$ (**Slope**): Rate of expense change per month.
- $c$ (**Intercept**): Theoretical baseline expenditure at $X = 0$.

### 3. Forecasting Next Month
To forecast spending for month $N + 1$:
$$\hat{y}_{\text{next}} = m \cdot (N + 1) + c$$

### 4. Evaluation Metrics
- **$R^2$ Score**: Proportion of variance in spending explained by the passage of time.
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
  Measures the average error magnitude in Rupees.

---

## 📁 6. Project Folder Structure

```text
expense-management-tracker/
│
├── app.py                  # Main Streamlit web application & multi-page UI
├── database.py             # SQLite database layer, schema, CRUD & sample generator
├── expense_utils.py        # INR currency formatting, financial metrics & rule-based insights
├── ml_model.py             # Scikit-learn LinearRegression model training & forecasting
├── charts.py               # Interactive Plotly visual charts (bar, pie, trend, savings)
├── test_app.py             # Automated unit test suite verifying all logic
├── requirements.txt        # Python dependency specifications
├── README.md               # Detailed project documentation & college viva guide
├── .gitignore              # Ignores virtual environments, caches, and local databases
│
├── data/                   # SQLite database storage (expenses.db generated at runtime)
├── models/                 # Serialized ML model artifacts (.joblib)
└── screenshots/            # Directory reserved for UI presentation screenshots
```

---

## 💻 7. Installation & Running Instructions (Windows)

Follow these simple steps in **PowerShell** or **Command Prompt**:

### Step 1: Open the Project Directory
```powershell
cd C:\Users\princ\.gemini\antigravity\scratch\expense-management-tracker
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```
*(You will see `(.venv)` appear at the beginning of your terminal prompt).*

### Step 4: Install Required Packages
```powershell
python -m pip install -r requirements.txt
```

### Step 5: Run the Automated Tests (Optional but Recommended)
```powershell
python test_app.py
```
*(All 8 test suites should pass with `OK`).*

### Step 6: Launch the Streamlit Application
```powershell
streamlit run app.py
```

Your default browser will automatically open:
```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 🎓 8. College Project Demonstration & Viva Guide

When presenting this project to your professor or lab examiner, follow this 4-step demonstration flow:

### Step 1: Introduction (30 seconds)
> *"Good morning Sir/Ma'am. My project is a Smart Expense Management Tracker built with Python, Streamlit, and Scikit-learn. It allows students to manage personal finances, maintain monthly budgets, and uses Machine Learning to forecast upcoming expenses."*

### Step 2: Live Demo Walkthrough (2 minutes)
1. **Show Settings & Demo Data**: Click on `⚙️ Settings & Demo Data` and click **'Load 6-Month Demo Transactions'**. Explain that this loads 6 consecutive months of realistic student spending (May to October 2026).
2. **Show the Dashboard**: Navigate to `📊 Dashboard`. Highlight the KPI cards (Income, Expense, Balance, Budget Usage) and the Plotly Cash Flow bar chart.
3. **Show Budget Alerts**: Navigate to `🎯 Budget Planner`. Demonstrate how the progress bars change color (Green <80%, Orange 80-100%, Red >100%) when spending approaches the budget limit.
4. **Show ML Expense Prediction**: Navigate to `🤖 ML Expense Prediction`.
   - Point to the formula: $y = mx + c$.
   - Point to the calculated slope ($m$) and explain: *"The slope indicates our spending growth rate per month."*
   - Show the star marker on the Plotly chart representing the next month's projection.
   - Show the comparison with the user's budget limit.

### Step 3: Common Viva Questions & Ideal Answers

| Viva Question | Ideal Student Answer |
| :--- | :--- |
| **Q1: Why Linear Regression instead of Deep Learning?** | *"For monthly personal finance with 5 to 12 monthly data points, deep learning neural networks would severely overfit due to lack of big data. Linear regression provides high mathematical interpretability ($y = mx + c$) and runs instantly."* |
| **Q2: What is your feature vector?** | *"Our independent feature $X$ is the chronological month index ($1, 2, 3, \dots, N$) representing time progression. Our target variable $y$ is the total expense in Rupees for that month."* |
| **Q3: How do you prevent SQL injection?** | *"We use SQLite with parameterized queries (`?` placeholders) in all database operations inside `database.py`, ensuring user input is never directly concatenated into SQL strings."* |
| **Q4: What happens if there is only 1 month of data?** | *"Our application detects insufficient training data (<3 months) and refuses to fit a fake regression line. Instead, it offers an empirical average fallback and invites the user to add records or load sample data."* |

---

## 🚀 9. Step-by-Step GitHub Upload Guide

To publish this project to your GitHub profile:

### Step 1: Initialize Git Repository
In your project folder terminal:
```powershell
git init
git branch -M main
```

### Step 2: Stage and Commit Files
```powershell
git add .
git commit -m "Initial commit: Smart Expense Management Tracker with ML"
```
*(Notice that `.gitignore` automatically prevents your `.venv`, `__pycache__`, and `expenses.db` from being uploaded).*

### Step 3: Create a New Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Name the repository: `smart-expense-management-tracker`.
3. Set visibility to **Public**.
4. Do **not** initialize with a README, `.gitignore`, or license (we already have them).
5. Click **Create repository**.

### Step 4: Link and Push to GitHub
Copy your repository URL from GitHub and run:
```powershell
git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/smart-expense-management-tracker.git
git push -u origin main
```

> **Security Note**: Never enter your GitHub password in the terminal. When prompted, use a **GitHub Personal Access Token (PAT)** or the GitHub CLI (`gh auth login`).

---

## 🔮 10. Limitations & Future Enhancements

1. **Category-Level Natural Language Classification**: Use Naive Bayes or DistilBERT to automatically categorize transaction descriptions (e.g. typing "McSpicy burger" automatically classifies as "Food").
2. **Anomaly Detection**: Integrate Scikit-learn's `IsolationForest` to flag abnormal one-off transactions (e.g., accidental duplicate subscriptions or unusually high charges).
3. **Advanced Time Series Forecasting**: When 24+ months of data are recorded, integrate **ARIMA** or **Prophet** to capture seasonal patterns (like holiday spikes or semester tuition).
4. **Receipt Scanner (OCR)**: Integrate Tesseract OCR or Google Gemini Vision to upload paper receipts and automatically parse date, merchant, and total amount.

---

## 📄 License
This project is open-source under the MIT License — feel free to use, modify, and present it for academic and educational purposes.
