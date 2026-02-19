import sqlite3
import pandas as pd
import os

from src.config import DB_PATH

EMPLOYEES_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    salary REAL,
    department_id INTEGER,
    hire_date DATE
);
"""

DEPARTMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT,
    location TEXT
);
"""

EMPLOYEE_COLUMNS = [
    "employee_id",
    "first_name",
    "last_name",
    "email",
    "salary",
    "department_id",
    "hire_date",
]

DEPARTMENT_COLUMNS = [
    "department_id",
    "department_name",
    "location",
]

def setup_database():
    os.makedirs("database", exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(DEPARTMENTS_SCHEMA)
    cursor.execute(EMPLOYEES_SCHEMA)

    # ---- LOAD DEPARTMENTS ----
    dept_df = pd.read_csv("data/departments.csv")
    dept_df = dept_df[dept_df.columns.intersection(DEPARTMENT_COLUMNS)]
    dept_df = dept_df.reindex(columns=DEPARTMENT_COLUMNS)

    dept_df.to_sql("departments", conn, if_exists="append", index=False)

    # ---- LOAD EMPLOYEES ----
    emp_df = pd.read_csv("data/employees.csv")
    emp_df = emp_df[emp_df.columns.intersection(EMPLOYEE_COLUMNS)]
    emp_df = emp_df.reindex(columns=EMPLOYEE_COLUMNS)

    emp_df.to_sql("employees", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print("✅ Database setup completed successfully")
