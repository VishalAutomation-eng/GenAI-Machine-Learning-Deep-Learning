from src.db_setup import setup_database
from src.text2sql import run_text2sql

setup_database()

print(run_text2sql("Show average salary by department"))
print(run_text2sql("Which department has highest paid employees?"))
print(run_text2sql("List employees hired after 2022"))
