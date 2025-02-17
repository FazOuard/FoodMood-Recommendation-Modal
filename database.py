import sqlalchemy as sa
from sqlalchemy import create_engine
import urllib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read database config from .env
DB_SERVER = os.getenv('DB_SERVER')
DB_DATABASE = os.getenv('DB_DATABASE')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Debugging: Check if environment variables are loaded
if not all([DB_SERVER, DB_DATABASE, DB_USER, DB_PASSWORD]):
    print("❌ ")
    exit()

# ODBC Connection String
conn_str = urllib.parse.quote_plus(
    f'Driver={{ODBC Driver 17 for SQL Server}};' 
    f'Server={DB_SERVER};'
    f'Database={DB_DATABASE};'
    f'UID={DB_USER};'
    f'PWD={DB_PASSWORD};'
)

# Create SQLAlchemy Engine
try:
    ApiSQLEngine = create_engine(f'mssql+pyodbc:///?odbc_connect={conn_str}')
    print("✅")

except Exception as e:
    print("❌ ", e)
