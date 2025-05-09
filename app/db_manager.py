import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",)
DB_PATH = os.getenv("DB_PATH")
os.makedirs(DB_PATH, exist_ok=True)
DB_NAME = os.path.join(DB_PATH, os.getenv("DB_FILENAME"))

def db_connect():
    # Connect to database
    try:
        connection = sqlite3.connect(DB_NAME)
        try:
            create_tables(connection)
        except Exception as e:
            exit()
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def db_dump(connection):
    # Dump database to file
    try:
        with open("dump.sql", "w") as file:
            for line in connection.iterdump():
                file.write(f"{line}\n")
        print("Database dumped to file dump.sql")
    except Exception as e:
        print(f"Error dumping database: {e}")
        exit()

def create_tables(connection):
        cursor = connection.cursor()
        try:
            # Create table company if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp datetime,
                    prediction boolean,
                    prediction_rate float,
                    page_values integer,
                    exit_rates float,
                    bounce_rates float,
                    weekend boolean,
                    administrative integer,
                    informational integer,
                    product_related integer,
                    administrative_duration integer,
                    informational_duration integer,
                    product_related_duration integer,
                    month integer,
                    new_visitor boolean
                    
                )
            """)
            print("Table checked/created")
            connection.commit()
        except Exception as e:
            print(f"Error creating table: {e}")
            exit()


           
def save_prediction(prediction, connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO stock_value (
                timestamp, prediction, prediction_rate, page_values, exit_rates,
                bounce_rates, weekend, administrative, informational,
                product_related, administrative_duration, informational_duration,
                product_related_duration, month, new_visitor
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.timestamp, prediction.prediction, prediction.prediction_rate,
            prediction.page_values, prediction.exit_rates, prediction.bounce_rates,
            prediction.weekend, prediction.administrative, prediction.informational,
            prediction.product_related, prediction.administrative_duration,
            prediction.informational_duration, prediction.product_related_duration,
            prediction.month, prediction.new_visitor
        ))
        connection.commit()
    except Exception as e:
        logger.error(f"Error saving stock value: {e}")
        print(f"Error saving stock value: {e}")
        exit()




def db_file_delete(db_file):
    try:
        while True:
            if input(f"Type 'yes' to delete file {db_file}. CTRL+C to abort: ") == "yes":
                os.remove(db_file)
                print(f"File {db_file} deleted")
                logger.warning(f'File {db_file} deleted')
                exit()
            else:
                print("Wrong input!!!")
    except Exception as e:
        logger.error(f"Error deleting file {db_file}: {e}")
        print(f"Error deleting file dump.sql: {e}")
        exit()