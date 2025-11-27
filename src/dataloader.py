import sqlite3
import pandas as pd
import requests
import os

class DataLoader:
    def __init__(self, db_url: str, local_path: str = "data/phishing.db"):
        self.db_url = db_url
        self.local_path = local_path
        
    def download_database(self):
        """Download database if not exists locally"""
        os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
        
        if not os.path.exists(self.local_path):
            print("Downloading phishing database...")
            response = requests.get(self.db_url)
            with open(self.local_path, 'wb') as f:
                f.write(response.content)
            print("Download completed.")
        return self.local_path
    
    def load_data(self, table_name: str = "phishing_data") -> pd.DataFrame:
        """Load data from SQLite database"""
        db_path = self.download_database()
        
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            print(f"Loaded {len(df)} records from {table_name}")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")