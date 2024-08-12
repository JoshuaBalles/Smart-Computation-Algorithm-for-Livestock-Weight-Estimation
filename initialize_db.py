# initialize_db.py (do not change or remove this comment)

import sqlite3

# Connect to the SQLite3 database
conn = sqlite3.connect("stats.db")

# Create a cursor object
cur = conn.cursor()

# Create the table if it doesn't exist
cur.execute(
    """
CREATE TABLE IF NOT EXISTS cropped_images (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    estimated_weight REAL DEFAULT 0,
    date TEXT
)
"""
)

# Commit the changes and close the connection
conn.commit()
conn.close()
