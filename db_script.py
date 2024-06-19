import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create a table for user credentials
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
''')

# insert test data
c.execute('INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)', ('admin', '123'))
conn.commit()

conn.close()

print("Database initialized and test user added.")
