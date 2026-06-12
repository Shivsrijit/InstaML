import requests
import sqlite3
import os
import sys

# Inject workspace/backend paths
sys.path.insert(0, r"c:\Users\SSN\OneDrive - Shiv Nadar University - Chennai\Desktop\instaml")
sys.path.insert(0, r"c:\Users\SSN\OneDrive - Shiv Nadar University - Chennai\Desktop\instaml\backend")

from backend.app.db.session import query_sync, execute_sync

def main():
    username = "temprunner"
    email = "temprunner@example.com"
    password = "runnerpassword123"
    
    print("Attempting to register user via running server...")
    try:
        res = requests.post("http://127.0.0.1:8000/api/auth/register", json={
            "username": username,
            "email": email,
            "password": password
        })
        print("Register response:", res.status_code, res.text)
    except Exception as e:
        print("Error registering:", e)
        return
        
    print("\n1. Checking in local SQLite database...")
    local_db_path = r"c:\Users\SSN\OneDrive - Shiv Nadar University - Chennai\Desktop\instaml\backend\instaml.db"
    if os.path.exists(local_db_path):
        conn = sqlite3.connect(local_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        print("Found in local SQLite:", row is not None)
        conn.close()
    else:
        print("Local DB file does not exist.")
        
    print("\n2. Checking in Turso database...")
    try:
        users = query_sync("SELECT * FROM users WHERE username = ?", [username])
        print("Found in Turso DB:", len(users) > 0)
    except Exception as e:
        print("Error querying Turso:", e)
        
    # Clean up from whichever database it was written to
    print("\nCleaning up temprunner...")
    try:
        res = requests.delete(f"http://127.0.0.1:8000/api/users/{username}") # if there is a delete user endpoint, otherwise delete directly
    except:
        pass
    
    # Direct cleanup from both just in case
    try:
        execute_sync("DELETE FROM users WHERE username = ?", [username])
    except:
        pass
    try:
        if os.path.exists(local_db_path):
            conn = sqlite3.connect(local_db_path)
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
    except:
        pass

if __name__ == "__main__":
    main()
