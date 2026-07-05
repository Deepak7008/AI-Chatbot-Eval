import sqlite3
import os
import sys

# Path to the SQLite database
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_results.db'))

def delete_run(run_id: int):
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    
    # CRITICAL: SQLite requires foreign keys to be explicitly enabled per connection 
    # for 'ON DELETE CASCADE' to automatically delete the associated eval_results.
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    try:
        # Check if the run exists first
        cursor.execute("SELECT run_name, timestamp FROM eval_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        
        if not row:
            print(f"Error: Run with ID {run_id} does not exist in the database.")
            return

        run_name, timestamp = row
        print(f"Found run to delete: {run_name} (Created: {timestamp})")
        
        # Delete from eval_runs
        # The 'ON DELETE CASCADE' constraint on eval_results will automatically delete the child rows.
        cursor.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
        conn.commit()
        
        print(f"✅ Successfully deleted Run ID {run_id} and all its associated test case results!")
    except Exception as e:
        print(f"❌ Failed to delete run: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_run.py <run_id>")
        print("Example: python delete_run.py 12")
    else:
        try:
            target_id = int(sys.argv[1])
            delete_run(target_id)
        except ValueError:
            print("Error: run_id must be a valid integer.")
