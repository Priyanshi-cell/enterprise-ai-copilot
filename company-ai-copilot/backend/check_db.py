import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

# Show table structure
print("=" * 60)
print("TABLE STRUCTURE")
print("=" * 60)
cursor.execute("PRAGMA table_info(chat_history)")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Show total count
print()
print("=" * 60)
print("CHAT HISTORY SUMMARY")
print("=" * 60)
cursor.execute("SELECT COUNT(*) FROM chat_history")
count = cursor.fetchone()[0]
print(f"Total conversations saved: {count}")

cursor.execute("SELECT DISTINCT company FROM chat_history")
companies = cursor.fetchall()
print(f"Companies in database: {[c[0] for c in companies]}")

# Show all rows
print()
print("=" * 60)
print("ALL CONVERSATIONS")
print("=" * 60)
cursor.execute(
    "SELECT id, company, question, answer, created_at "
    "FROM chat_history ORDER BY created_at DESC"
)
rows = cursor.fetchall()

if not rows:
    print("No conversations yet.")
else:
    for row in rows:
        print(f"\nID       : {row[0]}")
        print(f"Company  : {row[1]}")
        print(f"Time     : {row[4]}")
        print(f"Question : {row[2]}")
        print(f"Answer   : {row[3][:200]}...")
        print("-" * 60)

conn.close()
print("\nDatabase check complete.")