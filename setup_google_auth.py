import gspread
import os
from pathlib import Path

# Ensure credentials directory exists
Path("credentials").mkdir(exist_ok=True)

print("--------------------------------------------------")
print("Starting Google Authentication Process...")
print("This will open a browser window for you to log in.")
print("--------------------------------------------------\n")

try:
    gc = gspread.oauth(
        credentials_filename='client_secret.json',
        authorized_user_filename='credentials/authorized_user.json'
    )
    print("\n✅ Authentication successful!")
    print("The file credentials/authorized_user.json has been created.")
    print("Your bot can now access Google Sheets securely.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Make sure your client_secret.json is in the correct folder.")
