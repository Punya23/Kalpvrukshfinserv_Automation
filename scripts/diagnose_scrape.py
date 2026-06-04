import requests
import random

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

headers = {
    "User-Agent": random.choice(USER_AGENTS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

url = "https://www.google.com/search?q=doctors+in+pune&tbm=lcl"
response = requests.get(url, headers=headers)
print("Status Code:", response.status_code)
print("Response Length:", len(response.text))

with open("google_lcl_response.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Saved to google_lcl_response.html")
