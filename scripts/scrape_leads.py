import os
import re
import csv
import time
import random
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# List of realistic user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def scrape_google_local(query: str, max_results: int = 50):
    """
    Scrapes local business listings from Google Search Local Results (tbm=lcl).
    This endpoint returns standard HTML listings including Name, Phone, Rating, and Address.
    """
    logger.info(f"Starting scrape for query: '{query}'")
    
    results = []
    start = 0
    
    while len(results) < max_results:
        # tbm=lcl forces Google to return local map listings in a list format
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=lcl&start={start}"
        logger.info(f"Fetching page starting at result index {start}...")
        
        try:
            response = requests.get(url, headers=get_headers(), timeout=15)
            if response.status_code != 200:
                logger.error(f"Failed to fetch page. Status code: {response.status_code}")
                break
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all business card elements on the Google Local page
            # Google often changes class names, so we look for elements containing the core data
            cards = soup.find_all("div", class_=re.compile(r"VkCve|rllt__details"))
            
            if not cards:
                # Try a broader selector if classes changed
                cards = soup.find_all("div", attrs={"data-cid": True})
                
            if not cards:
                logger.info("No more listings found or page structure changed.")
                break
                
            logger.info(f"Found {len(cards)} listings on this page.")
            
            for card in cards:
                if len(results) >= max_results:
                    break
                    
                try:
                    # Extract Name
                    name_elem = card.find(class_=re.compile(r"OSrXXb|dbg0pd|q81Yee"))
                    name = name_elem.text.strip() if name_elem else "N/A"
                    
                    if name == "N/A" or name in [r["name"] for r in results]:
                        continue
                        
                    # Extract details block text
                    details_text = card.text.strip()
                    
                    # Extract Rating
                    rating_match = re.search(r"(\d\.\d)\s*★", details_text)
                    rating = rating_match.group(1) if rating_match else "N/A"
                    
                    # Extract Phone Number (Indian format matches like +91 90228 73952 or 020 2543 2321)
                    phone_match = re.search(
                        r"(\+91[\s-]?\d{4,5}[\s-]?\d{5}|\b\d{5}[\s-]?\d{5}\b|\b0\d{2,4}[\s-]?\d{6,8}\b)", 
                        details_text
                    )
                    phone = phone_match.group(1).replace(" ", "").replace("-", "") if phone_match else "N/A"
                    
                    # Clean up phone number format
                    if phone != "N/A":
                        if phone.startswith("0") and not phone.startswith("091"):
                            # Landline or local number
                            pass
                        elif not phone.startswith("+") and not phone.startswith("91"):
                            # Convert to international standard
                            phone = f"+91{phone}"
                        elif phone.startswith("91") and not phone.startswith("+"):
                            phone = f"+{phone}"
                    
                    # Address extraction (usually follows the phone or contains landmarks/roads)
                    # We extract lines to isolate the address part
                    lines = [l.strip() for l in details_text.split("·") if l.strip()]
                    address = "N/A"
                    for line in lines:
                        if "Pune" in line or "Maharashtra" in line or any(k in line.lower() for k in ["road", "nagar", "peth", "chowk", "society"]):
                            address = line
                            break
                    
                    # If we couldn't isolate it, grab the last lines of the card text
                    if address == "N/A" and len(lines) > 1:
                        address = lines[-1]
                        
                    results.append({
                        "name": name,
                        "phone": phone,
                        "rating": rating,
                        "address": address,
                        "category": query
                    })
                    logger.info(f"Scraped: {name} | Phone: {phone} | Rating: {rating}")
                    
                except Exception as card_e:
                    logger.debug(f"Error parsing card: {card_e}")
                    continue
            
            # Google Local Results pagination is usually 20 items per page
            start += 20
            
            # Anti-ban sleep delay between pages
            sleep_time = random.uniform(4.0, 8.0)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds to mimic human browsing...")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error fetching/parsing page: {e}")
            break
            
    return results

def save_to_csv(data, filename: str):
    """Saves the scraped listings to a CSV file in the data/leads directory."""
    output_dir = Path("data/leads")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    headers = ["name", "phone", "rating", "address", "category"]
    
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully saved {len(data)} leads to {filepath}")
    except Exception as e:
        logger.error(f"Error saving leads to CSV: {e}")

if __name__ == "__main__":
    # Test Scrape: Doctors and Architects in Pune
    queries = [
        "doctors in pune",
        "architects in pune",
        "interior designers in kalyani nagar pune",
        "clinics in baner pune"
    ]
    
    all_leads = []
    for query in queries:
        leads = scrape_google_local(query, max_results=15)
        all_leads.extend(leads)
        
        # Pause between different query searches
        time.sleep(random.uniform(5.0, 10.0))
        
    if all_leads:
        # Save to single consolidated leads file
        save_to_csv(all_leads, "scraped_hni_leads.csv")
    else:
        logger.warning("No leads were scraped.")
