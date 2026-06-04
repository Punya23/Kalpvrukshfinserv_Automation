import os
import csv
import time
import random
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def scrape_google_maps(query: str, max_results: int = 20):
    logger.info(f"Starting Playwright scrape for: '{query}'")
    leads = []
    
    with sync_playwright() as p:
        # Launch browser. Using headless=True is faster, but we can set it to False for debugging.
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        
        # Navigate directly to the search page
        import urllib.parse
        search_url = f"https://www.google.com/maps/search/{urllib.parse.quote(query)}"
        logger.info(f"Navigating directly to: {search_url}")
        page.goto(search_url)
        
        # Wait for the results pane to load
        page.wait_for_selector("div[role='feed']", timeout=15000)
        page.wait_for_timeout(2000)
        
        feed = page.locator("div[role='feed']")
        
        # Scroll logic to load enough listings
        logger.info("Scrolling the listings feed...")
        last_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 25
        
        while scroll_attempts < max_scroll_attempts:
            # Scroll down the feed container
            feed.evaluate("el => el.scrollTop = el.scrollHeight")
            page.wait_for_timeout(2000)
            
            # Count the number of listings currently loaded
            listings = page.locator("a[href*='/maps/place/']").all()
            current_count = len(listings)
            logger.info(f"Loaded {current_count} listings so far...")
            
            if current_count >= max_results:
                logger.info(f"Reached targeted results count ({current_count} >= {max_results})")
                break
                
            # If the count didn't increase, try scrolling a bit more, otherwise stop
            if current_count == last_count:
                # Scroll up slightly then down again to trigger lazy loading
                feed.evaluate("el => el.scrollTop = el.scrollHeight - 500")
                page.wait_for_timeout(1000)
                feed.evaluate("el => el.scrollTop = el.scrollHeight")
                page.wait_for_timeout(2000)
                
                # Check again
                listings = page.locator("a[href*='/maps/place/']").all()
                if len(listings) == last_count:
                    logger.info("No more new listings loading. Stopping scroll.")
                    break
            
            last_count = current_count
            scroll_attempts += 1
            
        # Re-fetch all matching listing links
        listing_links = page.locator("a[href*='/maps/place/']").all()
        logger.info(f"Found total {len(listing_links)} listings to process.")
        
        processed_names = set()
        
        for idx, link in enumerate(listing_links):
            if len(leads) >= max_results:
                break
                
            try:
                # Scroll the element into view and click it to load the detail panel
                link.scroll_into_view_if_needed()
                link.click()
                
                # Wait for the detail panel to load (we check for heading h1)
                page.wait_for_selector("h1.DUwDvf", timeout=5000)
                page.wait_for_timeout(1500) # Give panel components a moment to render
                
                # Extract Name
                name_elem = page.locator("h1.DUwDvf").first
                name = name_elem.inner_text().strip() if name_elem.count() > 0 else "N/A"
                
                if name == "N/A" or name in processed_names:
                    continue
                    
                processed_names.add(name)
                
                # Extract Phone Number
                # Google Maps uses a button with data-item-id="phone:tel:xxxx"
                phone = "N/A"
                phone_btn = page.locator("button[data-item-id^='phone:tel:']")
                if phone_btn.count() > 0:
                    raw_phone = phone_btn.first.get_attribute("data-item-id")
                    if raw_phone:
                        # Extract number portion and clean up formatting
                        phone = raw_phone.replace("phone:tel:", "").strip()
                        phone = phone.replace(" ", "").replace("-", "")
                        if not phone.startswith("+") and not phone.startswith("91"):
                            phone = f"+91{phone}"
                        elif phone.startswith("91") and not phone.startswith("+"):
                            phone = f"+{phone}"
                
                # Extract Address
                address = "N/A"
                address_btn = page.locator("button[data-item-id='address']")
                if address_btn.count() > 0:
                    address = address_btn.first.inner_text().strip()
                
                # Extract Rating
                rating = "N/A"
                rating_elem = page.locator("div.F7nice span[aria-hidden='true']").first
                if rating_elem.count() > 0:
                    rating = rating_elem.inner_text().strip()
                
                leads.append({
                    "name": name,
                    "phone": phone,
                    "rating": rating,
                    "address": address,
                    "category": query
                })
                logger.info(f"[{len(leads)}] Scraped: {name} | Phone: {phone} | Rating: {rating} | Address: {address[:40]}...")
                
                # Randomized sleep between clicks to prevent rate limiting
                time.sleep(random.uniform(1.5, 3.5))
                
            except Exception as item_e:
                logger.debug(f"Error scraping item details at index {idx}: {item_e}")
                continue
                
        browser.close()
        
    return leads

def save_leads_to_csv(leads_data, filename: str):
    output_dir = Path("data/leads")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    headers = ["name", "phone", "rating", "address", "category"]
    
    try:
        # Check if file exists to see if we should write header
        file_exists = filepath.exists()
        
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            writer.writerows(leads_data)
        logger.info(f"Successfully saved {len(leads_data)} leads to {filepath}")
    except Exception as e:
        logger.error(f"Error saving leads to CSV: {e}")

if __name__ == "__main__":
    # Categories to target
    queries = [
        "doctors in pune",
        "architects in pune",
        "civil engineers in pune",
        "interior designers in pune"
    ]
    
    # We scrape up to 10 leads per query to keep the execution time fast and safe
    output_file = "hni_leads_pune.csv"
    
    # Reset CSV file on start
    filepath = Path("data/leads") / output_file
    if filepath.exists():
        filepath.unlink()
        
    for query in queries:
        try:
            leads = scrape_google_maps(query, max_results=12)
            if leads:
                # Save immediately after each query scrape completes
                save_leads_to_csv(leads, output_file)
            time.sleep(random.uniform(3.0, 7.0))
        except Exception as q_e:
            logger.error(f"Failed to scrape query '{query}': {q_e}")
            continue
            
    logger.info("Scraping workflow completed successfully.")
