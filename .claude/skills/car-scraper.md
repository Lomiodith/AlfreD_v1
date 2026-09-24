---
name: car-scraper
description: Run the car scraper to search Autovit.ro and OLX.ro for cars matching criteria (<=2000cc, >=170HP, <=220k km, 4x4, <=12k EUR)
user_invocable: true
---

# Car Scraper Skill

Run the car scraper script that searches Autovit.ro and OLX.ro for cars matching the configured criteria.

## Steps

1. Run the car scraper script:
   ```
   python car_scraper.py
   ```
2. After the script completes, read `car_results.json` to see the results.
3. Present a summary to the user:
   - Total number of matching cars found
   - For each car: title, price, year, mileage, engine/HP, location, and URL
   - Sort by price (lowest first)
4. If no matches were found, report that and suggest relaxing filters.
