import os
from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    # follows Streamlit's /-/auth redirect handshake automatically
    page.goto(URL, wait_until="networkidle", timeout=90000)

    # If the app is asleep, click the wake button.
    clicked = False
    for name in ["Yes, get this app back up", "get this app back up", "app back up"]:
        btn = page.get_by_role("button", name=name)
        if btn.count() > 0:
            btn.first.click()
            print("Clicked wake button.")
            clicked = True
            break
    if not clicked:
        print("App already awake (no wake button found).")

    # Hold a real session so the inactivity timer resets.
    page.wait_for_timeout(45000)
    print("Done.")
    browser.close()
