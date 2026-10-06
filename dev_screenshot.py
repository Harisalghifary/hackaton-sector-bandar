"""Dev-only visual check: screenshot the running app (dark + light, boot + refreshed).
Not part of the test suite; uses system Chrome via Playwright."""
from playwright.sync_api import sync_playwright

OUT = "/private/var/folders/xj/48c5hwbj51ldcgy7rklnfzcm0000gn/T/opencode/ui_shots"


def shoot(url: str, path: str, click_refresh: bool = False) -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1600, "height": 1200})
        pg.goto(url)
        pg.wait_for_selector('[data-testid="stMetricValue"]', timeout=45000)
        pg.wait_for_timeout(1500)
        if click_refresh:
            pg.click('button:has-text("Force Live Refresh")')
            pg.wait_for_selector(".bd-table", timeout=90000)
            pg.wait_for_timeout(1200)
        pg.screenshot(path=path, full_page=True)
        b.close()
        print("shot:", path)


shoot("http://localhost:8599", f"{OUT}/dark_boot.png")
shoot("http://localhost:8599", f"{OUT}/dark_refreshed.png", click_refresh=True)
shoot("http://localhost:8600", f"{OUT}/light_boot.png")
