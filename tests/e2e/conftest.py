"""
Playwright E2E Test Configuration

Provides fixtures and configuration for E2E tests using Playwright.
"""

import pytest

try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


if PLAYWRIGHT_AVAILABLE:
    @pytest.fixture(scope="session")
    def browser():
        """Create browser instance for session."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()

    @pytest.fixture(scope="function")
    def context(browser: Browser):
        """Create browser context for each test."""
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
        )
        yield context
        context.close()

    @pytest.fixture(scope="function")
    def page(context: BrowserContext):
        """Create page for each test."""
        page = context.new_page()
        yield page
        page.close()


# Test markers
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (deselect with '-m \"not e2e\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
