import pytest

pytest.importorskip("playwright")
from playwright.sync_api import Page, expect

# Test configuration
BASE_URL = "http://localhost:5173"


@pytest.fixture(scope="function")
def authenticated_page(page: Page):
    """Create a page with authenticated session using dev login."""
    page.goto(f"{BASE_URL}/login")
    dev_login = page.locator("text=Demo Admin")
    expect(dev_login).to_be_visible(timeout=5000)
    dev_login.click()
    page.wait_for_url(f"{BASE_URL}/", timeout=10000)
    return page


def test_annotation_flow_pairwise(authenticated_page: Page):
    """Test the complete annotation flow for a pairwise task."""

    # 1. Navigate to Annotations page
    authenticated_page.goto(f"{BASE_URL}/annotations")

    # 2. Verify page load
    expect(authenticated_page.get_by_role("heading", name="Annotations")).to_be_visible(
        timeout=10000
    )

    # 3. Wait for task to load (look for specific pairwise UI elements)
    try:
        # First verify ANY content from the task is loaded (e.g. Prompt)
        # Use a regex to match "Prompt" followed by any number
        expect(authenticated_page.locator("text=/Prompt \\d+/")).to_be_visible(
            timeout=10000
        )

        # Then verify the pairwise specific elements
        expect(authenticated_page.locator("text=Response A")).to_be_visible(
            timeout=5000
        )
        expect(authenticated_page.locator("text=Response B")).to_be_visible()
    except Exception as e:
        print("\nPage Content at Failure:\n")
        print(authenticated_page.content())
        raise e

    # 4. Select a winner (Response A)
    # The card has "Response A" text, we click the card
    response_a_card = (
        authenticated_page.locator(".cursor-pointer")
        .filter(has_text="Response A")
        .first
    )
    response_a_card.click()

    # 5. Verify selection visual cue (border change or checkmark)
    # The component adds 'ring-2 ring-primary' class
    # But checking class is brittle, let's checking for checkmark icon if visible
    # Or just proceed to submit

    # 6. Fill optional rationale (good for testing input)
    authenticated_page.fill("textarea", "Test rationale for automation")

    # 7. Submit
    submit_btn = authenticated_page.get_by_role("button", name="Submit Annotation")
    expect(submit_btn).to_be_enabled()
    submit_btn.click()

    # 8. Verify success toast
    expect(authenticated_page.locator("text=Annotation submitted")).to_be_visible(
        timeout=5000
    )

    # 9. Verify new task loads (or same task mechanism refreshes)
    # We can check that the selection is cleared
    # After submission, selectedWinner is set to null
    # So "Response A" card should not have the ring class, or just re-appear
    # Let's wait for toast to disappear or check for a visual reset
    # A simple check is that the submit button becomes disabled again (since selection is cleared)
    expect(submit_btn).to_be_disabled(timeout=5000)
