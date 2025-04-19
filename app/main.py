# app/main.py

import time
import os
import sys
from dotenv import load_dotenv
from helper_functions import (
    connect_device_remote,
    get_screen_resolution,
    capture_screenshot,
    swipe,
    generate_joke_from_screenshots
)


def scroll_profile_and_capture(device, width, height, profile_num):
    """Scroll through profile and capture screenshots."""
    screenshots = []

    # Take initial screenshot
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_part1")
    screenshots.append(screenshot_path)

    # Take 5 more screenshots (6 total)
    for i in range(1, 6):
        print(f"Scroll down #{i}")
        # Scroll down
        swipe(device, "down")
        time.sleep(1)  # Wait for scroll to complete

        # Take screenshot after scroll
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part{i+1}")
        screenshots.append(screenshot_path)

    return screenshots


def main():
    """Main function to run the Hinge automation."""
    try:
        # Get device IP from environment variable
        device_ip = os.getenv("DEVICE_IP", "192.168.12.32")
        print(f"Connecting to device at IP: {device_ip}")

        # Connect to device
        device = connect_device_remote(device_ip)
        if not device:
            print("Failed to connect to device")
            sys.exit(1)

        # Get screen dimensions
        width, height = get_screen_resolution(device)
        print(f"Screen resolution: {width}x{height}")

        # Take screenshots of the profile
        print("\nTaking screenshots of profile...")
        screenshots = scroll_profile_and_capture(device, width, height, 1)

        # Get absolute paths for resource files
        format_txt_path = os.path.join(os.path.dirname(__file__), "format.txt")
        prompts_txt_path = os.path.join(
            os.path.dirname(__file__), "prompts.txt")
        captions_txt_path = os.path.join(
            os.path.dirname(__file__), "captions.txt")
        polls_txt_path = os.path.join(os.path.dirname(__file__), "polls.txt")

        # Use AI to determine if person is desirable
        print("\nAnalyzing profile...")
        result = generate_joke_from_screenshots(
            screenshots,
            format_txt_path,
            prompts_txt_path,
            captions_txt_path,
            polls_txt_path
        )

        # If result is None or has empty fields, person is not desirable
        is_desirable = result is not None and all([
            result.get("prompt"),
            result.get("response"),
            result.get("conversation_starter")
        ])

        print(
            f"\nProfile analysis complete. Person is {'desirable' if is_desirable else 'not desirable'}.")
        return is_desirable

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    main()
