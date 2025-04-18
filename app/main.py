# app/main.py

import time
import os
import sys
from dotenv import load_dotenv
from ppadb.client import Client as AdbClient
import difflib

from helper_functions import (
    connect_device_remote,
    get_screen_resolution,
    capture_screenshot,
    extract_text_from_image_with_boxes,
    generate_joke_from_screenshots,
    tap,
    create_visual_debug_overlay,
    group_boxes_into_lines,
    group_lines_into_paragraphs,
    type_text_slow,
    isAtBottom,
    fuzzy_match_text,
    swipe,
)


def scroll_profile_and_capture(device, width, height, profile_num):
    """Scroll through profile and capture screenshots."""
    screenshots = []

    # Take initial screenshot
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_part1")
    screenshots.append(screenshot_path)

    # Scroll and capture fixed number of screenshots
    x_scroll = int(width * 0.5)  # Center of screen
    y_scroll_start = int(height * 0.84)  # Start at 84% of screen height
    # End at 16% of screen height (68% scroll distance)
    y_scroll_end = int(height * 0.16)

    # Take 5 more screenshots (6 total)
    for i in range(1, 6):
        print(f"Scroll down #{i}")
        # Scroll down
        swipe(device, x_scroll, y_scroll_start, x_scroll, y_scroll_end, 1500)

        # Take screenshot after scroll
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part{i+1}")
        screenshots.append(screenshot_path)

    # Scroll back up the same number of times (no screenshots needed)
    for i in range(1, 6):
        print(f"Scroll up #{i}")
        # Scroll up (reversed coordinates)
        swipe(device, x_scroll, y_scroll_end, x_scroll, y_scroll_start, 1500)

    return screenshots


def take_screenshot_and_extract_text(device, filename):
    """Take a screenshot and extract text from it."""
    screenshot_path = capture_screenshot(device, filename)
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        return None, None, None, None

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Create visualization
    create_visual_debug_overlay(
        screenshot_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/{filename}_visual.png"
    )

    return boxes, lines, paragraphs, screenshot_path


def process_screenshot_with_visualization(image_path, profile_num, screenshot_num):
    """Process a screenshot and create visualization overlay.

    Args:
        image_path: Path to the screenshot to process
        profile_num: Current profile number
        screenshot_num: Current screenshot number

    Returns:
        tuple: (paragraphs, visualization_path)
    """
    # Extract text boxes
    boxes = extract_text_from_image_with_boxes(image_path)
    if not boxes:
        print("No text boxes found in screenshot")
        return None, None

    # Group boxes into lines and paragraphs
    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Create visualization overlay
    base_name = os.path.splitext(image_path)[0]
    vis_path = create_visual_debug_overlay(
        image_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/profile_{profile_num}_screenshot_{screenshot_num}_visual.png"
    )
    print(f"Created visualization: {vis_path}")

    return paragraphs, vis_path


def capture_profile_screenshots(device, count):
    """Capture screenshots while scrolling through the profile."""
    screenshots = []
    last_position = None

    # Take initial screenshot
    screenshot_path = capture_screenshot(device, f"profile_{count}_part0")
    screenshots.append(screenshot_path)

    # Scroll and capture until bottom
    width, height = get_screen_resolution(device)
    x_center = width // 2
    y_start = height * 0.8
    y_end = height * 0.7

    i = 1
    while True:
        # Scroll down and check if we can continue
        can_scroll, last_position = swipe(
            device, x_center, y_start, x_center, y_end, 100, last_position)
        if not can_scroll:
            break

        # Take screenshot after scroll
        screenshot_path = capture_screenshot(
            device, f"profile_{count}_part{i}")
        screenshots.append(screenshot_path)
        i += 1

    return screenshots


def find_and_click_prompt_response(device, target_prompt, target_response, joke, screenshot_index, coordinates, profile_num):
    """Find and click on the prompt/response location based on provided coordinates."""
    width, height = get_screen_resolution(device)

    # First scroll to top
    swipe(device, width//2, height*0.2, width//2, height*0.8, 100)
    time.sleep(1)

    # Use the same scrolling logic as scroll_profile_and_capture
    x_scroll = int(width * 0.5)  # Center of screen
    y_scroll_start = int(height * 0.84)  # Start at 84% of screen height
    y_scroll_end = int(height * 0.16)    # End at 16% of screen height

    # Scroll down the exact number of times needed to reach the target screenshot
    for i in range(screenshot_index):
        print(f"Scrolling to position {i+1} of {screenshot_index}")
        swipe(device, x_scroll, y_scroll_start, x_scroll, y_scroll_end, 1500)
        time.sleep(1)  # Wait for scroll to complete

    # Double click at the provided coordinates
    tap(device, coordinates["x"], coordinates["y"])
    time.sleep(0.1)  # Small delay between clicks
    tap(device, coordinates["x"], coordinates["y"])
    time.sleep(1)

    # Type joke
    type_text_slow(device, joke)
    print(
        f"Double-clicked at coordinates ({coordinates['x']}, {coordinates['y']}) in screenshot {screenshot_index}")
    return True


def main():
    device_ip = os.getenv("DEVICE_IP", "192.168.12.32")
    print(f"Connecting to device at IP: {device_ip}")

    device = connect_device_remote(device_ip)
    if not device:
        print("Failed to connect to device")
        sys.exit(1)

    width, height = get_screen_resolution(device)
    print(f"Screen resolution: {width}x{height}")

    # Process profiles until stopped
    profile_num = 1
    while True:
        print(f"\nProcessing profile #{profile_num}")
        print("="*50)

        # Step 1: Scroll and capture screenshots
        print("\nStep 1: Capturing screenshots...")
        screenshots = scroll_profile_and_capture(
            device, width, height, profile_num)
        print(f"Captured {len(screenshots)} screenshots")

        # Step 2: Generate joke
        print("\nStep 2: Generating joke...")
        result = generate_joke_from_screenshots(
            screenshots=screenshots,
            format_txt_path=os.path.join(
                os.path.dirname(__file__), 'format.txt'),
            prompts_txt_path=os.path.join(
                os.path.dirname(__file__), 'prompts.txt'),
            captions_txt_path=os.path.join(
                os.path.dirname(__file__), 'captions.txt'),
            polls_txt_path=os.path.join(
                os.path.dirname(__file__), 'polls.txt')
        )

        if not result:
            print("Failed to generate joke")
            profile_num += 1
            continue

        prompt = result.get("prompt", "")
        response = result.get("response", "")
        joke = result.get("joke", "")
        screenshot_index = result.get("screenshot_index", 0)
        coordinates = result.get("coordinates", {"x": 0, "y": 0})

        if not prompt or not joke:
            print("Invalid result from joke generator")
            profile_num += 1
            continue

        print(f"\nGenerated joke: {joke}")
        print(f"Target prompt: {prompt}")
        print(f"Target response: {response}")
        print(f"Screenshot index: {screenshot_index}")
        print(f"Coordinates: {coordinates}")

        # Step 3: Find and click prompt-response pair
        print("\nStep 3: Finding and clicking prompt-response pair...")
        success = find_and_click_prompt_response(
            device, prompt, response, joke, screenshot_index, coordinates, profile_num)
        if not success:
            print("Failed to find and click prompt-response pair")
            profile_num += 1
            continue

        print("\nSuccessfully processed profile!")
        profile_num += 1


if __name__ == "__main__":
    main()
