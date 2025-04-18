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


def find_and_click_prompt_response(device, target_prompt, target_response, joke, profile_num):
    """Find and click on the best matching prompt or response, then input the joke."""
    # Scroll to top first
    width, height = get_screen_resolution(device)
    swipe(device, width//2, height*0.2, width//2, height*0.8, 100)
    time.sleep(1)

    # Search for prompt while scrolling
    last_position = None
    search_count = 0
    while True:
        # Extract text from current screen
        screenshot_path = capture_screenshot(
            device, f"temp_prompt_search_{profile_num}_{search_count}")
        boxes = extract_text_from_image_with_boxes(screenshot_path)
        if not boxes:
            print("No text boxes found in screenshot")
            continue

        # Group boxes into lines and paragraphs
        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Find best match for either prompt or response
        best_match = None
        best_confidence = 0.8  # Minimum confidence threshold
        best_text = ""
        tap_x = 0
        tap_y = 0

        for para in paragraphs:
            # Check prompt match
            prompt_match, prompt_confidence, matched_prompt = fuzzy_match_text(
                target_prompt, para['text'])
            if prompt_match and prompt_confidence > best_confidence:
                best_match = para
                best_confidence = prompt_confidence
                best_text = "prompt"
                tap_x = (para['boxes'][0]['box'][0] +
                         para['boxes'][-1]['box'][2]) // 2
                tap_y = (para['boxes'][0]['box'][1] +
                         para['boxes'][-1]['box'][3]) // 2

            # Check response match
            response_match, response_confidence, matched_response = fuzzy_match_text(
                target_response, para['text'])
            if response_match and response_confidence > best_confidence:
                best_match = para
                best_confidence = response_confidence
                best_text = "response"
                tap_x = (para['boxes'][0]['box'][0] +
                         para['boxes'][-1]['box'][2]) // 2
                tap_y = (para['boxes'][0]['box'][1] +
                         para['boxes'][-1]['box'][3]) // 2

        # Create visualization overlay for debugging prompt search
        vis_path = create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/prompt_search_visual_{profile_num}_{search_count}.png",
            tap_target=(tap_x, tap_y) if best_match else None
        )
        print(f"Created visualization for prompt search: {vis_path}")

        if best_match:
            # Found a good match, click it
            tap(device, tap_x, tap_y)
            time.sleep(1)

            # Type joke
            type_text_slow(device, joke)
            print(
                f"Found matching {best_text} with confidence {best_confidence:.2f}")
            return True

        # Scroll down and check if we can continue
        can_scroll, last_position = swipe(
            device, width//2, height*0.8, width//2, height*0.7, 100, last_position)
        if not can_scroll:
            break

        search_count += 1

    return False


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

        if not prompt or not joke:
            print("Invalid result from joke generator")
            profile_num += 1
            continue

        print(f"\nGenerated joke: {joke}")
        print(f"Target prompt: {prompt}")
        print(f"Target response: {response}")

        # Step 3: Find and click prompt-response pair with visualization
        print("\nStep 3: Finding and clicking prompt-response pair...")
        success = find_and_click_prompt_response(
            device, prompt, response, joke, profile_num)
        if not success:
            print("Failed to find and click prompt-response pair")
            profile_num += 1
            continue

        print("\nSuccessfully processed profile!")
        profile_num += 1


if __name__ == "__main__":
    main()
