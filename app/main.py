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
)


def scroll_profile_and_capture(device, width, height, profile_num):
    """Scroll through profile and capture screenshots until bottom is reached."""
    screenshots = []
    x_scroll = int(width * 0.5)  # Center of screen
    y_scroll_start = int(height * 0.75)
    y_scroll_end = int(height * 0.08)

    # Initial screenshot
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_part1")
    screenshots.append(screenshot_path)

    # Scroll until bottom is reached
    i = 0
    while True:
        print(f"Scroll #{i+1}")
        device.shell(
            f"input swipe {x_scroll} {y_scroll_start} {x_scroll} {y_scroll_end} 1500")
        time.sleep(0.5)

        if isAtBottom(device):
            print("Reached bottom of profile")
            break

        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part{i+2}")
        screenshots.append(screenshot_path)

        i += 1
        if i >= 8:  # Safety limit
            print("Reached maximum scroll limit")
            break

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


def process_screenshot_with_vsualization(image_path):
    """Process a screenshot and create visualization."""
    boxes = extract_text_from_image_with_boxes(image_path)
    if not boxes:
        return None, None

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Create visualization
    vis_path = create_visual_debug_overlay(
        image_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"{image_path}_visual.png"
    )

    return paragraphs, vis_path


def find_and_click_prompt(device, width, height, target_text):
    """Find and click on matching text using sophisticated fuzzy matching.

    Process:
    1. Compare target text against paragraphs using multiple matching strategies
    2. For each paragraph:
       a) Try substring match (needle in haystack)
       b) Try reverse substring match (haystack in needle)
       c) Use ratio-based fuzzy matching with difflib
    3. Select best match above threshold (0.8)
    4. Handle multi-line matches within paragraphs
    5. Double-click the matched element if found
    """
    x_scroll = int(width * 0.5)
    y_scroll_start = int(height * 0.75)
    y_scroll_end = int(height * 0.08)

    # Initialize best match tracking
    best_match = None
    best_ratio = 0.8  # Threshold for fuzzy matching
    best_paragraph = None

    for _ in range(8):  # Max 8 scrolls
        # Take screenshot and extract text
        boxes, lines, paragraphs, _ = take_screenshot_and_extract_text(
            device, "search")
        if not paragraphs:
            continue

        # Check each paragraph for matches using multiple strategies
        for para in paragraphs:
            para_text = para['text'].lower()
            target_lower = target_text.lower()

            # Use the helper function's fuzzy matching
            is_match, ratio, matched = fuzzy_match_text(
                target_lower, para_text)

            if is_match and ratio > best_ratio:
                best_ratio = ratio
                best_paragraph = para
                print(f"Found match with ratio {ratio:.2f}")
                break

            # Try matching individual lines within the paragraph
            for line in para['lines']:
                line_text = ' '.join(box['text'] for box in line).lower()
                is_match, line_ratio, matched = fuzzy_match_text(
                    target_lower, line_text)
                if is_match and line_ratio > best_ratio:
                    best_ratio = line_ratio
                    best_paragraph = para
                    print(
                        f"Found line-level match with ratio {line_ratio:.2f}")
                    break

        # If we found a good match, click it
        if best_paragraph and best_ratio >= 0.8:
            # Get the first box in the paragraph
            box = best_paragraph['boxes'][0]['box']

            # Calculate click position: center of the first box
            x = box[0] + (box[2] // 2)
            y = box[1] + (box[3] // 2)

            print(
                f"Found match with confidence {best_ratio:.2f} at coordinates ({x}, {y})")

            # Double-click the match
            tap(device, x, y)
            time.sleep(0.1)
            tap(device, x, y)
            return True

        # Scroll down for next attempt
        device.shell(
            f"input swipe {x_scroll} {y_scroll_start} {x_scroll} {y_scroll_end} 1500")
        time.sleep(0.5)

    print(f"No match found above threshold {best_ratio:.2f}")
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
        for i, screenshot in enumerate(screenshots):
            print(f"  Screenshot {i+1}: {screenshot}")

        # Step 2: Generate joke
        print("\nStep 2: Generating joke...")
        result = generate_joke_from_screenshots(
            screenshots=screenshots,
            format_txt_path='app/data/format.txt',
            prompts_txt_path='app/data/prompts.txt',
            captions_txt_path='app/data/captions.txt',
            polls_txt_path='app/data/polls.txt'
        )

        if not result:
            print("Failed to generate joke")
            break

        target_text = result.get('prompt', '') or result.get('response', '')
        joke = result.get('joke', '')

        if not target_text or not joke:
            print("Invalid result from joke generator")
            break

        print("\nGenerated content:")
        print(f"Target text: {target_text}")
        print(f"Joke: {joke}")

        # Step 3: Scroll to top and search for match
        print("\nStep 3: Searching for match...")
        # Scroll to top
        for _ in range(3):
            device.shell(
                f"input swipe {width//2} {height//4} {width//2} {height*3//4} 1500")
            time.sleep(0.5)

        if not find_and_click_prompt(device, width, height, target_text):
            print("Could not find match, falling back to center tap")
            # Double-tap center of screen
            tap(device, width//2, height//2)
            time.sleep(0.1)
            tap(device, width//2, height//2)

        # Type joke but don't send
        print("\nStep 4: Typing joke (not sending)...")
        time.sleep(1)  # Wait for prompt to open
        type_text_slow(device, joke, per_char_delay=0.1)
        print("Joke typed successfully")
        print("\nStopping here for testing. Press Ctrl+C to exit.")

        # Wait for user to review
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        profile_num += 1


if __name__ == "__main__":
    main()
