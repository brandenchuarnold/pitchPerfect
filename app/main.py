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

    # Take 5 more screenshots (6 total)
    for i in range(1, 6):
        print(f"Scroll down #{i}")
        # Scroll down
        swipe(device, "down")

        # Take screenshot after scroll
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part{i+1}")
        screenshots.append(screenshot_path)

    # Scroll back up the same number of times (no screenshots needed)
    for i in range(1, 6):
        print(f"Scroll up #{i}")
        # Scroll up
        swipe(device, "up")

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


def scroll_to_screenshot(device, screenshot_index):
    """Scroll to the specified screenshot index using consistent 68% swipe distance."""
    # We're already at the top from scroll_profile_and_capture
    # Just scroll down to the target screenshot
    print(f"\nScrolling to screenshot index {screenshot_index}...")
    for i in range(screenshot_index):
        print(f"Scroll down #{i+1}")
        swipe(device, "down")
        time.sleep(1)  # Wait for scroll to complete

    return True


def detect_prompt_in_screenshot(device, target_prompt, screenshot_index, profile_num):
    """Detect and visualize the target prompt in a screenshot.

    Args:
        device: The ADB device
        target_prompt: The prompt text we're looking for
        screenshot_index: Index of the screenshot to analyze
        profile_num: Current profile number

    Returns:
        tuple: (found, tap_coordinates) where:
            - found: bool indicating if prompt was found
            - tap_coordinates: (x,y) coordinates to tap if found, None if not found
    """
    # Take a fresh screenshot at this position
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_prompt_detection")

    # Extract text and group into paragraphs
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        print("No text boxes found in screenshot")
        return False, None

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Debug print: Show all paragraphs and their match scores
    print("\nDebug: Comparing target prompt against OCR paragraphs:")
    print(f"Target prompt: '{target_prompt}'")
    print("\nOCR paragraphs found:")
    for i, para in enumerate(paragraphs):
        is_match, ratio, matched_text = fuzzy_match_text(
            target_prompt, para['text'])
        print(f"Paragraph {i+1}:")
        print(f"  Text: '{para['text']}'")
        print(f"  Match ratio: {ratio:.2f}")
        print(f"  Is match: {is_match}")
        print()

    # Try to find the target prompt in paragraphs
    best_match = None
    best_ratio = 0.0

    for para in paragraphs:
        is_match, ratio, matched_text = fuzzy_match_text(
            target_prompt, para['text'])
        if is_match and ratio > best_ratio:
            best_match = para
            best_ratio = ratio

    if best_match:
        print(f"Found prompt match with ratio {best_ratio:.2f}")

        # Calculate tap coordinates (center of the paragraph)
        boxes = best_match['boxes']
        min_x = min(box['box'][0] for box in boxes)
        max_x = max(box['box'][0] + box['box'][2] for box in boxes)
        min_y = min(box['box'][1] for box in boxes)
        max_y = max(box['box'][1] + box['box'][3] for box in boxes)

        tap_x = (min_x + max_x) // 2
        tap_y = (min_y + max_y) // 2

        # Create visualization with tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_prompt_detection_visual.png",
            tap_target=(tap_x, tap_y)
        )

        # Execute double tap at the calculated coordinates
        tap(device, tap_x, tap_y, double_tap=True)
        time.sleep(1)  # Wait for response interface to open

        return True, (tap_x, tap_y)
    else:
        print("No matching prompt found in screenshot")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_prompt_detection_visual.png"
        )
        return False, None


def match_prompt_against_authoritative(prompt, prompts_txt_path):
    """Match the AI's prompt against the authoritative prompts in prompts.txt.

    Args:
        prompt: The prompt text returned by the AI
        prompts_txt_path: Path to prompts.txt containing authoritative prompts

    Returns:
        tuple: (matched_prompt, confidence) where:
            - matched_prompt: The best matching prompt from prompts.txt
            - confidence: The confidence ratio of the match
    """
    try:
        with open(prompts_txt_path, 'r') as f:
            authoritative_prompts = f.read().splitlines()
    except Exception as e:
        print(f"Error reading prompts.txt: {e}")
        return None, 0.0

    best_match = None
    best_ratio = 0.0

    # Print all prompts that have a decent match for debugging
    print("\nChecking prompt matches:")
    for auth_prompt in authoritative_prompts:
        is_match, ratio, matched_text = fuzzy_match_text(
            prompt, auth_prompt, threshold=0.5)  # Lower threshold for debugging
        if ratio > 0.5:  # Show any decent matches
            print(f"  {auth_prompt}: {ratio:.2f}")
        if is_match and ratio > best_ratio:
            best_match = auth_prompt
            best_ratio = ratio

    if best_match:
        print(f"\nBest match:")
        print(f"AI prompt: '{prompt}'")
        print(f"Auth prompt: '{best_match}'")
        print(f"Confidence: {best_ratio:.2f}")
        return best_match, best_ratio
    else:
        print(f"\nNo good matches found for '{prompt}'")
        return None, 0.0


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

        # Step 2: Generate conversation starter
        print("\nStep 2: Generating conversation starter...")
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
            print("Failed to generate conversation starter")
            profile_num += 1
            continue

        prompt = result.get("prompt", "")
        conversation_starter = result.get("conversation_starter", "")
        screenshot_index = result.get("screenshot_index", 0)

        if not prompt or not conversation_starter:
            print("Invalid result from conversation starter generator")
            profile_num += 1
            continue

        print(f"\nGenerated conversation starter: {conversation_starter}")
        print(f"Target prompt: {prompt}")
        print(f"Screenshot index: {screenshot_index}")

        # Step 3: Match prompt against authoritative prompts
        print("\nStep 3: Matching prompt against authoritative prompts...")
        matched_prompt, confidence = match_prompt_against_authoritative(
            prompt,
            os.path.join(os.path.dirname(__file__), 'prompts.txt')
        )

        if not matched_prompt or confidence < 0.8:
            print(
                f"Could not find a good match for prompt '{prompt}' in prompts.txt")
            profile_num += 1
            continue

        print(f"Matched to authoritative prompt: {matched_prompt}")

        # Step 4: Scroll to the correct screenshot using the same 68% swipe distance
        print("\nStep 4: Scrolling to target screenshot...")
        success = scroll_to_screenshot(device, screenshot_index)
        if not success:
            print("Failed to scroll to target screenshot")
            profile_num += 1
            continue

        # Step 5: Detect and visualize the target prompt in the screenshot
        print("\nStep 5: Detecting and visualizing target prompt...")
        found, tap_coordinates = detect_prompt_in_screenshot(
            device, matched_prompt, screenshot_index, profile_num)
        if not found:
            print("Failed to detect target prompt in screenshot")
            profile_num += 1
            continue

        profile_num += 1


if __name__ == "__main__":
    main()
