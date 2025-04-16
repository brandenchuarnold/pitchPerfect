# app/main.py

import time
import os
import sys
import openai
import random
from dotenv import load_dotenv
import re
import argparse
from pytesseract import image_to_data
import math
import difflib
from ppadb.client import Client as AdbClient

# Import your config
from config import OPENAI_API_KEY

# Import your existing helper functions
from helper_functions import (
    connect_device_remote,
    get_screen_resolution,
    capture_screenshot,
    extract_text_from_image,
    generate_comment_with_target,
    tap,
)

openai.api_key = OPENAI_API_KEY


def take_screenshot_and_extract_text(device, filename, profile_count, section_name):
    """Take a screenshot and extract text from it"""
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_count}_{filename}")
    text = extract_text_from_image(screenshot_path).strip()
    if text:
        print(f"Extracted text from {section_name} ({len(text)} chars)")
    return text


def scroll_profile_and_capture(device, width, height, profile_num, capture_screenshots=False):
    """Shared function for scrolling through a profile, optionally capturing screenshots"""
    all_profile_text = ""
    x_scroll = int(width * 0.5)  # Center of screen
    y_scroll_start = int(height * 0.75)
    y_scroll_end = int(height * 0.08)

    # Initial screenshot if needed
    if capture_screenshots:
        text = take_screenshot_and_extract_text(
            device, "part1", profile_num, "top of profile")
        if text:
            all_profile_text += text + "\n"
            print(f"Initial text length: {len(text)} chars")

    # Perform 8 scrolls instead of 7
    for i in range(8):
        print(f"Scroll #{i+1}")
        device.shell(
            f"input swipe {x_scroll} {y_scroll_start} {x_scroll} {y_scroll_end} 1500")
        time.sleep(0.5)

        if capture_screenshots:
            text = take_screenshot_and_extract_text(
                device, f"part{i+2}", profile_num, f"after scroll {i+1}")
            if text:
                all_profile_text += text + "\n"
                print(f"Text length after scroll {i+1}: {len(text)} chars")

    return all_profile_text.strip() if capture_screenshots else None


def scroll_up_to_top(device, width, height):
    """Scrolls up multiple times to ensure we are at the top."""
    print("Scrolling to top...")
    scroll_y_start = int(height * 0.2)
    scroll_y_end = int(height * 0.8)
    center_x = int(width / 2)
    for _ in range(5):  # Scroll up 5 times should be enough
        device.shell(
            f"input swipe {center_x} {scroll_y_start} {center_x} {scroll_y_end} 300")
        time.sleep(0.3)
    time.sleep(1)  # Wait for scrolling to settle


def scroll_once_hinge(device, width, height):
    """Performs one coarse downward scroll for initial data gathering (Trying 70% -> 10% height = 60% coverage, 800ms duration)."""
    scroll_y_start = int(height * 0.70)  # Start at 70% down
    scroll_y_end = int(height * 0.10)  # End at 10% down (Covers 60% height)
    center_x = int(width / 2)
    swipe_duration = 800  # Keep duration (milliseconds)
    # print(f"  Coarse scroll: {scroll_y_start} -> {scroll_y_end} ({swipe_duration}ms)") # Optional debug print
    device.shell(
        f"input swipe {center_x} {scroll_y_start} {center_x} {scroll_y_end} {swipe_duration}")
    # Keep time.sleep in the calling loop


def scroll_precisely(device, width, height, distance):
    """Scrolls precisely up or down by the specified pixel distance using small swipes."""
    if distance == 0:
        return

    direction = 1 if distance > 0 else -1  # 1 for down, -1 for up
    abs_distance = abs(distance)
    print(
        f"Precise scroll: {'Down' if direction == 1 else 'Up'} by {abs_distance}px")

    small_swipe_pixels = int(height * 0.15)  # Adjust swipe size if needed
    num_small_swipes = math.ceil(abs_distance / small_swipe_pixels)

    center_x = int(width / 2)
    swipe_duration = 200  # milliseconds

    for i in range(num_small_swipes):
        remaining = abs_distance - (i * small_swipe_pixels)
        current_swipe_distance = min(remaining, small_swipe_pixels)

        if direction == 1:  # Scroll Down
            y_start = int(height * 0.6)  # Start swipe lower middle
            y_end = y_start - current_swipe_distance
        else:  # Scroll Up
            y_start = int(height * 0.4)  # Start swipe upper middle
            y_end = y_start + current_swipe_distance

        # Ensure y_end doesn't go off-screen
        y_end = max(5, min(height - 5, y_end))
        y_start = max(5, min(height - 5, y_start))

        # Don't swipe if start and end are too close
        if abs(y_start - y_end) < 5:
            continue

        print(
            f"  Swipe {i+1}/{num_small_swipes}: {y_start} -> {y_end} ({current_swipe_distance}px)")
        device.shell(
            f"input swipe {center_x} {y_start} {center_x} {y_end} {swipe_duration}")
        time.sleep(0.2)  # Small pause between swipes

    time.sleep(1)  # Wait for final scroll to settle


def send_dislike_hinge(device, width, height):
    """Sends a dislike (taps X)."""
    # Implement tapping the X button coordinates
    x_dislike = int(width * 0.15)  # Example coordinates
    y_dislike = int(height * 0.85)
    print(f"Sending dislike by tapping ({x_dislike}, {y_dislike})")
    tap(device, x_dislike, y_dislike)
    time.sleep(1)


def send_like_center_tap(device, width, height):
    """Sends a like by double-tapping the center of the screen."""
    center_x = int(width * 0.5)
    center_y = int(height * 0.5)
    print(f"Sending like by double-tapping center ({center_x}, {center_y})")
    tap(device, center_x, center_y)
    time.sleep(0.1)
    tap(device, center_x, center_y)
    time.sleep(1)


def prepare_comment_for_input(comment):
    """Escapes a comment string for adb shell input text."""
    # First, escape special characters that could break the shell command
    escaped = comment.replace("'", "\\'").replace("\"", "\\\"").replace("(", "\\(").replace(
        ")", "\\)").replace("&", "\\&").replace("<", "\\<").replace(">", "\\>").replace("|", "\\|").replace(";", "\\;").replace("-", "\\-")

    # Handle various types of dashes and hyphens
    escaped = escaped.replace("—", "\\—")  # Em dash
    escaped = escaped.replace("–", "\\–")  # En dash
    escaped = escaped.replace("−", "\\−")  # Minus sign
    escaped = escaped.replace("‐", "\\‐")  # Hyphen
    escaped = escaped.replace("‑", "\\‑")  # Non-breaking hyphen
    escaped = escaped.replace("‒", "\\‒")  # Figure dash
    escaped = escaped.replace("―", "\\―")  # Horizontal bar
    escaped = escaped.replace("﹘", "\\﹘")  # Small em dash
    escaped = escaped.replace("﹨", "\\﹨")  # Small reverse solidus

    # Handle various types of apostrophes and quotes
    escaped = escaped.replace("'", "\\'")  # Straight apostrophe
    escaped = escaped.replace("'", "\\'")  # Curly apostrophe
    escaped = escaped.replace("'", "\\'")  # Smart apostrophe
    escaped = escaped.replace("'", "\\'")  # Single quote
    escaped = escaped.replace("'", "\\'")  # Single closing quote

    # Handle semicolons and similar characters
    escaped = escaped.replace(";", "\\;")  # Regular semicolon
    escaped = escaped.replace("；", "\\；")  # Full-width semicolon
    escaped = escaped.replace("﹔", "\\﹔")  # Small semicolon

    # Then, replace spaces with %s for proper input
    escaped = escaped.replace(" ", "%s")

    # Finally, wrap in single quotes to handle any remaining special characters
    return f"'{escaped}'"


def extract_text_from_image_with_boxes(image_path):
    """Extract text and bounding boxes from an image using OCR."""
    try:
        data = image_to_data(image_path, output_type='dict',
                             config='--psm 6')  # Assume paragraphs/blocks
        boxes = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            # level 5 is word, maybe filter by confidence or block level (level 2/3)?
            # Let's aggregate words into lines or blocks if needed, or just use detected boxes.
            # For now, let's take boxes with reasonable confidence and non-empty text.
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            if conf > 40 and text:  # Confidence threshold (adjust as needed)
                (x, y, w, h) = (data['left'][i], data['top']
                                [i], data['width'][i], data['height'][i])
                if w > 0 and h > 0:  # Ensure valid dimensions
                    boxes.append(
                        {'text': text, 'box': (x, y, w, h), 'conf': conf})
        # TODO: Consider merging adjacent boxes for better prompt matching?
        return boxes
    except Exception as e:
        print(f"Error during OCR: {e}")
        return []


def group_boxes_into_paragraphs(boxes, y_threshold=10, x_threshold=50):
    """Group text boxes into lines and paragraphs based on their spatial relationships.

    Args:
        boxes: List of text boxes with 'text' and 'box' fields
        y_threshold: Maximum vertical distance (in pixels) for boxes to be considered on the same line
        x_threshold: Maximum horizontal distance (in pixels) for boxes to be considered part of the same paragraph

    Returns:
        List of paragraphs, where each paragraph contains:
        - text: Combined text of all boxes in the paragraph
        - boxes: List of all boxes in the paragraph
        - lines: List of lines in the paragraph, where each line contains boxes with similar y-coordinates
    """
    if not boxes:
        return []

    # Sort boxes by y-coordinate first, then by x-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: (b['box'][1], b['box'][0]))

    paragraphs = []
    current_paragraph = []
    current_line = []
    current_y = None

    for box in sorted_boxes:
        box_y = box['box'][1]  # Get y-coordinate from box position
        if current_y is None:
            # First box - start new line and paragraph
            current_y = box_y
            current_line = [box]
            current_paragraph = [current_line]
        else:
            # Check if box is on the same line (within y_threshold)
            if abs(box_y - current_y) <= y_threshold:
                current_line.append(box)
            else:
                # Box is on a new line
                # Check if this new line should be part of the current paragraph
                if current_paragraph:
                    last_line = current_paragraph[-1]
                    last_box = last_line[-1]
                    last_box_y = last_box['box'][1]
                    # If the new line is close enough to the last line, it's part of the same paragraph
                    if abs(box_y - last_box_y) <= y_threshold * 2:
                        current_line = [box]
                        current_paragraph.append(current_line)
                    else:
                        # Start a new paragraph
                        if current_paragraph:
                            paragraphs.append({
                                'text': ' '.join(' '.join(b['text'] for b in line) for line in current_paragraph),
                                'boxes': [box for line in current_paragraph for box in line],
                                'lines': current_paragraph
                            })
                        current_line = [box]
                        current_paragraph = [current_line]
                current_y = box_y

    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append({
            'text': ' '.join(' '.join(b['text'] for b in line) for line in current_paragraph),
            'boxes': [box for line in current_paragraph for box in line],
            'lines': current_paragraph
        })

    return paragraphs


def scroll_to_target_precise(device, width, height, target_abs_y, target_height):
    """Scroll precisely to position the target at the desired screen position using absolute coordinates."""
    # First scroll to top to ensure consistent starting point
    scroll_up_to_top(device, width, height)
    time.sleep(1)  # Wait for scroll to complete

    # Calculate desired screen position (30% from top)
    target_screen_y = int(height * 0.30)

    # Calculate total scroll needed from absolute position
    total_scroll_needed = target_abs_y - target_screen_y

    print(f"\nScrolling details:")
    print(f"Target absolute Y: {target_abs_y}")
    print(f"Target screen Y: {target_screen_y}")
    print(f"Total scroll needed: {total_scroll_needed}px")

    # Use very small scroll steps for precision (5% of screen height)
    scroll_step = int(height * 0.05)
    num_steps = abs(total_scroll_needed) // scroll_step
    remaining_scroll = abs(total_scroll_needed) % scroll_step

    # Determine scroll direction
    direction = 1 if total_scroll_needed > 0 else -1

    print(f"Scroll steps: {num_steps} of {scroll_step}px each")
    print(f"Remaining scroll: {remaining_scroll}px")

    # Perform the scroll steps
    for i in range(num_steps):
        scroll_precisely(device, width, height, direction * scroll_step)
        time.sleep(0.3)  # Shorter wait between steps

    # Handle remaining scroll
    if remaining_scroll > 2:  # Only scroll if remaining distance is significant
        scroll_precisely(device, width, height, direction * remaining_scroll)
        time.sleep(0.3)

    # Final small adjustment to center the target
    final_adjustment = int(target_height * 0.5)  # Center the target vertically
    scroll_precisely(device, width, height, -final_adjustment)
    time.sleep(0.5)


def run_bumble_automation(device, width, height):
    """Main automation function for processing 40 Bumble profiles"""
    print("Starting Bumble automation...")
    total_profiles = 40  # Changed from 20 to 40
    profiles_processed = 0
    comments_sent = 0
    dislikes_sent = 0
    superlikes_sent = 0
    likes_sent = 0

    # Process first two profiles with comments
    print("\n=== Processing First Two Profiles with Comments ===")
    for i in range(2):
        print(f"\nProcessing profile #{profiles_processed + 1} with comment")

        # Take initial screenshot and check for specific text
        initial_text = take_screenshot_and_extract_text(
            device, "part1", profiles_processed + 1, "top of profile")
        has_bond_text = "Things you can bond over" in initial_text if initial_text else False

        # Continue with profile scrolling and text capture
        all_profile_text = scroll_profile_and_capture(
            device, width, height, profiles_processed + 1, capture_screenshots=True)
        if initial_text:
            all_profile_text = initial_text + "\n" + all_profile_text

        print("\n=== Full Profile Text ===")
        print(all_profile_text)
        print("=== End Profile Text ===\n")

        # Generate and send comment
        if all_profile_text:
            comment = generate_comment(all_profile_text)
            if comment and comment.strip():
                # Scroll back to top before clicking heart
                print("Scrolling back to top...")
                scroll_up_to_top(device, width, height)
                time.sleep(0.5)

                # Choose heart button coordinates based on profile type
                if has_bond_text:
                    heart_x = 130  # Fixed x coordinate for bond profiles
                    heart_y = 1810  # Fixed y coordinate for bond profiles
                    print("Using bond profile heart coordinates")
                else:
                    heart_x = 125  # Fixed x coordinate for regular profiles
                    heart_y = 2075  # Fixed y coordinate for regular profiles
                    print("Using regular profile heart coordinates")

                print(f"Tapping heart button at: ({heart_x}, {heart_y})")
                tap(device, heart_x, heart_y)
                time.sleep(1.0)

                # Continue with comment flow
                tap(device, int(width * 0.074), int(height * 0.938))  # Text box
                time.sleep(0.5)
                command_safe = prepare_comment_for_input(comment)
                device.shell(f"input text {command_safe}")
                time.sleep(0.3)
                tap(device, int(width * 0.912),
                    int(height * 0.592))  # Send button
                time.sleep(1.0)
                comments_sent += 1
                profiles_processed += 1
                print(f"Comment sent for profile #{profiles_processed}")

    # Send one dislike
    print("\n=== Sending Dislike ===")
    scroll_profile_and_capture(
        device, width, height, profiles_processed + 1)
    send_dislike_hinge(device, width, height)
    dislikes_sent += 1
    profiles_processed += 1

    # Send 10 superlikes
    print("\n=== Sending 10 Superlikes ===")
    for i in range(10):
        scroll_profile_and_capture(
            device, width, height, profiles_processed + 1)
        send_bumble_superlike(device, width, height)
        superlikes_sent += 1
        profiles_processed += 1
        print(f"Superlike #{superlikes_sent}/10 sent")

    # Continue with like/dislike pattern
    print("\n=== Starting Regular Like/Dislike Pattern ===")
    while profiles_processed < total_profiles:
        # Send likes for 4-8 profiles instead of 7-12
        likes_in_group = random.randint(4, 8)
        for i in range(likes_in_group):
            if profiles_processed >= total_profiles:
                break
            scroll_profile_and_capture(
                device, width, height, profiles_processed + 1)
            # Regular like button coordinates
            like_x = int(width * 0.85)  # Right side like button
            like_y = int(height * 0.67)  # Same height as dislike
            print(f"Sending like #{likes_sent + 1}")
            tap(device, like_x, like_y)
            time.sleep(1.0)
            likes_sent += 1
            profiles_processed += 1

        # Send a dislike if we haven't reached total_profiles
        if profiles_processed < total_profiles:
            scroll_profile_and_capture(
                device, width, height, profiles_processed + 1)
            send_bumble_dislike(device, width, height)
            dislikes_sent += 1
            profiles_processed += 1

    # Print final statistics
    print("\n=== Automation Complete! ===")
    print(f"Total profiles processed: {profiles_processed}")
    print(f"Comments sent: {comments_sent}")
    print(f"Superlikes sent: {superlikes_sent}")
    print(f"Regular likes sent: {likes_sent}")
    print(f"Dislikes sent: {dislikes_sent}")


def run_hinge_automation(device, width, height):
    """Main automation function for processing 40 Hinge profiles"""
    print("Starting Hinge automation...")
    profiles_processed = 0
    total_profiles = 40  # Changed from 20 to 40

    while profiles_processed < total_profiles:
        # Send likes for 4-8 profiles
        likes_in_group = random.randint(4, 8)
        for i in range(likes_in_group):
            if profiles_processed >= total_profiles:
                break

            # Process profile and check if we should continue
            if not process_profile_hinge(device, width, height, profiles_processed + 1):
                print(
                    f"\nAutomation stopped after processing {profiles_processed} profiles")
                return

            profiles_processed += 1

        # Send a dislike if we haven't reached total_profiles
        if profiles_processed < total_profiles:
            send_dislike_hinge(device, width, height)
            profiles_processed += 1

    print(f"\nAutomation complete! Processed {profiles_processed} profiles")


def find_and_click_prompt(device, target_prompt_text, paragraphs, width, height):
    """Find and click on the target prompt text by first checking existing paragraphs, then searching with same scrolling pattern.

    Behavior:
    1. First check if the target prompt exists in the already collected paragraphs with threshold 0.4
    2. If found, scroll to top and use same scrolling pattern to find it again
    3. When found during second scroll, click the center of the first box in the paragraph
    """
    print(f"\nLooking for prompt: {target_prompt_text}")

    # First check all existing paragraphs for matches with threshold 0.4
    best_match = None
    best_ratio = 0.4  # Fixed threshold as specified

    for paragraph in paragraphs:
        ratio = difflib.SequenceMatcher(
            None, paragraph['text'].lower(), target_prompt_text.lower()).ratio()
        print(f"Existing paragraph similarity ratio: {ratio:.2f}")

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = paragraph
            print(f"New best match found with ratio {ratio:.2f}")

    # If we found a match with ratio >= 0.4, scroll to top and search with same pattern
    if best_match and best_ratio >= 0.4:
        print(
            f"\nFound match in existing paragraphs with ratio {best_ratio:.2f}, searching with same scrolling pattern")

        # Scroll to top
        scroll_up_to_top(device, width, height)
        time.sleep(1)

        # Use same scrolling pattern as text collection
        scroll_amount = int(height * 0.7)  # 70% of screen height
        scroll_duration = 800
        MAX_SCROLLS = 7

        for i in range(MAX_SCROLLS):
            print(f"\nSearch attempt {i + 1}/{MAX_SCROLLS}")

            # Take screenshot and get text boxes
            screenshot_path = capture_screenshot(
                device, f"prompt_search_{i}")
            boxes = extract_text_from_image_with_boxes(screenshot_path)

            if not boxes:
                print("No text boxes found in screenshot")
                continue

            # Group boxes into paragraphs using the same logic as initial scan
            current_paragraphs = group_boxes_into_paragraphs(boxes)

            print(
                f"\nFound {len(current_paragraphs)} paragraphs in this screenshot:")
            for p in current_paragraphs:
                print(f"  Paragraph: {p['text'][:60]}...")

            # Check each paragraph for matches
            for paragraph in current_paragraphs:
                ratio = difflib.SequenceMatcher(
                    None, paragraph['text'].lower(), target_prompt_text.lower()).ratio()
                print(f"Paragraph similarity ratio: {ratio:.2f}")

                if ratio >= 0.4:  # If we find a match ≥ 0.4, click it
                    print(f"\nFound match with ratio {ratio:.2f}:")
                    print(f"Text: {paragraph['text']}")

                    # Get the first box in the paragraph (top-left most)
                    first_box = paragraph['boxes'][0]
                    box = first_box['box']

                    # Calculate click position: bottom-left of the first box
                    x = box[0]  # Left edge of box
                    y = box[1] + box[3]  # Top + height = bottom edge

                    print(
                        f"First box coordinates: ({box[0]}, {box[1]}) to ({box[2]}, {box[3]})")
                    print(f"Double-tapping at coordinates: ({x}, {y})")

                    # Use input tap command with proper timing for double-tap
                    device.shell(f"input tap {x} {y}")
                    time.sleep(0.05)  # Very short delay between taps (50ms)
                    device.shell(f"input tap {x} {y}")
                    time.sleep(1)  # Wait for prompt to open
                    return True

            # Scroll down for next attempt using same pattern as text collection
            device.shell(
                f"input swipe {width//2} {int(height * 0.7)} {width//2} {int(height * 0.1)} {scroll_duration}")
            time.sleep(0.8)  # Same wait time as in text collection

    # If no matches found with ratio >= 0.4, click bottom right corner
    print("\nNo matches found with ratio >= 0.4, clicking bottom right corner")

    # Calculate bottom right corner coordinates (120px from right, 520px from bottom)
    corner_x = width - 120
    corner_y = height - 520
    print(
        f"Tapping at bottom right corner coordinates: ({corner_x}, {corner_y})")
    device.shell(f"input tap {corner_x} {corner_y}")
    time.sleep(1)  # Wait for prompt to open
    return True


def process_profile_hinge(device, width, height, profile_count):
    """Process Hinge profile: collect text, get AI comment+target, find target box, precisely scroll, double-tap."""
    print(f"\n--- Processing Hinge Profile #{profile_count} ---")

    # --- Collect Text & Boxes ---
    all_profile_text = ""
    all_boxes = []
    current_scroll_offset = 0
    scroll_amount_coarse = int(height * 0.7)
    MAX_SCROLLS = 7

    # Initial screenshot
    print("Taking initial screenshot...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_count}_part1")
    initial_boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not initial_boxes:
        print("Warning: No text boxes found in initial screenshot.")

    for item in initial_boxes:
        if "You've seen everyone for now" in item['text']:
            print("\nReached end of available profiles!")
            return False
        all_profile_text += item['text'] + "\n"
        all_boxes.append(item)
    print(
        f"Part 1: Found {len(initial_boxes)} boxes. Total text ~{len(all_profile_text)} chars.")

    # Scroll and capture subsequent parts
    for i in range(MAX_SCROLLS):
        print(f"Scrolling down (Coarse scroll {i+1}/{MAX_SCROLLS})...")
        scroll_once_hinge(device, width, height)
        time.sleep(0.8)  # Slightly increased sleep

        screenshot_path = capture_screenshot(
            device, f"profile_{profile_count}_part{i+2}")
        text_boxes = extract_text_from_image_with_boxes(screenshot_path)
        if not text_boxes:
            print(f"Warning: No text boxes found in screenshot part {i+2}.")
            continue  # Might just be a large image part

        part_text = ""
        end_detected_in_part = False
        num_boxes_part = 0
        for item in text_boxes:
            if "You've seen everyone for now" in item['text']:
                print("\nReached end of available profiles during scroll!")
                end_detected_in_part = True
                break
            part_text += item['text'] + "\n"
            all_boxes.append(item)
            num_boxes_part += 1

        all_profile_text += part_text
        print(
            f"Part {i+2}: Found {num_boxes_part} boxes. Total text ~{len(all_profile_text)} chars.")
        if end_detected_in_part:
            break

    # Group boxes into paragraphs
    paragraphs = group_boxes_into_paragraphs(all_boxes)

    print("\n--- Grouped Paragraphs ---")
    for paragraph in paragraphs:
        print(f"  Paragraph:")
        print(f"  {paragraph['text']}")
    print("--- End Grouped Paragraphs ---\n")

    # Combine paragraphs into text for AI
    ai_input_text = "\n\n".join(p['text'] for p in paragraphs)

    print("\n--- Text Sent to AI ---")
    print(ai_input_text)
    print("--- End Text Sent to AI ---\n")

    if not paragraphs:
        print("ERROR: No paragraphs found on profile after all scrolls. Sending dislike.")
        send_dislike_hinge(device, width, height)
        return True

    # --- Get Comment & Target Prompt from AI ---
    comment, target_prompt_text = generate_comment_with_target(ai_input_text)

    if not comment or not target_prompt_text:
        print("ERROR: AI did not generate a comment or identify a target prompt. Sending dislike.")
        send_dislike_hinge(device, width, height)
        return True

    print(f"AI generated comment: {comment}")
    print(f"AI target prompt text: '{target_prompt_text}'")

    # --- Find Target Paragraph using Sequence Matching ---
    found_matches = []
    target_lower = target_prompt_text.lower()
    MATCH_THRESHOLD = 0.4  # Changed to match the threshold used in find_and_click_prompt

    print(
        f"Searching for paragraphs with similarity >= {MATCH_THRESHOLD} to: '{target_lower[:60]}...'"
    )

    for paragraph in paragraphs:
        paragraph_text = paragraph['text']
        if len(paragraph_text) < 4:  # Minimum text length
            continue

        paragraph_text_lower = paragraph_text.lower()

        # Calculate similarity ratio
        matcher = difflib.SequenceMatcher(
            None, target_lower, paragraph_text_lower)
        ratio = matcher.ratio()

        # Log ratio for every comparison
        print(
            f"  Comparing with: '{paragraph_text[:60]}...' -> Ratio: {ratio:.2f}")

        if ratio >= MATCH_THRESHOLD:
            print(f"    ^ Potential match found!")
            found_matches.append({
                'box_info': paragraph,
                'ratio': ratio
            })

    if not found_matches:
        print(f"ERROR: Could not find the target prompt '{target_prompt_text[:60]}...' "
              f"anywhere on the profile after scanning (threshold: {MATCH_THRESHOLD}). "
              f"Cannot proceed with comment interaction. Falling back to center tap like.")
        send_like_center_tap(device, width, height)

        # Continue with comment workflow
        print("Continuing with comment workflow...")
        # Tap the "Add a comment" box
        x_comment_box = int(width * 0.5)
        y_comment_box = int(height * 0.78)
        print(f"Tapping comment box at: ({x_comment_box}, {y_comment_box})")
        tap(device, x_comment_box, y_comment_box)
        time.sleep(2.5)  # Wait for keyboard to appear

        # Clear any existing text
        device.shell("input keyevent KEYCODE_CTRL_A")
        time.sleep(0.8)  # Wait for selection
        device.shell("input keyevent KEYCODE_DEL")
        time.sleep(0.8)  # Wait for deletion

        print("Typing comment character by character...")
        # Type the comment character by character
        for char in comment:
            # Use input text for each character
            device.shell(f"input text '{char}'")
            time.sleep(0.1)  # Small delay between characters

        time.sleep(1.5)  # Wait for text to be fully entered

        # Close the keyboard
        device.shell("input keyevent 4")  # KEYCODE_BACK
        time.sleep(1.5)  # Wait for keyboard to close

        # Tap the "Send Like" button
        send_button_x = int(width * 0.5)
        send_button_y = int(height * 0.85)
        print(f"Tapping Send button at: ({send_button_x}, {send_button_y})")
        tap(device, send_button_x, send_button_y)
        time.sleep(5)  # Wait for the like to be sent

        return True  # Move to next profile

    # --- Find and Click Target Prompt ---
    print(
        f"\nAttempting to find and click target prompt: '{target_prompt_text[:60]}...'")
    if not find_and_click_prompt(device, target_prompt_text, paragraphs, width, height):
        print("Failed to find and click prompt, falling back to center tap like")
        send_like_center_tap(device, width, height)

        # Continue with comment workflow
        print("Continuing with comment workflow...")
        # Tap the "Add a comment" box
        x_comment_box = int(width * 0.5)
        y_comment_box = int(height * 0.78)
        print(f"Tapping comment box at: ({x_comment_box}, {y_comment_box})")
        tap(device, x_comment_box, y_comment_box)
        time.sleep(2.5)  # Wait for keyboard to appear

        # Clear any existing text
        device.shell("input keyevent KEYCODE_CTRL_A")
        time.sleep(0.8)  # Wait for selection
        device.shell("input keyevent KEYCODE_DEL")
        time.sleep(0.8)  # Wait for deletion

        print("Typing comment character by character...")
        # Type the comment character by character
        for char in comment:
            # Use input text for each character
            device.shell(f"input text '{char}'")
            time.sleep(0.1)  # Small delay between characters

        time.sleep(1.5)  # Wait for text to be fully entered

        # Close the keyboard
        device.shell("input keyevent 4")  # KEYCODE_BACK
        time.sleep(1.5)  # Wait for keyboard to close

        # Tap the "Send Like" button
        send_button_x = int(width * 0.5)
        send_button_y = int(height * 0.85)
        print(f"Tapping Send button at: ({send_button_x}, {send_button_y})")
        tap(device, send_button_x, send_button_y)
        time.sleep(5)  # Wait for the like to be sent

        return True  # Move to next profile

    time.sleep(1.5)  # Wait for prompt interaction

    # --- Commenting Workflow ---
    # Tap the "Add a comment" box
    x_comment_box = int(width * 0.5)
    y_comment_box = int(height * 0.78)
    print(f"Tapping comment box at: ({x_comment_box}, {y_comment_box})")
    tap(device, x_comment_box, y_comment_box)
    time.sleep(2.5)  # Wait for keyboard to appear

    # Clear any existing text
    device.shell("input keyevent KEYCODE_CTRL_A")
    time.sleep(0.8)  # Wait for selection
    device.shell("input keyevent KEYCODE_DEL")
    time.sleep(0.8)  # Wait for deletion

    print("Typing comment character by character...")
    # Type the comment character by character
    for char in comment:
        # Use input text for each character
        device.shell(f"input text '{char}'")
        time.sleep(0.1)  # Small delay between characters

    time.sleep(1.5)  # Wait for text to be fully entered

    # Close the keyboard
    device.shell("input keyevent 4")  # KEYCODE_BACK
    time.sleep(1.5)  # Wait for keyboard to close

    # Tap the "Send Like" button
    send_button_x = int(width * 0.5)
    send_button_y = int(height * 0.85)
    print(f"Tapping Send button at: ({send_button_x}, {send_button_y})")
    tap(device, send_button_x, send_button_y)
    time.sleep(5)  # Wait for the like to be sent

    print(f"Profile #{profile_count} processed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', choices=['hinge', 'bumble'], required=True,
                        help='Which dating app to automate')
    args = parser.parse_args()

    device_ip = os.getenv("DEVICE_IP", "192.168.12.32")
    print(f"Connecting to device at IP: {device_ip}")

    device = connect_device_remote(device_ip)
    if not device:
        print("Failed to connect to device")
        sys.exit(1)

    width, height = get_screen_resolution(device)
    print(f"Screen resolution: {width}x{height}")

    if args.app == 'bumble':
        run_bumble_automation(device, width, height)
    elif args.app == 'hinge':
        run_hinge_automation(device, width, height)
    else:
        print("Please use --app bumble or hinge for this automation")
        sys.exit(1)


if __name__ == "__main__":
    main()
