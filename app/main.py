# app/main.py

import time
import os
import sys
import random
from dotenv import load_dotenv
import re
import argparse
from pytesseract import image_to_data
import math
import difflib
from ppadb.client import Client as AdbClient
import json

# Import your existing helper functions
from helper_functions import (
    connect_device_remote,
    get_screen_resolution,
    capture_screenshot,
    extract_text_from_image,
    extract_text_from_image_with_boxes,
    generate_joke_from_screenshots,
    tap,
    create_visual_debug_overlay,
    group_boxes_into_lines,
    group_lines_into_paragraphs,
    type_text_slow,
)


def take_screenshot_and_extract_text(device, filename, profile_count, section_name):
    """Take a screenshot and extract text from it"""
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_count}_{filename}")

    # Extract text and boxes
    boxes = extract_text_from_image_with_boxes(screenshot_path)

    # Group boxes into lines and paragraphs
    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Create visualization
    create_visual_debug_overlay(
        screenshot_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/profile_{profile_count}_{filename}_visual.png"
    )

    # Extract plain text for processing
    text = " ".join(box['text'] for box in boxes).strip()
    if text:
        print(f"Extracted text from {section_name} ({len(text)} chars)")
    return text, boxes, lines, paragraphs


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


def group_boxes_into_paragraphs(boxes, y_threshold=10, paragraph_spacing=30):
    """Group text boxes into a hierarchical structure: words → lines → paragraphs → pairs.

    Args:
        boxes: List of text boxes with 'text' and 'box' fields
        y_threshold: Maximum vertical distance for boxes to be on same line
        paragraph_spacing: Minimum vertical distance between paragraphs

    Returns:
        List of paragraphs, where each paragraph contains:
        - text: Combined text of all boxes
        - boxes: List of all boxes in the paragraph
        - lines: List of lines in the paragraph
        - spatial_info: Metadata about the paragraph's position
    """
    if not boxes:
        return []

    # 1. Box Processing (Words)
    # Sort boxes spatially: primary by vertical position, secondary by horizontal
    sorted_boxes = sorted(boxes, key=lambda b: (b['box'][1], b['box'][0]))

    # 2. Line Detection
    lines = []
    current_line = []
    current_y = None

    for box in sorted_boxes:
        box_y = box['box'][1]

        if current_y is None:
            # Start first line
            current_y = box_y
            current_line = [box]
        else:
            # Check if box belongs to current line
            if abs(box_y - current_y) <= y_threshold:
                current_line.append(box)
            else:
                # Start new line
                if current_line:
                    lines.append(current_line)
                current_line = [box]
                current_y = box_y

    # Add last line if exists
    if current_line:
        lines.append(current_line)

    # 3. Paragraph Formation
    paragraphs = []
    current_paragraph = []
    last_line_bottom = None

    for line in lines:
        # Get bottom y-coordinate of the line
        line_bottom = max(box['box'][1] + box['box'][3] for box in line)

        if last_line_bottom is None:
            # Start first paragraph
            last_line_bottom = line_bottom
            current_paragraph = [line]
        else:
            # Check if line belongs to current paragraph based on vertical spacing
            if line_bottom - last_line_bottom <= paragraph_spacing:
                current_paragraph.append(line)
                last_line_bottom = line_bottom
            else:
                # Start new paragraph
                if current_paragraph:
                    paragraphs.append(
                        create_paragraph_structure(current_paragraph))
                current_paragraph = [line]
                last_line_bottom = line_bottom

    # Add last paragraph if exists
    if current_paragraph:
        paragraphs.append(create_paragraph_structure(current_paragraph))

    return paragraphs


def create_paragraph_structure(lines):
    """Create a structured paragraph object from a list of lines.

    Args:
        lines: List of lines, where each line is a list of boxes

    Returns:
        Dictionary containing paragraph structure and metadata
    """
    # Flatten boxes from all lines
    all_boxes = [box for line in lines for box in line]

    # Calculate paragraph boundaries
    min_x = min(box['box'][0] for box in all_boxes)
    max_x = max(box['box'][0] + box['box'][2] for box in all_boxes)
    min_y = min(box['box'][1] for box in all_boxes)
    max_y = max(box['box'][1] + box['box'][3] for box in all_boxes)

    # Combine text from all boxes
    paragraph_text = ' '.join(
        ' '.join(box['text'] for box in line) for line in lines)

    return {
        'text': paragraph_text,
        'boxes': all_boxes,
        'lines': lines,
        'spatial_info': {
            'bounds': (min_x, min_y, max_x - min_x, max_y - min_y),
            'center': ((min_x + max_x) // 2, (min_y + max_y) // 2)
        }
    }


def identify_prompt_response_pairs(paragraphs, prompt_paragraphs, response_paragraphs, max_distance=50):
    """Identify pairs of prompts and responses based on spatial relationships.

    Args:
        paragraphs: List of paragraphs with spatial_info
        prompt_paragraphs: List of paragraph indices that are prompts
        response_paragraphs: List of paragraph indices that are responses
        max_distance: Maximum vertical distance between prompt and response

    Returns:
        List of (prompt_idx, response_idx) tuples for valid pairs
    """
    pairs = []

    # Sort prompts and responses by vertical position
    sorted_prompts = sorted(prompt_paragraphs,
                            key=lambda idx: paragraphs[idx]['spatial_info']['center'][1])
    sorted_responses = sorted(response_paragraphs,
                              key=lambda idx: paragraphs[idx]['spatial_info']['center'][1])

    # Track used responses to prevent double-pairing
    used_responses = set()

    for prompt_idx in sorted_prompts:
        prompt = paragraphs[prompt_idx]
        prompt_bottom = prompt['spatial_info']['bounds'][1] + \
            prompt['spatial_info']['bounds'][3]

        # Find closest unused response below
        closest_response = None
        min_distance = float('inf')

        for response_idx in sorted_responses:
            if response_idx in used_responses:
                continue

            response = paragraphs[response_idx]
            response_top = response['spatial_info']['bounds'][1]

            # Calculate vertical distance
            distance = response_top - prompt_bottom

            # Check if response is below prompt and within max distance
            if 0 <= distance <= max_distance and distance < min_distance:
                # Additional spatial checks
                prompt_center_x = prompt['spatial_info']['center'][0]
                response_center_x = response['spatial_info']['center'][0]

                # Check horizontal alignment (within 20% of screen width)
                if abs(prompt_center_x - response_center_x) <= max_distance * 0.2:
                    closest_response = response_idx
                    min_distance = distance

        if closest_response is not None:
            pairs.append((prompt_idx, closest_response))
            used_responses.add(closest_response)

    return pairs


def process_screenshot_with_vsualization(image_path):
    """Process a screenshot aind create visualization.

    Args:
        image_path: Path to the screenshot image

    Returns:
        Tuple of (paragraphs, prompt_paragraphs, response_paragraphs, pairs, json_output)
    """
    # Extract text boxes
    boxes = extract_text_from_image_with_boxes(image_path)

    # Group into paragraphs
    paragraphs = group_boxes_into_paragraphs(boxes)

    # TODO: Implement prompt/response identification
    # For now, use dummy data
    prompt_paragraphs = []  # Will be filled with actual prompt indices
    response_paragraphs = []  # Will be filled with actual response indices

    # Identify pairs
    pairs = identify_prompt_response_pairs(
        paragraphs, prompt_paragraphs, response_paragraphs)

    # Create JSON output
    json_output = create_json_output(
        paragraphs, prompt_paragraphs, response_paragraphs, pairs)

    # Create visualization
    vis_path = create_visual_debug_overlay(
        image_path,
        boxes,
        [line for para in paragraphs for line in para['lines']],
        paragraphs,
        prompt_paragraphs,
        response_paragraphs,
        pairs
    )

    print(f"Created visualization: {vis_path}")

    return paragraphs, prompt_paragraphs, response_paragraphs, pairs, json_output


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
            comment = generate_joke_from_json(all_profile_text)
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

    # Process paragraphs into JSON structure
    prompt_paragraphs = []  # TODO: Implement prompt identification
    response_paragraphs = []  # TODO: Implement response identification
    pairs = identify_prompt_response_pairs(
        paragraphs, prompt_paragraphs, response_paragraphs)
    json_output = create_json_output(
        paragraphs, prompt_paragraphs, response_paragraphs, pairs)

    print("\n--- JSON Structure ---")
    print(json.dumps(json_output, indent=2))
    print("--- End JSON Structure ---\n")

    if not paragraphs:
        print("ERROR: No paragraphs found on profile after all scrolls. Sending dislike.")
        send_dislike_hinge(device, width, height)
        return True

    # --- Get Comment & Target Prompt from AI ---
    result = generate_joke_from_json(json_output)

    if not result or not isinstance(result, dict):
        print("ERROR: AI did not generate a valid response. Sending dislike.")
        send_dislike_hinge(device, width, height)
        return True

    target_prompt = result.get('target_prompt', {})
    target_prompt_text = target_prompt.get('text', '')
    comment = result.get('joke_response', '')

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
        return True

    # --- Find and Click Target Prompt ---
    print(
        f"\nAttempting to find and click target prompt: '{target_prompt_text[:60]}...'")
    if not find_and_click_prompt(device, target_prompt_text, paragraphs, width, height):
        print("Failed to find and click prompt, falling back to center tap like")
        send_like_center_tap(device, width, height)
        return True

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


def create_json_output(paragraphs, prompt_paragraphs, response_paragraphs, pairs):
    """Create JSON output structure for prompts and responses.

    Args:
        paragraphs: List of paragraphs with text and spatial_info
        prompt_paragraphs: List of paragraph indices that are prompts
        response_paragraphs: List of paragraph indices that are responses
        pairs: List of (prompt_idx, response_idx) tuples for valid pairs

    Returns:
        List of objects with prompt and response keys, where:
        - Both keys have values for pairs
        - Only prompt has value for standalone prompts
        - Only response has value for standalone responses
    """
    # Track used paragraphs to prevent double-counting
    used_paragraphs = set()
    json_output = []

    # First, add all pairs
    for prompt_idx, response_idx in pairs:
        prompt_text = paragraphs[prompt_idx]['text']
        response_text = paragraphs[response_idx]['text']
        json_output.append({
            "prompt": prompt_text,
            "response": response_text
        })
        used_paragraphs.add(prompt_idx)
        used_paragraphs.add(response_idx)

    # Then add standalone prompts
    for prompt_idx in prompt_paragraphs:
        if prompt_idx not in used_paragraphs:
            prompt_text = paragraphs[prompt_idx]['text']
            json_output.append({
                "prompt": prompt_text,
                "response": None
            })
            used_paragraphs.add(prompt_idx)

    # Finally add standalone responses
    for response_idx in response_paragraphs:
        if response_idx not in used_paragraphs:
            response_text = paragraphs[response_idx]['text']
            json_output.append({
                "prompt": None,
                "response": response_text
            })
            used_paragraphs.add(response_idx)

    # Sort by vertical position
    json_output.sort(key=lambda x:
                     paragraphs[prompt_paragraphs[0]]['spatial_info']['center'][1] if x['prompt'] else
                     paragraphs[response_paragraphs[0]
                                ]['spatial_info']['center'][1] if x['response'] else 0
                     )

    return json_output


def generate_joke_from_json(json_output):
    """
    Generate a joke from the JSON output of a profile.

    Args:
        json_output (dict): The JSON output containing profile information

    Returns:
        dict: A dictionary containing:
            - target_prompt (dict): The matched prompt with text and type
            - joke_response (str): The generated joke/comment
            - confidence (float): Confidence score for prompt matching
    """
    try:
        screenshots = []
        for paragraph in json_output.get('paragraphs', []):
            screenshots.append({
                'text': paragraph.get('text', ''),
                'type': paragraph.get('type', '')
            })

        if not screenshots:
            print("No screenshots found in JSON output")
            return None

        result = generate_joke_from_screenshots(
            screenshots=screenshots,
            format_txt_path='app/data/format.txt',
            prompts_txt_path='app/data/prompts.txt',
            captions_txt_path='app/data/captions.txt'
        )

        return result

    except Exception as e:
        print(f"Error generating joke from JSON: {str(e)}")
        return None


def scroll_and_search_prompt(device, width, height, target_prompt, target_response, prompts_txt_path, max_scrolls=8):
    """Scroll through profile and search for target prompt or response.

    Args:
        device: ADB device
        width: Screen width
        height: Screen height
        target_prompt: The prompt text to search for
        target_response: The response text to search for
        prompts_txt_path: Path to prompts.txt
        max_scrolls: Maximum number of scrolls before fallback

    Returns:
        tuple: (found, matched_element, screenshot_path)
            where matched_element is either the prompt or response that was found
    """
    # Initialize variables
    scroll_count = 0
    x_scroll = int(width * 0.5)  # Center of screen
    y_scroll_start = int(height * 0.75)
    y_scroll_end = int(height * 0.08)

    while scroll_count < max_scrolls:
        # Take screenshot and extract text
        screenshot_path = capture_screenshot(
            device, f"search_scroll_{scroll_count}")
        boxes = extract_text_from_image_with_boxes(screenshot_path)

        if not boxes:
            print(f"No text found in scroll {scroll_count}")
            scroll_count += 1
            continue

        # Group boxes into lines and paragraphs
        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Create visualization for debugging
        vis_path = create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/search_scroll_{scroll_count}_visual.png"
        )

        # Check each paragraph for matches with either prompt or response
        for para in paragraphs:
            para_text = para['text']

            # Try matching with prompt
            is_match, ratio, matched = fuzzy_match_text(
                target_prompt, para_text)
            if is_match:
                print(f"Found prompt match with confidence {ratio:.2f}")
                return True, para, screenshot_path

            # Try matching with response
            is_match, ratio, matched = fuzzy_match_text(
                target_response, para_text)
            if is_match:
                print(f"Found response match with confidence {ratio:.2f}")
                return True, para, screenshot_path

        # Scroll down for next attempt
        device.shell(
            f"input swipe {x_scroll} {y_scroll_start} {x_scroll} {y_scroll_end} 1500")
        time.sleep(0.5)
        scroll_count += 1

    # If we get here, we didn't find either the prompt or response
    print("Reached maximum scrolls without finding prompt or response")
    return False, None, None


def process_profile_with_prompt_search(device, width, height, profile_count, prompts_txt_path):
    """Process a profile with prompt search and response.

    Args:
        device: ADB device
        width: Screen width
        height: Screen height
        profile_count: Current profile number
        prompts_txt_path: Path to prompts.txt
    """
    print(f"\nProcessing profile {profile_count}")

    # Take initial screenshot
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_count}_initial")
    boxes = extract_text_from_image_with_boxes(screenshot_path)

    if not boxes:
        print("No text found in initial screenshot")
        return

    # Group boxes into lines and paragraphs
    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Create visualization
    vis_path = create_visual_debug_overlay(
        screenshot_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/profile_{profile_count}_initial_visual.png"
    )

    # Generate joke and get target prompt/response
    joke_result = generate_joke_from_screenshots(
        [screenshot_path],
        "app/data/format.txt",
        prompts_txt_path,
        "app/data/captions.txt"
    )

    if not joke_result:
        print("Could not generate joke")
        return

    target_prompt = joke_result.get('prompt', '')
    target_response = joke_result.get('response', '')

    if not target_prompt and not target_response:
        print("No target prompt or response found")
        return

    # Search for either prompt or response
    found, matched_element, search_screenshot = scroll_and_search_prompt(
        device, width, height, target_prompt, target_response, prompts_txt_path
    )

    if found:
        # Get the center coordinates of the matched element's box
        box = matched_element['boxes'][0]  # Get first box of the paragraph
        x = box['box'][0] + (box['box'][2] // 2)  # Center of box
        y = box['box'][1] + (box['box'][3] // 2)

        print(f"Found match at coordinates ({x}, {y})")

        # Double-tap the center of the matched element
        tap(device, x, y)
        time.sleep(0.1)
        tap(device, x, y)
        time.sleep(1)

        # Input generated joke as photo comment
        joke = joke_result.get('joke', '')
        if joke:
            print("Typing joke character by character...")
            type_text_slow(device, joke, per_char_delay=0.1)
            time.sleep(1)

            # Tap send button (adjust coordinates as needed)
            send_x = int(width * 0.9)
            send_y = int(height * 0.9)
            tap(device, send_x, send_y)
            time.sleep(2)

            print(f"Successfully processed profile {profile_count}")
            return
    else:
        print("No match found, falling back to center image tap")
        # Double-tap center of screen (last photo)
        center_x = width // 2
        center_y = height // 2
        tap(device, center_x, center_y)
        time.sleep(0.1)
        tap(device, center_x, center_y)
        time.sleep(1)

        # Input generated joke as photo comment
        joke = joke_result.get('joke', '')
        if joke:
            print("Typing joke character by character...")
            type_text_slow(device, joke, per_char_delay=0.1)
            time.sleep(1)

            # Tap send button (adjust coordinates as needed)
            send_x = int(width * 0.9)
            send_y = int(height * 0.9)
            tap(device, send_x, send_y)
            time.sleep(2)

            print(
                f"Successfully processed profile {profile_count} with fallback")
            return

    print(f"Could not process profile {profile_count}")


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
