# app/main.py

import time
import os
import sys
import random
import threading
from dotenv import load_dotenv
from ppadb.client import Client as AdbClient
import difflib
import logging

from helper_functions import (
    connect_device_remote,
    get_screen_resolution,
    capture_screenshot,
    extract_text_from_image_with_boxes,
    generate_hinge_reply_from_screenshots,
    tap,
    input_text,
    create_visual_debug_overlay,
    group_boxes_into_lines,
    group_lines_into_paragraphs,
    fuzzy_match_text,
    swipe,
    detect_prompt_in_screenshot,
    send_response_to_story,
    dislike_profile,
    save_profile_results,
    check_for_end_of_profiles,
    logger,
)

# Global variable to store AI response
ai_response = None
ai_response_lock = threading.Lock()


def scroll_profile_and_capture(device, width, height, profile_num):
    """Scroll through profile and capture screenshots."""
    try:
        screenshots = []

        # Take initial screenshot
        logger.info("Capturing initial screenshot")
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part1")
        screenshots.append(screenshot_path)

        # Take 6 more screenshots (7 total)
        for i in range(1, 7):
            logger.info(f"Scroll down #{i}")
            # Scroll down
            swipe(device, "down")

            # Take screenshot after scroll
            screenshot_path = capture_screenshot(
                device, f"profile_{profile_num}_part{i+1}")
            screenshots.append(screenshot_path)

        logger.info(
            f"Captured {len(screenshots)} screenshots for profile #{profile_num}")
        return screenshots
    except Exception as e:
        logger.error(f"Error in scroll_profile_and_capture: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return []


def scroll_back_to_top(device):
    """Scroll back to the top of the profile."""
    try:
        for i in range(1, 7):
            logger.info(f"Scroll up #{i}")
            # Scroll up
            swipe(device, "up")
        return True
    except Exception as e:
        logger.error(f"Error in scroll_back_to_top: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return False


def process_ai_response(screenshots, format_txt_path, prompts_txt_path, captions_txt_path, polls_txt_path, locations_txt_path):
    """
    Process screenshots and generate a conversation starter using Claude API.

    Args:
        screenshots: List of paths to screenshot images
        format_txt_path: Path to hingeFormat.txt
        prompts_txt_path: Path to hingePrompts.txt
        captions_txt_path: Path to hingeCaptions.txt
        polls_txt_path: Path to hingePolls.txt
        locations_txt_path: Path to locations.txt

    Returns:
        dict: AI response containing prompt, response, conversation starter, and screenshot index
    """
    global ai_response
    try:
        result = generate_hinge_reply_from_screenshots(
            screenshots,
            format_txt_path,
            prompts_txt_path,
            captions_txt_path,
            polls_txt_path,
            locations_txt_path
        )
        with ai_response_lock:
            ai_response = result
    except SystemExit:
        # Re-raise SystemExit to propagate to main thread
        logger.critical(
            "API error occurred in background thread - propagating exit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in AI processing thread: {e}")
        logger.debug("", exc_info=True)
        with ai_response_lock:
            ai_response = None
        # Also exit the application on unhandled exceptions
        sys.exit(1)


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
        logger.warning("No text boxes found in screenshot")
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
    logger.debug(f"Created visualization: {vis_path}")

    return paragraphs, vis_path


def scroll_to_screenshot(device, screenshot_index):
    """Scroll to the specified screenshot index using consistent 68% swipe distance."""
    # We're already at the top from scroll_profile_and_capture
    # Just scroll down to the target screenshot
    logger.info(f"\nScrolling to screenshot index {screenshot_index}...")
    for i in range(screenshot_index):
        logger.info(f"Scroll down #{i+1}")
        swipe(device, "down")
        time.sleep(1)  # Wait for scroll to complete

    return True


def match_prompt_against_authoritative(prompt, prompts_txt_path):
    """
    Match a prompt from AI response against the authoritative list of prompts.

    Args:
        prompt: The prompt string from AI
        prompts_txt_path: Path to hingePrompts.txt containing authoritative prompts

    Returns:
        tuple: (matched_prompt, confidence)
    """
    try:
        with open(prompts_txt_path, 'r') as f:
            authoritative_prompts = f.read().splitlines()
    except Exception as e:
        logger.error(f"Error reading prompts.txt: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return None, 0.0

    best_match = None
    best_ratio = 0.0

    # Print all prompts that have a decent match for debugging
    logger.info("Checking prompt matches:")
    for auth_prompt in authoritative_prompts:
        is_match, ratio, matched_text = fuzzy_match_text(
            prompt, auth_prompt, threshold=0.5)  # Lower threshold for debugging
        if ratio > 0.5:  # Show any decent matches
            logger.info(f"  {auth_prompt}: {ratio:.2f}")
        if is_match and ratio > best_ratio:
            best_match = auth_prompt
            best_ratio = ratio

    if best_match:
        logger.info("\nBest match:")
        logger.info(f"AI prompt: '{prompt}'")
        logger.info(f"Auth prompt: '{best_match}'")
        logger.info(f"Confidence: {best_ratio:.2f}")
        return best_match, best_ratio
    else:
        logger.warning(f"\nNo good matches found for '{prompt}'")
        return None, 0.0


def main():
    """Main function to run the Hinge automation."""
    try:
        # Get device IP from environment variable
        device_ip = os.getenv("DEVICE_IP", "192.168.12.32")
        logger.info(f"Connecting to device at IP: {device_ip}")

        # Connect to device
        device = connect_device_remote(device_ip)
        if not device:
            logger.error("Failed to connect to device")
            sys.exit(1)

        # Get screen dimensions
        width, height = get_screen_resolution(device)
        logger.info(f"Screen resolution: {width}x{height}")

        # Initialize profile counter, successful interaction counter, and target interactions
        profile_num = 1
        successful_interactions = 0
        target_interactions = random.randint(4, 6)
        max_profiles = 40  # Maximum number of profiles to process
        logger.info(f"Initial target interactions: {target_interactions}")
        logger.info(f"Maximum profiles to process: {max_profiles}")

        # Initialize file paths
        # Get the absolute path to the app directory
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Default paths for text files
        format_txt_path = os.path.join(app_dir, 'hingeFormat.txt')
        prompts_txt_path = os.path.join(app_dir, 'hingePrompts.txt')
        captions_txt_path = os.path.join(app_dir, 'hingeCaptions.txt')
        polls_txt_path = os.path.join(app_dir, 'hingePolls.txt')
        locations_txt_path = os.path.join(app_dir, 'locations.txt')

        while profile_num <= max_profiles:  # Main profile loop with limit
            logger.info(f"\nProcessing profile #{profile_num}")

            # Check if we've reached the end of available profiles
            end_reached, end_message = check_for_end_of_profiles(
                device, profile_num)
            if end_reached:
                logger.info(
                    f"Reached end of available profiles: '{end_message}'. Exiting...")
                break

            # Check if we've reached our target interactions
            if successful_interactions >= target_interactions:
                logger.info(
                    f"Reached target of {target_interactions} successful interactions")
                dislike_profile(device)
                successful_interactions = 0  # Reset counter
                target_interactions = random.randint(
                    4, 6)  # Generate new target
                logger.info(f"New target interactions: {target_interactions}")
                profile_num += 1
                continue

            # Reset AI response
            global ai_response
            ai_response = None

            # Scroll through profile and capture screenshots
            screenshots = scroll_profile_and_capture(
                device, width, height, profile_num)

            # Start AI processing in a separate thread
            logger.info("Starting AI processing in separate thread")
            ai_thread = threading.Thread(
                target=process_ai_response,
                args=(screenshots, format_txt_path, prompts_txt_path,
                      captions_txt_path, polls_txt_path, locations_txt_path)
            )
            ai_thread.start()

            # Scroll back to top while AI is processing
            logger.info("Scrolling back to top while AI processes")
            scroll_back_to_top(device)

            # Wait for AI response
            logger.info("Waiting for AI processing to complete")
            ai_thread.join()
            logger.info("AI processing complete")

            # Save profile results regardless of outcome
            results_dir = save_profile_results(
                profile_num, screenshots, ai_response)
            logger.info(f"Saved profile results to: {results_dir}")

            # Check if profile is undesirable (empty response)
            if not ai_response or not ai_response.get('prompt') or not ai_response.get('response') or not ai_response.get('conversation_starter') or ai_response.get('screenshot_index') == -1:
                logger.info("Profile marked as undesirable - disliking")
                dislike_profile(device)
                successful_interactions = 0  # Reset counter when disliking for any reason
                target_interactions = random.randint(
                    4, 6)  # Generate new target
                profile_num += 1
                continue

            # Match prompt against authoritative list
            matched_prompt, confidence = match_prompt_against_authoritative(
                ai_response['prompt'], prompts_txt_path)

            if not matched_prompt or confidence < 0.8:
                logger.warning(
                    f"Low confidence prompt match ({confidence:.2f})")
                logger.warning(f"Original: {ai_response['prompt']}")
                logger.warning(f"Matched: {matched_prompt}")
                dislike_profile(device)
                successful_interactions = 0  # Reset counter when disliking for any reason
                target_interactions = random.randint(
                    4, 6)  # Generate new target
                profile_num += 1
                continue

            # Scroll to the screenshot containing the prompt
            scroll_to_screenshot(device, ai_response['screenshot_index'])

            # Try to find and tap the prompt
            found, tap_coords = detect_prompt_in_screenshot(
                device,
                matched_prompt,
                ai_response['response'],
                ai_response['screenshot_index'],
                profile_num
            )

            if not found:
                logger.warning(
                    "Failed to find prompt on target screenshot - trying one screen up...")
                # Scroll up one screen and try again
                swipe(device, "up")
                time.sleep(1)  # Wait for scroll to complete

                found, tap_coords = detect_prompt_in_screenshot(
                    device,
                    matched_prompt,
                    ai_response['response'],
                    ai_response['screenshot_index'],
                    profile_num
                )

                if not found:
                    logger.warning(
                        "Failed to find prompt on screen above - returning to target screenshot...")
                    # Scroll back down to target screenshot
                    swipe(device, "down")
                    time.sleep(1)  # Wait for scroll to complete

                    # Try one more time on target screenshot
                    found, tap_coords = detect_prompt_in_screenshot(
                        device,
                        matched_prompt,
                        ai_response['response'],
                        ai_response['screenshot_index'],
                        profile_num
                    )

                    if not found:
                        logger.error(
                            "Failed to find prompt after all attempts")
                        dislike_profile(device)
                        successful_interactions = 0  # Reset counter when disliking for any reason
                        target_interactions = random.randint(
                            4, 6)  # Generate new target
                        profile_num += 1
                        continue

            # Send the response
            success = send_response_to_story(
                device, ai_response['conversation_starter'], profile_num)

            if not success:
                logger.error("Failed to send response")
                dislike_profile(device)
                successful_interactions = 0  # Reset counter when disliking for any reason
                target_interactions = random.randint(
                    4, 6)  # Generate new target
            else:
                # Increment successful interactions counter
                successful_interactions += 1
                logger.info(
                    f"Successful interactions: {successful_interactions}/{target_interactions}")

            profile_num += 1

        logger.info(
            f"\nReached maximum profile limit of {max_profiles}. Exiting...")

    except KeyboardInterrupt:
        logger.info("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        raise


if __name__ == "__main__":
    main()
