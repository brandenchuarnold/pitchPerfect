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
    generate_bumble_reply_from_screenshots,
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
    open_hinge,
    close_hinge,
    open_bumble,
    close_bumble,
    setup_logging,
    check_for_bumble_advertisement,
)

# Global variable to store AI response
ai_response = None
ai_response_lock = threading.Lock()


def scroll_profile_and_capture(device, width, height, profile_num, num_screenshots=7):
    """Scroll through profile and capture screenshots.

    Args:
        device: The ADB device
        width: Screen width
        height: Screen height
        profile_num: Current profile number
        num_screenshots: Number of screenshots to capture (7 for Hinge, 9 for Bumble)

    Returns:
        list: Paths to captured screenshots
    """
    try:
        screenshots = []

        # Take initial screenshot
        logger.info("Capturing initial screenshot")
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_part1")
        screenshots.append(screenshot_path)

        # Take remaining screenshots
        for i in range(1, num_screenshots):
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


def scroll_back_to_top(device, num_scrolls=6):
    """Scroll back to the top of the profile.

    Args:
        device: The ADB device
        num_scrolls: Number of scrolls to perform (6 for Hinge with 7 screenshots, 8 for Bumble with 9 screenshots)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        for i in range(1, num_scrolls+1):
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


def process_hinge_profile(device, width, height, profile_num, target_likes_before_dislike, disliked_profiles, total_likes, total_likes_target, format_txt_path, prompts_txt_path, captions_txt_path, polls_txt_path, locations_txt_path):
    """Process a Hinge profile with scrolling, capturing screenshots, and AI evaluation."""
    # Reset AI response
    global ai_response
    ai_response = None

    # Scroll through profile and capture screenshots
    screenshots = scroll_profile_and_capture(
        device, width, height, profile_num)

    # Check if we need to force dislike based on the counter logic
    # If we've reached our target likes and haven't disliked any, dislike this one
    force_dislike = False
    if total_likes >= target_likes_before_dislike and disliked_profiles == 0:
        force_dislike = True
        logger.info(
            f"Reached {total_likes} likes without any dislikes. Forcing dislike.")

    if force_dislike:
        # Save profile results without AI evaluation
        results_dir = save_profile_results(profile_num, screenshots, None)
        logger.info(f"Saved profile results to: {results_dir}")

        logger.info("Disliking profile based on counter logic")
        dislike_profile(device)
        disliked_profiles += 1
        return disliked_profiles, total_likes, False

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

    # Save profile results with AI response
    results_dir = save_profile_results(profile_num, screenshots, ai_response)
    logger.info(f"Saved profile results to: {results_dir}")

    # Check if profile is undesirable (empty response)
    if not ai_response or not ai_response.get('prompt') or not ai_response.get('response') or not ai_response.get('conversation_starter') or ai_response.get('screenshot_index') == -1:
        logger.info("Profile marked as undesirable - disliking")
        dislike_profile(device)
        disliked_profiles += 1
        return disliked_profiles, total_likes, False

    # Match prompt against authoritative list
    matched_prompt, confidence = match_prompt_against_authoritative(
        ai_response['prompt'], prompts_txt_path)

    if not matched_prompt or confidence < 0.8:
        logger.warning(f"Low confidence prompt match ({confidence:.2f})")
        logger.warning(f"Original: {ai_response['prompt']}")
        logger.warning(f"Matched: {matched_prompt}")
        logger.info("Exiting program due to low confidence prompt match")
        sys.exit(1)

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
                logger.error("Failed to find prompt after all attempts")
                logger.info("Exiting program due to prompt detection failure")
                sys.exit(1)

    # Send the response
    success = send_response_to_story(
        device, ai_response['conversation_starter'], profile_num)

    if not success:
        logger.error("Failed to send response")
        logger.info("Exiting program due to failure to send response")
        sys.exit(1)
    else:
        # Increment likes counter
        total_likes += 1
        logger.info(f"Total likes: {total_likes}/{total_likes_target}")

    return disliked_profiles, total_likes, True


def process_bumble_profile(device, width, height, profile_num, target_likes_before_dislike, disliked_profiles, total_likes, total_likes_target, format_txt_path, prompts_txt_path, interests_txt_path, metadata_txt_path):
    """Process a Bumble profile with scrolling, capturing screenshots, and AI evaluation."""
    # Reset AI response
    global ai_response
    ai_response = None

    # Check for advertisements before processing the profile
    ad_detected = check_for_bumble_advertisement(device, profile_num)
    if ad_detected:
        logger.info("Bumble advertisement detected and dismissed")
        # Wait a moment for the next profile to load properly after dismissing
        time.sleep(4.0)

    # Scroll through profile and capture screenshots
    screenshots = scroll_profile_and_capture(
        device, width, height, profile_num, num_screenshots=9)

    # Check if we need to force dislike based on the counter logic
    # If we've reached our target likes and haven't disliked any, dislike this one
    force_dislike = False
    if total_likes >= target_likes_before_dislike and disliked_profiles == 0:
        force_dislike = True
        logger.info(
            f"Reached {total_likes} likes without any dislikes. Forcing dislike.")

    if force_dislike:
        # Save profile results without AI evaluation
        results_dir = save_profile_results(profile_num, screenshots, None)
        logger.info(f"Saved profile results to: {results_dir}")

        logger.info("Disliking profile based on counter logic")
        # Tap dislike button (coordinates for Bumble)
        tap(device, 150, 1600, with_additional_swipe=False)
        disliked_profiles += 1
        time.sleep(4)  # Wait for next profile to load
        return disliked_profiles, total_likes, False

    # Start AI processing in a separate thread
    logger.info("Starting AI processing in separate thread")
    ai_thread = threading.Thread(
        target=process_bumble_ai_response,
        args=(screenshots, format_txt_path, prompts_txt_path,
              interests_txt_path, metadata_txt_path)
    )
    ai_thread.start()

    # Unlike Hinge, we stay at the bottom of the profile while AI processes
    logger.info("Waiting at the bottom of profile while AI processes")

    # Wait for AI response
    logger.info("Waiting for AI processing to complete")
    ai_thread.join()
    logger.info("AI processing complete")

    # Save profile results with AI response
    results_dir = save_profile_results(profile_num, screenshots, ai_response)
    logger.info(f"Saved profile results to: {results_dir}")

    # Check if profile is undesirable based on AI response
    if not ai_response or ai_response.get('screenshot_index', -1) < 0:
        logger.info("Profile marked as undesirable - disliking")
        # Tap dislike button (coordinates for Bumble)
        tap(device, 150, 1600, with_additional_swipe=False)
        disliked_profiles += 1
        time.sleep(4)  # Wait for next profile to load
        return disliked_profiles, total_likes, False

    # Profile is desirable - like it
    logger.info("Profile marked as desirable - liking")

    # Clear the fields for desirable profiles (Bumble doesn't need these)
    if ai_response:
        ai_response['prompt'] = ""
        ai_response['response'] = ""
        ai_response['conversation_starter'] = ""
        # Update the saved results with the modified response
        save_profile_results(profile_num, screenshots, ai_response)

    # Tap like button (coordinates for Bumble)
    tap(device, 900, 1600, with_additional_swipe=False)
    time.sleep(4)  # Wait for next profile to load

    # Increment likes counter
    total_likes += 1
    logger.info(f"Total likes: {total_likes}/{total_likes_target}")

    return disliked_profiles, total_likes, True


def process_bumble_ai_response(screenshots, format_txt_path, prompts_txt_path, interests_txt_path, metadata_txt_path):
    """Process screenshots from Bumble and generate an AI response."""
    global ai_response
    try:
        result = generate_bumble_reply_from_screenshots(
            screenshots,
            format_txt_path,
            prompts_txt_path,
            interests_txt_path,
            metadata_txt_path,
            compliments_available=False
        )
        with ai_response_lock:
            ai_response = result
    except SystemExit:
        # Re-raise SystemExit to propagate to main thread
        logger.critical(
            "API error occurred in background thread - propagating exit")
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Unhandled exception in Bumble AI processing thread: {e}")
        logger.debug("", exc_info=True)
        with ai_response_lock:
            ai_response = None
        # Also exit the application on unhandled exceptions
        sys.exit(1)


def main():
    """Main function to run the dating app automation."""
    try:
        # Get device IP from environment variable
        device_ip = os.getenv("DEVICE_IP", "192.168.12.32")

        # Select which dating app to use (Hinge, Bumble, or all)
        dating_app = os.getenv("DATING_APP", "hinge").lower()

        # Set up logging based on the selected app
        global logger
        if dating_app != "all":
            # For single app, set up app-specific logger
            logger = setup_logging(app_name=dating_app)
        # For "all" option, keep the default logger for main process

        logger.info(f"Selected dating app: {dating_app}")
        logger.info(f"Connecting to device at IP: {device_ip}")

        # Connect to device
        device = connect_device_remote(device_ip)
        if not device:
            logger.error("Failed to connect to device")
            sys.exit(1)

        # Get screen dimensions
        width, height = get_screen_resolution(device)
        logger.info(f"Screen resolution: {width}x{height}")

        # Initialize profile counter for logging/screenshots only
        profile_num = 1

        if dating_app == "all":
            logger.info("Running automation on all dating apps sequentially")

            # First run on Hinge
            logger.info("Starting automation on Hinge")
            run_automation_on_app(device, width, height, "hinge")

            # Then run on Bumble
            logger.info("Starting automation on Bumble")
            run_automation_on_app(device, width, height, "bumble")

            logger.info("Completed automation on all dating apps")

        else:
            # Run on a single app
            run_automation_on_app(device, width, height, dating_app)

    except KeyboardInterrupt:
        logger.info("\nExiting gracefully...")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        sys.exit(1)


def run_automation_on_app(device, width, height, dating_app):
    """Run the automation on a specific dating app.

    Args:
        device: The ADB device
        width: Screen width
        height: Screen height
        dating_app: The dating app to run automation on (hinge or bumble)
    """
    try:
        # Initialize profile counter for logging/screenshots only
        profile_num = 1

        # Counters for tracking likes and dislikes
        disliked_profiles = 0  # Track if we've disliked any profiles in the current cycle
        total_likes = 0  # Track total likes

        # Initialize file paths
        # Get the absolute path to the app directory
        app_dir = os.path.dirname(os.path.abspath(__file__))

        if dating_app == "hinge":
            # Open Hinge app
            open_hinge(device)

            # Default paths for Hinge text files
            format_txt_path = os.path.join(app_dir, 'hingeFormat.txt')
            prompts_txt_path = os.path.join(app_dir, 'hingePrompts.txt')
            captions_txt_path = os.path.join(app_dir, 'hingeCaptions.txt')
            polls_txt_path = os.path.join(app_dir, 'hingePolls.txt')
            locations_txt_path = os.path.join(app_dir, 'locations.txt')

            # Set a single random target for number of likes before forcing a dislike
            target_likes_before_dislike = random.randint(
                4, 7)  # Random number between 4-7
            total_likes_target = 8

            logger.info(
                f"Using Hinge like/dislike logic: After {target_likes_before_dislike} likes without a dislike, force one. Continue to {total_likes_target} total likes")
        elif dating_app == "bumble":
            # Open Bumble app
            open_bumble(device)

            # Default paths for Bumble text files
            format_txt_path = os.path.join(app_dir, 'bumbleFormat.txt')
            prompts_txt_path = os.path.join(app_dir, 'bumblePrompts.txt')
            interests_txt_path = os.path.join(app_dir, 'bumbleInterests.txt')
            metadata_txt_path = os.path.join(app_dir, 'bumbleMetadata.txt')

            # Set a single random target for number of likes before forcing a dislike
            target_likes_before_dislike = random.randint(
                6, 9)  # Random number between 6-9
            total_likes_target = 9

            logger.info(
                f"Using Bumble like/dislike logic: After {target_likes_before_dislike} likes without a dislike, force one. Continue to {total_likes_target} total likes")
        else:
            logger.error(f"Unsupported dating app: {dating_app}")
            return

        # Continue until we've both reached the target number of likes AND had at least one dislike
        while total_likes < total_likes_target or disliked_profiles == 0:
            logger.info(f"\nProcessing profile #{profile_num}")
            logger.info(
                f"Progress: {total_likes}/{total_likes_target} likes, {disliked_profiles} dislikes")

            # Check if we've reached the end of available profiles
            end_reached, end_message = check_for_end_of_profiles(
                device, profile_num)
            if end_reached:
                logger.info(
                    f"Reached end of available profiles: '{end_message}'. Exiting...")
                break

            # Process profile based on selected dating app
            if dating_app == "hinge":
                disliked_profiles, total_likes, success = process_hinge_profile(
                    device, width, height, profile_num, target_likes_before_dislike,
                    disliked_profiles, total_likes, total_likes_target, format_txt_path, prompts_txt_path,
                    captions_txt_path, polls_txt_path, locations_txt_path
                )
            elif dating_app == "bumble":
                disliked_profiles, total_likes, success = process_bumble_profile(
                    device, width, height, profile_num, target_likes_before_dislike,
                    disliked_profiles, total_likes, total_likes_target, format_txt_path, prompts_txt_path,
                    interests_txt_path, metadata_txt_path
                )

            profile_num += 1

        # Log completion status
        if total_likes >= total_likes_target and disliked_profiles > 0:
            logger.info(
                f"\nSuccess! Completed with {total_likes}/{total_likes_target} likes and {disliked_profiles} dislikes.")
        else:
            logger.info(
                f"Exiting with final stats: {total_likes}/{total_likes_target} likes, {disliked_profiles} dislikes")

        # Close the app when finished
        if dating_app == "hinge":
            close_hinge(device)
        elif dating_app == "bumble":
            close_bumble(device)

    except Exception as e:
        logger.error(f"Error in run_automation_on_app for {dating_app}: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level

        # Make sure to close the app even if there's an error
        try:
            if dating_app == "hinge":
                close_hinge(device)
            elif dating_app == "bumble":
                close_bumble(device)
        except Exception as close_error:
            logger.error(f"Error closing app after exception: {close_error}")


if __name__ == "__main__":
    main()
