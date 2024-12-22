from ppadb.client import Client as AdbClient
import time
from PIL import Image
import numpy as np
import cv2
import pytesseract
import openai
from dotenv import load_dotenv
import os
import cv2
import numpy as np

from helper_functions import (
    connect_device,
    get_screen_resolution,
    open_hinge,
    swipe,
    capture_screenshot,
    extract_text_from_image,
    do_comparision,
    find_icon,
    generate_comment,
    tap,
    input_text,
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    device = connect_device()
    if not device:
        return

    width, height = get_screen_resolution(device)

    # Approximate coordinates based on experimentation
    x_select_like_button_approx = int(width * 0.90)
    y_select_like_button_approx = int(height * 0.67 * 0.75)

    x_select_comment_button_approx = 540
    y_select_comment_button_approx = 1755

    x_select_done_button_approx = int(width * 0.85)
    y_select_done_button_approx = int(height * 0.50)

    x_send_like_button = int(width * 0.75)
    y_send_like_button = int(height * 0.80)

    x_dislike_button_approx = int(width * 0.15)
    y_dislike_button_approx = int(height * 0.85)

    x1_swipe = int(width * 0.15)
    x2_swipe = x1_swipe

    y1_swipe = int(height * 0.5)
    y2_swipe = int(y1_swipe * 0.75)

    # Load sample images for matching criteria
    like_images = [cv2.imread(path) for path in ["like2.jpeg"] if os.path.exists(path)]
    dislike_images = [
        cv2.imread(path) for path in ["dislike.jpeg"] if os.path.exists(path)
    ]

    open_hinge(device=device)
    time.sleep(5)

    previous_profile_text = ""  # Track previous profile's text

    for _ in range(10):
        swipe(device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        screenshot_path = capture_screenshot(device, "screen")

        # OCR for text extraction
        current_profile_text = extract_text_from_image(screenshot_path).strip()
        if not current_profile_text:
            print("Warning: OCR returned empty text.")

        profile_image = cv2.imread(screenshot_path)

        # Compute similarity scores (lower = better match)
        match_like = do_comparision(profile_image, like_images)
        match_dislike = do_comparision(profile_image, dislike_images)

        print("Calculated scores => Like:", match_like, "Dislike:", match_dislike)

        # Identify like button coordinates
        x_select_like_button, y_select_like_button = find_icon(
            "screen.png",
            "heart1.png",
            threshold=0.75,
            min_matches=10,
            approx_x=x_select_like_button_approx,
            approx_y=y_select_like_button_approx,
        )

        # Decision-making logic:
        # If like score is better (lower) than dislike score, and we found the like button
        if (
            match_like < match_dislike
            and x_select_like_button is not None
            and y_select_like_button is not None
            or True
        ):
            comment = (
                generate_comment(current_profile_text)
                if current_profile_text
                else "Hey, I'd love to meet up!"
            )
            print(f"Generated Comment: {comment}")

            tap(device, x_select_like_button, y_select_like_button)
            print("Like tapped at:", x_select_like_button, y_select_like_button)

            tap(device, x_select_comment_button_approx, y_select_comment_button_approx)
            input_text(device, comment)
            # type_text_slow(device, comment, per_char_delay=0.1)
            # You can uncomment these if you want to finalize the comment and send the like:
            capture_screenshot(device, "screen_after_message")
            tap(device, x_send_like_button, y_send_like_button)
        else:
            # If same profile text as previous, might be stuck, handle accordingly
            if (
                previous_profile_text == current_profile_text
                and current_profile_text != ""
            ):
                print("Dislike (same profile encountered again)")
            else:
                print("Dislike (new profile or no like match)")

            print(
                "Dislike tapped at:", x_dislike_button_approx, y_dislike_button_approx
            )
            tap(device, x_dislike_button_approx, y_dislike_button_approx)

        previous_profile_text = current_profile_text
        time.sleep(2)


main()
