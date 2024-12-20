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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def find_icon(
    screenshot_path,
    template_path,
    approx_x=None,
    approx_y=None,
    margin_x=100,
    margin_y=100,
    min_matches=10,
    threshold=0.8,
    scales=[0.9, 1.0, 1.1],
):
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if img is None or template is None:
        print("Error: Could not load screenshot or template.")
        return None, None

    if approx_x is not None and approx_y is not None:
        H, W = img.shape[:2]
        x_start = max(0, approx_x - margin_x)
        y_start = max(0, approx_y - margin_y)
        x_end = min(W, approx_x + margin_x)
        y_end = min(H, approx_y + margin_y)
        cropped_img = img[y_start:y_end, x_start:x_end]
        offset_x, offset_y = x_start, y_start
    else:
        cropped_img = img
        offset_x, offset_y = 0, 0

    scene_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) > min_matches:
            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h_t, w_t = template_gray.shape
                pts = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(
                    -1, 1, 2
                )
                dst_corners = cv2.perspectiveTransform(pts, M)

                center_x_cropped = int(np.mean(dst_corners[:, 0, 0]))
                center_y_cropped = int(np.mean(dst_corners[:, 0, 1]))
                center_x = center_x_cropped + offset_x
                center_y = center_y_cropped + offset_y
                return center_x, center_y

    # Fallback: Multi-Scale Template Matching
    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    w_t, h_t = template_gray.shape[::-1]

    for scale in scales:
        resized_template = cv2.resize(
            template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        if len(loc[0]) != 0:
            top_left = (loc[1][0], loc[0][0])
            tw, th = resized_template.shape[::-1]
            center_x_cropped = top_left[0] + tw // 2
            center_y_cropped = top_left[1] + th // 2
            center_x = center_x_cropped + offset_x
            center_y = center_y_cropped + offset_y
            return center_x, center_y

    # If no match found
    return None, None


def connect_device():
    adb = AdbClient(host="127.0.0.1", port=5037)
    devices = adb.devices()
    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def capture_screenshot(device):
    result = device.screencap()
    with open("screen.png", "wb") as fp:
        fp.write(result)
    return "screen.png"


def tap(device, x, y):
    device.shell(f"input tap {x} {y}")


def input_text(device, text):
    # Escape spaces in the text
    text = text.replace(" ", "%s")
    print("text to be written: ", text)
    device.shell(f'input text "{text}"')


def swipe(device, x1, y1, x2, y2, duration=500):
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def do_comparision(profile_image, sample_images):
    """
    Returns an average distance score for the best match among the sample_images.
    A lower score indicates a better match.
    If no matches found, returns a high value (indicating poor match).
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(profile_image, None)
    if des1 is None or len(des1) == 0:
        return float("inf")  # No features in profile image

    best_score = float("inf")
    for sample_image in sample_images:
        kp2, des2 = orb.detectAndCompute(sample_image, None)
        if des2 is None or len(des2) == 0:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            continue

        matches = sorted(matches, key=lambda x: x.distance)
        score = sum([match.distance for match in matches]) / len(matches)
        if score < best_score:
            best_score = score

    return best_score if best_score != float("inf") else float("inf")


def generate_comment(profile_text):
    prompt = f"""
    Based on the following profile description, generate a 1-line friendly and personalized comment asking them to go out with you:

    Profile Description:
    {profile_text}

    Comment:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly and likable person who is witty and humorous",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    comment = response.choices[0].message["content"].strip()
    return comment


def get_screen_resolution(device):
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)


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
        screenshot_path = capture_screenshot(device)

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
            # You can uncomment these if you want to finalize the comment and send the like:
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
