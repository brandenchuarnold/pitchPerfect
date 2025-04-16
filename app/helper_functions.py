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

    if img is None:
        print("Error: Could not load screenshot.")
        return None, None

    if template is None:
        print("Error: Could not load template.")
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
        res = cv2.matchTemplate(
            img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
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


def type_text_slow(device, text, per_char_delay=0.1):
    """
    Simulates typing text character by character.
    Slower, but you can see it appear on screen.
    """
    for char in text:
        # Handle space character, since 'input text " "' can be problematic
        # '%s' is recognized as a space by ADB shell.
        if char == " ":
            char = "%s"
        # You may also need to handle special characters or quotes
        device.shell(f"input text {char}")
        time.sleep(per_char_delay)


# Use to connect directly
def connect_device(user_ip_address="127.0.0.1"):
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


# Use to connect remotely from docker container
def connect_device_remote(user_ip_address="127.0.0.1"):
    adb = AdbClient(host="host.docker.internal", port=5037)
    connection_result = adb.remote_connect(user_ip_address, 5555)
    print("Connection result:", connection_result)
    devices = adb.devices()

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def capture_screenshot(device, filename):
    result = device.screencap()
    with open("images/" + str(filename) + ".png", "wb") as fp:
        fp.write(result)
    return "images/" + str(filename) + ".png"


def tap(device, x, y):
    """
    Enhanced tap function using multiple methods to ensure the tap registers.
    """
    print(f"Executing tap at coordinates: {x}, {y}")

    # Standard tap
    device.shell(f"input tap {x} {y}")
    time.sleep(0.3)

    # Try a long press (100ms) - sometimes more reliable for buttons
    device.shell(f"input swipe {x} {y} {x} {y} 100")

    return True


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


def generate_comment_with_target(profile_text):
    prompt = f"""
    You will see text of sections of a woman's dataing app profile description, which is a single string of text generated in three steps:
    1. Screenshots are taken by a script of all parts of a woman's profile on a dating app.
    2. Optical Character Recognition (OCR) is used to extract text from each screenshot.
    3. The text from each screenshot is seperated or delimited by newlines from texts from other screenshots. Then it is concatenated together to form a single string.
    Use the following step-by-step process to generate a response to the woman's profile description:
    1. Seperate the profile description into a set of unique sections by delimiting on the new lines within the string of text. Each section represents a personal statement the woman made about herself. Now you have a set of personal statement sections.
    2. For each personal statement section, create three distinct assumptions about what this personal statement conveys about the woman's personality. The three assumptions are described below in 2a, 2b, and 2c.
    2a. Create the first assumption based on how you believe the woman logistically spends her time based on the personal statment. Include things like where she goes and what things she needs and what skills she has based on the personal statement.
    2b. Create the second assumption based on how you believe the woman's sense of humor is based on the personal statement. Include things like how she uses sarcasm and how she uses irony and what she finds funny based on the personal statement.
    2c. Create the third assumption based on how you believe the woman's values are based on the personal statement. Include things like what she believes in and what she stands for and what she is passionate about based on the personal statement.
    Now you have a set of three distinct assumptions based on each personal statement section.
    3. Take every set of three distinct assumptions based on each personal statement section and list them idividually one at a time to form a list that contains all assumptions. Now you have a single list of assumptions which we will henceforth call the woman's characteristics. 
    4. Return to looking at the personal statement sections. For each personal statment section, create a one-sentence joke addressed to the woman of around twelve words which describes the most unintentional or unexpected way in which that specific woman's personal statement section has described her dating preferences. Now you have a set of jokes.
    5. Look at all of the jokes and pick the single joke that is the most funny. Also record the personal statement section that the joke is based on. Now you have a single funny joke and the personal statement section that the joke is based on.
    6. Return the personal statement section prefixed by [Personal Statement] and then a new line and then the single funny response prefixed by [Funny Response]. Do not include any quotes around the personal statement section or the funny response.
    Profile Description:
    {profile_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4.5-preview",
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a highly capable, thoughtful, and precise writer.
                You want to hold a conversation with a woman on an online dating app.
                You know that if this woman finds you physically attractive, she will take the time to read your message.
                You believe that you are a unique person that all women would like to understand more in order to determine if they would like to go on a date with you.
                """
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
    )

    comment = response.choices[0].message["content"].strip()
    print("\ncomment: ", comment)

    # Split based on the bracketed labels
    if "[Personal Statement]" in comment and "[Funny Response]" in comment:
        # Split on [Funny Response] to separate the two parts
        parts = comment.split("[Funny Response]")
        if len(parts) == 2:
            # Extract the personal statement (remove the label)
            target_prompt_text = parts[0].replace(
                "[Personal Statement]", "").strip()
            # The second part is the comment text
            comment_text = parts[1].strip()
        else:
            print(
                "Warning: Could not properly split using [Funny Response] label")
            return "", ""
    else:
        print(
            "Warning: Missing required labels [Personal Statement] or [Funny Response]")
        return "", ""

    print("\ntarget_prompt_text: ", target_prompt_text)
    print("\ncomment_text: ", comment_text)
    return comment_text, target_prompt_text


def get_screen_resolution(device):
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(
        f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)
