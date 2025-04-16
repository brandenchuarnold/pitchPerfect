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
import json

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


def generate_joke_from_json(profile_json):
    """
    Generate a joke based on structured JSON input containing prompts and responses.

    Args:
        profile_json: List of objects containing prompt-response pairs and standalone content

    Returns:
        Tuple of (target_message, joke) where:
        - target_message is the full message (prompt-response pair or standalone response) to respond to
        - joke is the generated joke (1-2 sentences max)
    """
    prompt = f"""
    You will see a structured JSON representation of a woman's dating app profile. The JSON structure follows this format:
    [
        {{
            "prompt": "What's your favorite travel memory?",
            "response": "Backpacking through Thailand and getting lost in the jungle for 3 days"
        }},
        {{
            "prompt": "Two truths and a lie",
            "response": "I've been skydiving, I speak 5 languages, I've never been to Europe"
        }},
        {{
            "prompt": null,
            "response": "I once met a celebrity at a coffee shop and didn't realize it until they left"
        }}
    ]

    Important Context:
    - All prompts in the profile are pre-written questions that the woman has specifically chosen to answer
    - The woman must select three prompts to participate in the app and provide three responses
    - Due to OCR limitations, we may not always capture all three prompts and responses
    - We might see more or fewer prompts/responses than expected
    - Each prompt represents a topic the woman chose to talk about herself
    - The responses are her personal answers to these chosen prompts

    Message Types and Relationships:
    - A "message" refers to any paragraph in the profile, which can be:
      * A prompt (a pre-written question)
      * A response (the woman's answer to a prompt)
      * A prompt-response pair (both the question and answer together)
    - When a prompt and response are paired in the JSON, we have determined with high confidence that this specific response belongs to that specific prompt
    - For standalone prompts or responses, we cannot be certain of their original pairings
    - You may make obvious connections between standalone prompts and responses when:
      * The connection is clear and unambiguous
      * The content strongly suggests a natural pairing
      * The woman's intent is evident from the context
    - However, avoid making speculative or overzealous connections
    - Remember that the woman wrote these responses to be interpreted fairly and accurately
    - When in doubt, treat prompts and responses as independent pieces of information

    The JSON contains three types of messages:
    1. Prompt-response pairs (both prompt and response have values)
    2. Standalone prompts (only prompt has value, response is null)
    3. Standalone responses (only response has value, prompt is null)
    
    The messages are ordered by vertical position on the screen, maintaining the natural flow of the profile.
    
    Use the following step-by-step process to generate a response:
    1. Analyze ALL messages to understand the context
    2. For EACH message, create three distinct observations:
       2a. What does this say about what the woman is logistically doing or involved in, or how she's physically spending her time?
           - Consider her activities, routines, and physical presence in the world
           - Think about where she goes and what she does in her daily life
       2b. What does this say about the woman's intellectual opinions, judgments, reasoning, and train of logic?
           - Consider how she thinks about and analyzes situations
           - Look at her decision-making process and cognitive approach
       2c. What does this say about the woman's values, priorities, feelings, and what she enjoys about life?
           - Consider her emotional responses and what matters to her
           - Think about what brings her joy and fulfillment
    3. Combine ALL observations from EVERY message to form a complete picture of her personality:
       - Each message generates three observations
       - Consider how these observations complement or contrast with each other
       - Look for patterns and themes across all observations
       - Use this comprehensive understanding to inform the joke generation
    4. For EACH response (a paragraph specifically tagged as a response, whether from a pair or standalone), generate ONE joke based on:
       - The complete picture of her personality formed from all observations
       - The most interesting or unexpected aspects of that specific response
       - How this response reflects the woman's dating preferences
       - Do NOT generate jokes based on prompts alone
    5. From all generated jokes, select the funniest one that:
       - Is respectful and appropriate
       - Shows you've paid attention to her profile
       - Has an element of surprise or wit
       - Is CONCISE (1-2 sentences maximum)
    6. Return the full message (either the prompt-response pair or just the response) prefixed by [Personal Statement] and the joke prefixed by [Funny Response]
    
    Profile JSON:
    {json.dumps(profile_json, indent=2)}
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
            target_message = parts[0].replace(
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

    print("\ntarget_message: ", target_message)
    print("\ncomment_text: ", comment_text)
    return comment_text, target_message


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


def create_visual_debug_overlay(image_path, boxes, lines, paragraphs, prompt_paragraphs=None, response_paragraphs=None, pairs=None):
    """Create a visual debugging overlay for text organization.

    Args:
        image_path: Path to the original screenshot
        boxes: List of text boxes with 'text' and 'box' fields
        lines: List of lines, each containing boxes
        paragraphs: List of paragraphs with 'boxes' and 'lines' fields
        prompt_paragraphs: List of paragraph indices that are prompts
        response_paragraphs: List of paragraph indices that are responses
        pairs: List of (prompt_idx, response_idx) tuples for pairs

    Returns:
        Path to the saved visualization image
    """
    import cv2
    import numpy as np

    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Create a copy for drawing
    vis_img = img.copy()

    # Draw text boxes (gray)
    for box in boxes:
        x, y, w, h = box['box']
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (128, 128, 128), 2)
        # Add text label
        cv2.putText(vis_img, box['text'], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Draw lines (red)
    for line in lines:
        # Find bounding box of line
        x_coords = [b['box'][0] for b in line]
        y_coords = [b['box'][1] for b in line]
        widths = [b['box'][2] for b in line]
        heights = [b['box'][3] for b in line]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x + w for x, w in zip(x_coords, widths))
        y_max = max(y + h for y, h in zip(y_coords, heights))

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Draw paragraphs (green)
    for i, para in enumerate(paragraphs):
        # Find bounding box of paragraph
        x_coords = [b['box'][0] for b in para['boxes']]
        y_coords = [b['box'][1] for b in para['boxes']]
        widths = [b['box'][2] for b in para['boxes']]
        heights = [b['box'][3] for b in para['boxes']]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x + w for x, w in zip(x_coords, widths))
        y_max = max(y + h for y, h in zip(y_coords, heights))

        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add paragraph type indicators
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        if prompt_paragraphs and i in prompt_paragraphs:
            cv2.circle(vis_img, (center_x, center_y), 10, (255, 0, 0), -1)
        elif response_paragraphs and i in response_paragraphs:
            cv2.circle(vis_img, (center_x, center_y), 10, (0, 0, 255), -1)

    # Draw pairs (purple)
    if pairs:
        for prompt_idx, response_idx in pairs:
            prompt_para = paragraphs[prompt_idx]
            response_para = paragraphs[response_idx]

            # Get combined bounding box
            prompt_boxes = prompt_para['boxes']
            response_boxes = response_para['boxes']

            all_boxes = prompt_boxes + response_boxes
            x_coords = [b['box'][0] for b in all_boxes]
            y_coords = [b['box'][1] for b in all_boxes]
            widths = [b['box'][2] for b in all_boxes]
            heights = [b['box'][3] for b in all_boxes]

            x_min = min(x_coords) - 5
            y_min = min(y_coords) - 5
            x_max = max(x + w for x, w in zip(x_coords, widths)) + 5
            y_max = max(y + h for y, h in zip(y_coords, heights)) + 5

            cv2.rectangle(vis_img, (x_min, y_min),
                          (x_max, y_max), (255, 0, 255), 2)

    # Save the visualization
    vis_path = image_path.replace('.png', '_visual.png')
    cv2.imwrite(vis_path, vis_img)
    return vis_path
