from ppadb.client import Client as AdbClient
import time
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import anthropic
from dotenv import load_dotenv
import os
import json
from config import ANTHROPIC_API_KEY
import difflib

load_dotenv()
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def connect_device(user_ip_address="127.0.0.1"):
    """Connect to a device directly"""
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def connect_device_remote(user_ip_address="127.0.0.1"):
    """Connect to a device remotely from docker container"""
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
    """Capture a screenshot and save it to the images directory"""
    result = device.screencap()
    with open("images/" + str(filename) + ".png", "wb") as fp:
        fp.write(result)
    return "images/" + str(filename) + ".png"


def tap(device, x, y, double_tap=False):
    """Execute a tap or double tap at the given coordinates"""
    print(
        f"Executing {'double tap' if double_tap else 'tap'} at coordinates: {x}, {y}")
    if double_tap:
        # First tap
        device.shell(f"input tap {x} {y}")
        time.sleep(0.15)  # Brief pause between taps
        # Second tap
        device.shell(f"input tap {x} {y}")
        time.sleep(0.5)  # Wait for double-tap to register
    else:
        device.shell(f"input tap {x} {y}")
        time.sleep(0.3)
        # Additional swipe to ensure tap registers
        device.shell(f"input swipe {x} {y} {x} {y} 100")
    return True


def input_text(device, text):
    """Input text into the device"""
    text = text.replace(" ", "%s")
    print("text to be written: ", text)
    device.shell(f'input text "{text}"')


def type_text_slow(device, text, per_char_delay=0.1):
    """Type text character by character with a delay"""
    for char in text:
        if char == " ":
            char = "%s"
        device.shell(f"input text {char}")
        time.sleep(per_char_delay)


def swipe(device, direction="down", duration=1500):
    """Execute a swipe gesture with consistent 68% scroll distance.

    Args:
        device: The ADB device
        direction: "up" or "down" to specify scroll direction
        duration: Duration of swipe in milliseconds

    Returns:
        tuple: (True, 0) for compatibility with existing code
    """
    try:
        width, height = get_screen_resolution(device)
        x_scroll = int(width * 0.5)  # Center of screen
        y_scroll_start = int(height * 0.84)  # Start at 84% of screen height
        y_scroll_end = int(height * 0.16)    # End at 16% of screen height

        # Execute the swipe in the specified direction
        if direction == "down":
            device.shell(
                f"input swipe {x_scroll} {y_scroll_start} {x_scroll} {y_scroll_end} {duration}")
        else:  # up
            device.shell(
                f"input swipe {x_scroll} {y_scroll_end} {x_scroll} {y_scroll_start} {duration}")

        time.sleep(0.5)  # Wait for scroll to complete
        return True, 0

    except Exception as e:
        print(f"Error during swipe: {e}")
        return False, 0


def get_screen_resolution(device):
    """Get the screen resolution of the device"""
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def open_hinge(device):
    """Open the Hinge app"""
    package_name = "co.match.android.matchhinge"
    device.shell(
        f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)


def isAtBottom(device, last_scroll_position=None):
    """Check if we've reached the bottom of a scrollable view using multiple methods.

    Args:
        device: The ADB device
        last_scroll_position: The Y position from the last scroll operation

    Returns:
        tuple: (bool, int) - (is_at_bottom, current_scroll_position)
    """
    try:
        # Method 1: Check scroll position
        window_info = device.shell("dumpsys window")
        current_position = 0
        for line in window_info.split('\n'):
            if 'mScrollY=' in line:
                current_position = int(line.split('=')[1].strip())
                break

        # If we have a last position to compare with
        if last_scroll_position is not None:
            # If we haven't moved since last scroll, we might be at the bottom
            if current_position == last_scroll_position:
                # Method 2: Check if we can scroll further using UiScrollable
                scrollable_check = device.shell("uiautomator dump /dev/tty")
                if "scrollable=\"true\"" in scrollable_check:
                    return False, current_position
                else:
                    return True, current_position

        return False, current_position

    except Exception as e:
        print(f"Error checking scroll position: {e}")
        return False, 0


def scroll_down(device, last_scroll_position=None):
    """Scroll down the screen and return the new scroll position.

    Args:
        device: The ADB device
        last_scroll_position: The Y position from the last scroll operation

    Returns:
        tuple: (bool, int) - (success, new_scroll_position)
    """
    try:
        width, height = get_screen_resolution(device)
        x_center = width // 2
        y_start = height * 0.8
        y_end = height * 0.2  # Scroll a larger distance to ensure movement

        # Execute the scroll
        device.shell(
            f"input swipe {x_center} {y_start} {x_center} {y_end} 500")
        time.sleep(0.5)  # Wait for scroll to complete

        # Get new scroll position and check if we're at the bottom
        is_bottom, new_position = isAtBottom(device, last_scroll_position)

        # If we're at the bottom, try one more time with a smaller scroll
        if is_bottom:
            device.shell(
                f"input swipe {x_center} {height*0.6} {x_center} {height*0.4} 300")
            time.sleep(0.5)
            is_bottom, new_position = isAtBottom(device, new_position)

        return not is_bottom, new_position

    except Exception as e:
        print(f"Error during scroll: {e}")
        return False, last_scroll_position


def extract_text_from_image_with_boxes(image_path):
    """Extract text and bounding boxes from an image using OCR"""
    try:
        image = Image.open(image_path)
        ocr_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT)

        boxes = []
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():
                box = {
                    'text': ocr_data['text'][i],
                    'box': (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['width'][i],
                        ocr_data['height'][i]
                    )
                }
                boxes.append(box)
        return boxes
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None


def group_boxes_into_lines(boxes, y_threshold=15):
    """Group text boxes into lines based on vertical alignment.

    Args:
        boxes: List of text boxes with 'text' and 'box' fields
        y_threshold: Maximum vertical distance for boxes to be on same line

    Returns:
        List of lines, where each line is a list of boxes
    """
    if not boxes:
        return []

    # Sort boxes by vertical position
    sorted_boxes = sorted(boxes, key=lambda b: b['box'][1])

    lines = []
    current_line = []
    current_y = None
    current_height = None

    for box in sorted_boxes:
        box_y = box['box'][1]
        box_height = box['box'][3]

        if current_y is None:
            # First box
            current_y = box_y
            current_height = box_height
            current_line.append(box)
        else:
            # Calculate vertical overlap
            # If boxes overlap vertically or are within y_threshold, consider them on same line
            box_top = box_y
            box_bottom = box_y + box_height
            current_top = current_y
            current_bottom = current_y + current_height

            vertical_overlap = (min(box_bottom, current_bottom) -
                                max(box_top, current_top))

            if vertical_overlap > 0 or abs(box_y - current_y) <= y_threshold:
                # Box is aligned with current line
                current_line.append(box)
                # Update current line bounds
                current_y = min(current_y, box_y)
                current_height = max(current_height,
                                     box_y + box_height - current_y)
            else:
                # Box starts a new line
                if current_line:
                    # Sort boxes in line by x position
                    current_line.sort(key=lambda b: b['box'][0])
                    lines.append(current_line)
                current_line = [box]
                current_y = box_y
                current_height = box_height

    # Add the last line
    if current_line:
        current_line.sort(key=lambda b: b['box'][0])
        lines.append(current_line)

    return lines


def group_lines_into_paragraphs(lines, paragraph_spacing=20):
    """Group lines into paragraphs based on vertical spacing.

    Args:
        lines: List of lines, where each line is a list of boxes
        paragraph_spacing: Minimum vertical distance between paragraphs

    Returns:
        List of paragraphs, where each paragraph contains:
        - text: Combined text of all boxes
        - boxes: List of all boxes in the paragraph
        - lines: List of lines in the paragraph
    """
    if not lines:
        return []

    paragraphs = []
    current_para = {
        'text': '',
        'boxes': [],
        'lines': []
    }
    last_line_bottom = None

    for line in lines:
        # Get vertical bounds of this line
        line_top = min(box['box'][1] for box in line)
        line_bottom = max(box['box'][1] + box['box'][3] for box in line)

        if last_line_bottom is None:
            # First line
            current_para['lines'].append(line)
            current_para['boxes'].extend(line)
        elif line_top - last_line_bottom <= paragraph_spacing:
            # Line is close enough to previous line - same paragraph
            current_para['lines'].append(line)
            current_para['boxes'].extend(line)
        else:
            # Line is far from previous line - start new paragraph
            if current_para['boxes']:
                current_para['text'] = ' '.join(
                    box['text'] for box in current_para['boxes'])
                paragraphs.append(current_para)
                current_para = {
                    'text': '',
                    'boxes': line.copy(),
                    'lines': [line]
                }

        last_line_bottom = line_bottom

    # Add the last paragraph
    if current_para['boxes']:
        current_para['text'] = ' '.join(box['text']
                                        for box in current_para['boxes'])
        paragraphs.append(current_para)

    return paragraphs


def fuzzy_match_text(target_text, text_to_match, threshold=0.8):
    """Perform fuzzy matching between two text strings.

    Args:
        target_text: The text we're looking for
        text_to_match: The text we're comparing against
        threshold: Minimum similarity ratio to consider a match (0.0 to 1.0)

    Returns:
        tuple: (is_match, similarity_ratio, matched_text)
    """
    # Convert to lowercase for case-insensitive matching
    target_lower = target_text.lower().strip()
    match_lower = text_to_match.lower().strip()

    # Get base similarity ratio using sequence matcher
    ratio = difflib.SequenceMatcher(None, target_lower, match_lower).ratio()

    # Apply length penalty if strings are very different in length
    len_ratio = min(len(target_lower), len(match_lower)) / \
        max(len(target_lower), len(match_lower))
    ratio = ratio * len_ratio

    # Small bonus (max 0.1) if one string contains the other
    contains_bonus = 0.0
    if target_lower in match_lower or match_lower in target_lower:
        # Bonus scaled by the length ratio of the contained string
        contains_ratio = min(len(target_lower), len(
            match_lower)) / max(len(target_lower), len(match_lower))
        contains_bonus = 0.1 * contains_ratio

    final_ratio = min(1.0, ratio + contains_bonus)

    # Only consider it a match if the ratio is above threshold
    is_match = final_ratio >= threshold
    return is_match, final_ratio, text_to_match


def create_visual_debug_overlay(image_path, boxes, lines=None, paragraphs=None, output_path=None, tap_target=None):
    """Create a visual debugging overlay showing text boxes, lines, and paragraphs.

    Args:
        image_path: Path to the original screenshot
        boxes: List of text boxes with 'text' and 'box' fields
        lines: Optional list of line groupings
        paragraphs: Optional list of paragraph groupings
        output_path: Optional path to save the visualization
        tap_target: Optional (x, y) coordinates for tap target

    Returns:
        PIL Image object with the visualization overlay
    """
    # Load the original image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Create output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_visual{ext}"

    # Draw text boxes (gray)
    for box in boxes:
        x, y, w, h = box['box']
        # Draw box
        draw.rectangle([x, y, x + w, y + h], outline='gray', width=2)
        # Add text label - handle Unicode text
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        try:
            # Try to draw the text, replacing any problematic characters
            safe_text = box['text'].encode('ascii', 'replace').decode('ascii')
            draw.text((x, y - 15), safe_text, fill='gray', font=font)
        except Exception as e:
            print(f"Warning: Could not draw text '{box['text']}': {e}")

    # Draw lines (red)
    if lines:
        for line in lines:
            # Find bounding box of line
            min_x = min(box['box'][0] for box in line)
            min_y = min(box['box'][1] for box in line)
            max_x = max(box['box'][0] + box['box'][2] for box in line)
            max_y = max(box['box'][1] + box['box'][3] for box in line)
            draw.rectangle([min_x, min_y, max_x, max_y],
                           outline='red', width=2)

    # Draw paragraphs (green)
    if paragraphs:
        for para in paragraphs:
            # Find bounding box of paragraph
            min_x = min(box['box'][0] for box in para['boxes'])
            min_y = min(box['box'][1] for box in para['boxes'])
            max_x = max(box['box'][0] + box['box'][2] for box in para['boxes'])
            max_y = max(box['box'][1] + box['box'][3] for box in para['boxes'])
            draw.rectangle([min_x, min_y, max_x, max_y],
                           outline='green', width=2)

    # Draw tap target circle if provided
    if tap_target:
        tap_x, tap_y = tap_target
        radius = 30
        draw.ellipse([tap_x - radius, tap_y - radius,
                     tap_x + radius, tap_y + radius],
                     outline='red', width=3)

    # Save the visualization
    img.save(output_path)
    return img


def generate_joke_from_screenshots(screenshots, format_txt_path, prompts_txt_path, captions_txt_path, polls_txt_path):
    """
    Generate a contextually appropriate joke response for a Hinge profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to format.txt describing profile structure
        prompts_txt_path: Path to prompts.txt containing available prompts
        captions_txt_path: Path to captions.txt containing possible captions
        polls_txt_path: Path to polls.txt containing available polls

    Returns:
        dict: Contains the prompt-response pair, generated joke, and screenshot index:
        {
            "prompt": str,      # The exact prompt text being responded to
            "response": str,    # The user's response to the prompt
            "conversation_starter": str,  # The generated conversation starter
            "screenshot_index": int,  # 0-based index of screenshot containing prompt/response
        }
    """
    # Read the content of our context files
    try:
        with open(format_txt_path, 'r') as f:
            format_content = f.read()
        with open(prompts_txt_path, 'r') as f:
            prompts_content = f.read()
        with open(captions_txt_path, 'r') as f:
            captions_content = f.read()
        with open(polls_txt_path, 'r') as f:
            polls_content = f.read()
    except Exception as e:
        print(f"Error reading context files: {e}")
        return None

    # Convert screenshots to base64 for API transmission
    import base64
    screenshot_data = []
    for i, screenshot_path in enumerate(screenshots):
        try:
            with open(screenshot_path, 'rb') as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
                screenshot_data.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded_string
                    }
                })
        except Exception as e:
            print(f"Error processing screenshot {i}: {e}")
            return None

    # System prompt containing all the structural information
    system_prompt = f"""You are a witty and natural conversationalist on a dating app. Your task is to analyze Hinge profiles and generate engaging conversation starters based on the prompts and responses in a woman's profile. Since she's already expressed interest by matching, balance natural conversation with clear intent - keeping it light while being specific enough for text.

PROFILE STRUCTURE:
You will receive exactly 6 screenshots of a Hinge profile in order (index 0 to 5). Each profile will contain the following guaranteed elements:
1. Exactly 6 photos (may have captions)
2. Exactly 3 prompt/response pairs
3. One section of profile basics
4. Optionally one voice prompt
5. Optionally one poll prompt

Each of these elements is a "story" about the woman - something she has chosen to share about herself. There will always be at least 10 stories (6 photos + 3 prompts + 1 basics) and up to 12 stories if she includes a voice prompt and poll prompt.

Your process:

STEP 1: READ AND UNDERSTAND THE CONTEXT
1. Read format.txt to understand the profile layout
2. Consult prompts.txt, captions.txt, and polls.txt to understand possible elements
3. Examine each screenshot in order (0 to 5) and identify all elements

STEP 2: ORGANIZE ELEMENTS INTO BUCKETS
For each screenshot, group elements into these buckets:
a. Photos (6 total)
   - May have a caption
   - If captioned, understand she chose that caption to describe the photo
b. Prompts and Responses (3 total)
   - Understand she chose the prompt and provided her own response
c. Profile Basics (1 total)
   - All bullet-points she provided to describe herself
d. Voice Prompt and/or Poll Prompt (0-2 total)
   - Voice prompt: Cannot understand the recording, only see the prompt
   - Poll prompt: Can read her provided options for engagement

STEP 3: ANALYZE EACH STORY'S IMPLICATIONS
For each story (element), determine what it reveals about the woman:
1. Photos: Consider what she's doing and how she made it happen
   - Example: "Playing uno in forest" → brought cards to wilderness
2. Prompt/Response: Consider her history and preferences
   - Example: "Bar trivia and salty snacks" → enjoys trivia as hobby
3. Profile Basics: Take her descriptions at face value
4. Voice Prompt: Unknown content, only prompt visible
5. Poll Prompt: Shows topics she'd discuss

STEP 4: COMPILE CHARACTERISTICS
1. List all characteristics derived from stories
2. Organize by confidence level
3. Note when multiple stories support same characteristic
   - Example: Two stories about exercise → high confidence in active lifestyle

STEP 5: GENERATE CONVERSATION STARTERS
For each prompt/response pair:
1. Use characteristics and stories as context
2. Create exactly ONE conversation starter that:
   * Uses natural, everyday language - how you'd actually talk to someone
   * Avoids flowery metaphors or dramatic language
   * Never uses marketing-style words like "perfect", "amazing", "journey", "adventure"
   * Asks direct questions without unnecessary qualifiers
   * Shows interest through specificity, not through enthusiasm
   * Aims for 15-20 words maximum
   * Easy to respond to over text

CRITICAL: People don't narrate their lives with dramatic language - they just ask what they want to know.

Examples of natural vs artificial language:
Natural (Good):
- "What book are you reading?"
- "How long have you been surfing?"
- "Which hiking trails do you like around here?"

Artificial (Avoid):
- "What book are you diving into there?"
- "How long have you been on your surfing journey?"
- "Which perfect hiking spots have you discovered?"

The key difference is that natural language:
1. Gets straight to the point
2. Doesn't add unnecessary drama or narrative
3. Asks what you actually want to know
4. Treats activities as normal parts of life, not epic adventures

STEP 6: SIMULATE CONVERSATION
For each starter:
1. Imagine her potential responses
2. Evaluate:
   - Would this sound natural in real conversation?
   - Does it avoid sounding like marketing copy?
   - Is it free of unnecessary qualifiers and drama?
   - Does it ask what you actually want to know?
   - Would she find it easy to respond?

STEP 7: SELECT BEST STARTER
1. Choose the starter that best:
   - Uses the most natural, everyday language
   - Gets straight to the point
   - Shows interest through specificity, not enthusiasm
   - Creates easy conversation flow
   - Respects the complexity of her personality
2. Note which prompt/response pair of the woman's profile matches the chosen starter
3. Reference prompts.txt and seperate the prompt/response pair into the prompt and the response. Take note of the prompt distinctly from the response.

STEP 8: IDENTIFY SCREENSHOT
1. Note which screenshot (0-5) contains the prompt of the woman's profile that matches the chosen starter
2. If prompt spans screenshots, use the one with most text

Return the chosen prompt, response, your conversation starter, and the screenshot index in this JSON format exactly. Do not return any other text or comments beyond the JSON.
{{
    "prompt": "The exact prompt text the woman chose",
    "response": "The woman's response to the prompt",
    "conversation_starter": "Your natural conversation starter",
    "screenshot_index": index_of_screenshot_containing_prompt_response  # 0-based index (0-5)
}}"""

    # User message - just the specific task
    user_message = """Please analyze these profile screenshots and generate a conversation starter based on the woman's existing response. Remember to:
1. First build your understanding of the woman's personality
2. Then identify prompt-response pairs and generate starters that balance flattery, humor, and flirtation
3. Select the best starter based on simulated conversation flow"""

    # Create the message for Claude
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                },
                *screenshot_data  # Add all screenshots as subsequent content items
            ]
        }
    ]

    try:
        # Make the API call to Claude
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=1,
            system=system_prompt,
            messages=messages
        )

        # Parse the response
        try:
            # The response should be in JSON format as requested
            import json
            result = json.loads(response.content[0].text)
            return {
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "conversation_starter": result.get("conversation_starter", ""),
                "screenshot_index": result.get("screenshot_index", 0)
            }
        except json.JSONDecodeError:
            print("Error: Response was not in expected JSON format")
            print("Raw response:", response.content[0].text)
            return None
        except (ValueError, TypeError) as e:
            print(f"Error parsing response values: {e}")
            return None

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def detect_prompt_in_screenshot(device, target_prompt, target_response, screenshot_index, profile_num):
    """Detect and visualize the target prompt or response in a screenshot.

    Args:
        device: The ADB device
        target_prompt: The prompt text we're looking for
        target_response: The response text we're looking for
        screenshot_index: Index of the screenshot to analyze
        profile_num: Current profile number

    Returns:
        tuple: (found, tap_coordinates) where:
            - found: bool indicating if prompt/response was found
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

    # First try to find the target prompt with high confidence
    print("\nDebug: Comparing target prompt against OCR paragraphs:")
    print(f"Target prompt: '{target_prompt}'")
    print("\nOCR paragraphs found:")

    best_prompt_match = None
    best_prompt_ratio = 0.0
    best_response_match = None
    best_response_ratio = 0.0

    for i, para in enumerate(paragraphs):
        # Check for prompt match
        is_prompt_match, prompt_ratio, matched_text = fuzzy_match_text(
            target_prompt, para['text'], threshold=0.8)
        print(f"Paragraph {i+1}:")
        print(f"  Text: '{para['text']}'")
        print(f"  Prompt match ratio: {prompt_ratio:.2f}")

        # Also check for response match with lower threshold
        is_response_match, response_ratio, _ = fuzzy_match_text(
            target_response, para['text'], threshold=0.7)
        print(f"  Response match ratio: {response_ratio:.2f}")
        print()

        if is_prompt_match and prompt_ratio > best_prompt_ratio:
            best_prompt_match = para
            best_prompt_ratio = prompt_ratio
        elif is_response_match and response_ratio > best_response_ratio:
            best_response_match = para
            best_response_ratio = response_ratio

    # Use prompt match if found, otherwise use response match
    best_match = best_prompt_match if best_prompt_match else best_response_match
    if best_match:
        print(f"Found {'prompt' if best_prompt_match else 'response'} match with ratio {max(best_prompt_ratio, best_response_ratio):.2f}")

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
        print("No matching prompt or response found in screenshot")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_prompt_detection_visual.png"
        )

        # Fallback: Scroll to bottom and double-click center
        print("\nFallback: Scrolling to bottom and double-clicking center...")

        # Calculate remaining scrolls (we've already done screenshot_index scrolls)
        # 5 is max scrolls (6 screenshots total, 0-5)
        remaining_scrolls = 5 - screenshot_index

        # Scroll the remaining distance to bottom
        for i in range(remaining_scrolls):
            print(f"Fallback scroll #{i+1}")
            swipe(device, "down")
            time.sleep(1)  # Wait for scroll to complete

        # Get screen dimensions for center tap
        width, height = get_screen_resolution(device)
        center_x = width // 2
        center_y = height // 2

        print(f"Double-clicking center of screen at ({center_x}, {center_y})")
        tap(device, center_x, center_y, double_tap=True)
        time.sleep(1)  # Wait for response interface to open

        # Return True since we executed the fallback
        return True, (center_x, center_y)


def dislike_profile(device):
    """Execute a dislike action by tapping the X button.

    Args:
        device: The ADB device

    Returns:
        None
    """
    # Dislike button is always at x=125, y=2075
    tap(device, 125, 2075)
    # Wait 4 seconds for next profile to load
    time.sleep(4)


def send_response_to_story(device, conversation_starter, profile_num):
    """Handle the flow of responding to an opened story.

    Args:
        device: The ADB device
        conversation_starter: The text to send as a response
        profile_num: Current profile number for debugging visualization

    Returns:
        bool: True if response was sent successfully, False otherwise
    """
    # PHASE 1: Find and click comment box
    print("\nPhase 1: Locating comment box...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase1")

    # Extract text and boxes
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        print("No text boxes found in initial screenshot")
        return False

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Find "Add a comment" box
    comment_box = None
    comment_ratio = 0.0
    for para in paragraphs:
        is_match, ratio, _ = fuzzy_match_text("Add a comment", para['text'])
        if is_match and ratio > comment_ratio:
            comment_box = para
            comment_ratio = ratio

    if not comment_box:
        print("Could not find comment box")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_response_phase1_visual.png"
        )
        return False

    # Calculate tap coordinates for comment box
    comment_boxes = comment_box['boxes']
    comment_x = (min(box['box'][0] for box in comment_boxes) +
                 max(box['box'][0] + box['box'][2] for box in comment_boxes)) // 2
    comment_y = (min(box['box'][1] for box in comment_boxes) +
                 max(box['box'][1] + box['box'][3] for box in comment_boxes)) // 2

    # Create visualization of phase 1
    create_visual_debug_overlay(
        screenshot_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/profile_{profile_num}_response_phase1_visual.png",
        tap_target=(comment_x, comment_y)
    )

    # Click comment box and enter text
    tap(device, comment_x, comment_y)
    time.sleep(0.5)
    input_text(device, conversation_starter)
    time.sleep(0.5)

    # Close keyboard
    device.shell('input keyevent 4')  # KEYCODE_BACK
    time.sleep(0.5)

    # PHASE 2: Find and click Send Priority Like button in new layout
    print("\nPhase 2: Locating Send Priority Like button...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase2")

    # Extract text and boxes again for new layout
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        print("No text boxes found in post-input screenshot")
        return False

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Find "Send Priority Like" button in new layout
    send_button = None
    send_ratio = 0.0
    for para in paragraphs:
        is_match, ratio, _ = fuzzy_match_text(
            "Send Priority Like", para['text'])
        if is_match and ratio > send_ratio:
            send_button = para
            send_ratio = ratio

    if not send_button:
        print("Could not find Send Priority Like button")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_response_phase2_visual.png"
        )
        return False

    # Calculate tap coordinates for send button
    send_boxes = send_button['boxes']
    send_x = (min(box['box'][0] for box in send_boxes) +
              max(box['box'][0] + box['box'][2] for box in send_boxes)) // 2
    send_y = (min(box['box'][1] for box in send_boxes) +
              max(box['box'][1] + box['box'][3] for box in send_boxes)) // 2

    # Create visualization of phase 2
    create_visual_debug_overlay(
        screenshot_path,
        boxes=boxes,
        lines=lines,
        paragraphs=paragraphs,
        output_path=f"images/profile_{profile_num}_response_phase2_visual.png",
        tap_target=(send_x, send_y)
    )

    # Click Send Priority Like button
    tap(device, send_x, send_y)

    # Wait 4 seconds for next profile to load
    time.sleep(4)

    return True
