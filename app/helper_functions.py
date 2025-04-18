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


def tap(device, x, y):
    """Execute a tap at the given coordinates"""
    print(f"Executing tap at coordinates: {x}, {y}")
    device.shell(f"input tap {x} {y}")
    time.sleep(0.3)
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


def swipe(device, x1, y1, x2, y2, duration=500):
    """Execute a swipe gesture"""
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


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


def isAtBottom(device):
    """Check if we've reached the bottom of a scrollable view using UiScrollable.

    Returns:
        bool: True if at bottom, False otherwise
    """
    try:
        # Get the UiScrollable object for the current view
        scrollable = device.uiAutomator.getUiScrollable()

        # Get max and current scroll steps
        max_steps = scrollable.getMaxScrollSteps()
        current_steps = scrollable.getCurrentScrollSteps()

        # If we can't get steps, try alternative method
        if max_steps is None or current_steps is None:
            return not scrollable.canScrollVertically()

        # Add small threshold to account for floating point differences
        return abs(max_steps - current_steps) < 0.1
    except Exception as e:
        print(f"Error checking scroll position: {e}")
        return False


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


def group_boxes_into_lines(boxes, y_threshold=10):
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

    for box in sorted_boxes:
        box_y = box['box'][1]

        if current_y is None:
            current_y = box_y
            current_line.append(box)
        elif abs(box_y - current_y) <= y_threshold:
            current_line.append(box)
        else:
            # Sort boxes in line by horizontal position
            current_line.sort(key=lambda b: b['box'][0])
            lines.append(current_line)
            current_line = [box]
            current_y = box_y

    # Add the last line
    if current_line:
        current_line.sort(key=lambda b: b['box'][0])
        lines.append(current_line)

    return lines


def group_lines_into_paragraphs(lines, paragraph_spacing=30):
    """Group lines into paragraphs based on vertical spacing.

    Args:
        lines: List of lines, where each line is a list of boxes
        paragraph_spacing: Minimum vertical distance between paragraphs

    Returns:
        List of paragraphs, where each paragraph contains:
        - text: Combined text of all boxes
        - boxes: List of all boxes in the paragraph
        - lines: List of lines in the paragraph
        - spatial_info: Metadata about the paragraph's position
    """
    if not lines:
        return []

    paragraphs = []
    current_para = {
        'text': '',
        'boxes': [],
        'lines': [],
        'spatial_info': None
    }
    last_line_bottom = None

    for line in lines:
        # Get the bottom of this line
        line_bottom = max(box['box'][1] + box['box'][3] for box in line)

        if last_line_bottom is None:
            # First line
            last_line_bottom = line_bottom
            current_para['lines'].append(line)
            current_para['boxes'].extend(line)
        elif line_bottom - last_line_bottom <= paragraph_spacing:
            # Continue current paragraph
            last_line_bottom = line_bottom
            current_para['lines'].append(line)
            current_para['boxes'].extend(line)
        else:
            # Start new paragraph
            # Finalize current paragraph
            current_para['text'] = ' '.join(
                box['text'] for box in current_para['boxes'])
            paragraphs.append(current_para)

            # Start new paragraph
            current_para = {
                'text': '',
                'boxes': line.copy(),
                'lines': [line],
                'spatial_info': None
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
    target_lower = target_text.lower()
    match_lower = text_to_match.lower()

    # Try different matching strategies
    strategies = [
        # Direct substring match
        lambda t, m: (m in t, 1.0, m),
        # Reverse substring match
        lambda t, m: (t in m, 1.0, t),
        # Sequence matcher ratio
        lambda t, m: (True, difflib.SequenceMatcher(None, t, m).ratio(), m)
    ]

    best_ratio = 0.0
    best_match = None

    for strategy in strategies:
        is_match, ratio, matched = strategy(target_lower, match_lower)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = matched

    return best_ratio >= threshold, best_ratio, best_match


def create_visual_debug_overlay(image_path, boxes, lines=None, paragraphs=None, output_path=None):
    """Create a visual debugging overlay showing text boxes, lines, and paragraphs.

    Args:
        image_path: Path to the original screenshot
        boxes: List of text boxes with 'text' and 'box' fields
        lines: Optional list of line groupings
        paragraphs: Optional list of paragraph groupings
        output_path: Optional path to save the visualization

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
        # Add text label
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        draw.text((x, y - 15), box['text'], fill='gray', font=font)

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

            # Add center point for prompts (blue) and responses (red)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            if para.get('is_prompt', False):
                draw.ellipse([center_x-10, center_y-10, center_x +
                             10, center_y+10], outline='blue', width=2)
            elif para.get('is_response', False):
                draw.ellipse([center_x-10, center_y-10, center_x +
                             10, center_y+10], outline='red', width=2)

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
        dict: Contains the prompt-response pair and generated joke:
        {
            "prompt": str,      # The prompt text being responded to
            "response": str,    # The user's response to the prompt
            "joke": str        # The generated joke
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
    system_prompt = f"""You are a witty and observant dating app assistant. Your task is to analyze Hinge dating profiles and generate clever jokes that continue the conversation based on the woman's existing response. Each joke should balance three key elements equally:
1. Flattery - making the woman feel appreciated and seen
2. Humor - creating something genuinely funny and engaging
3. Flirtation - adding a playful, romantic element

You have access to the following information:
PROFILE STRUCTURE INFORMATION:
{format_content}

AVAILABLE PROMPTS:
{prompts_content}

AVAILABLE PHOTO CAPTIONS:
{captions_content}

AVAILABLE POLLS:
{polls_content}

Your process has two distinct steps:

STEP A: UNDERSTAND THE PROFILE
1. Analyze the profile structure using the provided format information in PROFILE STRUCTURE INFORMATION
2. Identify all profile elements:
   - Photos and their captions (see AVAILABLE PHOTO CAPTIONS)
   - Prompts and responses (see AVAILABLE PROMPTS)
   - Profile Information (see PROFILE STRUCTURE INFORMATION)
3. Build a comprehensive understanding of the woman's personality through:
   a) Activities and Interests
      - What does she enjoy doing?
      - How does she spend her time?
      - What are her hobbies?
   
   b) Intellectual Framework
      - How does she think?
      - What opinions does she express?
      - What values does she show?
   
   c) Communication Style
      - How does she express herself?
      - What tone does she use?
      - What kind of humor does she show?

STEP B: GENERATE THE JOKE
1. Identify all prompt-response pairs in the profile
   - Use the format structure in PROFILE STRUCTURE INFORMATION to recognize prompts
   - Match prompts with their corresponding responses
   - Verify against the available prompts list which is the authoritative source for prompts

2. For each prompt-response pair:
   a) Analyze the context:
      - What is she trying to convey with this prompt/response?
      - What personality traits does it reveal?
      - What values or interests does it highlight?
   
   b) Generate potential jokes that balance:
      * Flattery: Show we've paid attention and highlight positive traits
      * Humor: Create something genuinely funny and unexpected
      * Flirtation: Add a playful, romantic element
      * Keep jokes concise (1-2 sentences)
   
   c) Evaluate each joke by considering:
      - How well it balances all three elements
      - How natural and authentic it feels
      - How it might make her feel
      - How it could lead to further conversation

3. For each joke, simulate the conversation:
   - How would she feel receiving this joke?
   - What elements (flattery, humor, flirtation) would stand out to her?
   - How could this lead to an engaging back-and-forth?
   - What might she say in response?

4. Select the joke that:
   - Feels most natural and authentic
   - Would make her want to continue the conversation
   - Has the most promising simulated conversation flow

Your response must be in this JSON format:
{{
    "prompt": "The exact prompt text you're responding to",
    "response": "The user's response to this prompt",
    "joke": "Your generated joke to continue the conversation"
}}"""

    # User message - just the specific task
    user_message = """Please analyze these profile screenshots and generate a joke that continues the conversation based on the woman's existing response. Remember to:
1. First build your understanding of the woman's personality
2. Then identify prompt-response pairs and generate jokes that balance flattery, humor, and flirtation
3. Select the best joke based on simulated conversation flow"""

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
                "joke": result.get("joke", "")
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
