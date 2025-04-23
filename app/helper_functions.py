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
import shutil
from datetime import datetime
import logging
import traceback
import sys

load_dotenv()
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Generate a unique timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set up logging to file and console


def setup_logging():
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a desktop logs directory
    desktop_logs_dir = f"/app/desktop/PitchPerfect_Results_{RUN_TIMESTAMP}/logs"
    if not os.path.exists(desktop_logs_dir):
        os.makedirs(desktop_logs_dir)

    # Configure logging to write to both file and console
    desktop_log_file = os.path.join(
        desktop_logs_dir, f"pitchperfect_{RUN_TIMESTAMP}.log")

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler for desktop logs directory
    desktop_file_handler = logging.FileHandler(desktop_log_file)
    desktop_file_handler.setLevel(logging.DEBUG)
    desktop_file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(desktop_file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {desktop_log_file}")

    return logger


# Initialize logger
logger = setup_logging()

# Custom exception handler to log uncaught exceptions


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default handler for KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception
    logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))
    logger.error(f"Exception type: {exc_type.__name__}")
    logger.error(f"Exception value: {exc_value}")
    logger.error("Traceback:")
    for line in traceback.format_tb(exc_traceback):
        logger.error(line.rstrip())

    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Set the exception handler
sys.excepthook = log_uncaught_exceptions


def connect_device_remote(user_ip_address="127.0.0.1"):
    """Connect to a device remotely from docker container"""
    try:
        adb = AdbClient(host="host.docker.internal", port=5037)
        connection_result = adb.remote_connect(user_ip_address, 5555)
        logger.info(f"Connection result: {connection_result}")
        devices = adb.devices()

        if len(devices) == 0:
            logger.error("No devices connected")
            return None
        device = devices[0]
        logger.info(f"Connected to {device.serial}")
        return device
    except Exception as e:
        logger.error(f"Error connecting to device: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return None


def capture_screenshot(device, filename):
    """Capture a screenshot and save it to the images directory"""
    try:
        result = device.screencap()
        if not os.path.exists("images"):
            os.makedirs("images")
        with open("images/" + str(filename) + ".png", "wb") as fp:
            fp.write(result)
        logger.debug(f"Screenshot captured: images/{filename}.png")
        return "images/" + str(filename) + ".png"
    except Exception as e:
        logger.error(f"Error capturing screenshot '{filename}': {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return None


def tap(device, x, y, double_tap=False):
    """Execute a tap or double tap at the given coordinates"""
    try:
        logger.info(
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
    except Exception as e:
        logger.error(f"Error executing tap at ({x}, {y}): {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return False


def input_text(device, text):
    """Input text into the device"""
    try:
        text = text.replace(" ", "%s")
        logger.debug(f"Text to be written: {text}")
        device.shell(f'input text "{text}"')
    except Exception as e:
        logger.error(f"Error inputting text: {e}")
        logger.debug("", exc_info=True)


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
        logger.error(f"Error during swipe: {e}")
        logger.debug("", exc_info=True)
        return False, 0


def get_screen_resolution(device):
    """Get the screen resolution of the device"""
    try:
        output = device.shell("wm size")
        logger.debug(f"Screen size: {output}")
        resolution = output.strip().split(":")[1].strip()
        width, height = map(int, resolution.split("x"))
        return width, height
    except Exception as e:
        logger.error(f"Error getting screen resolution: {e}")
        logger.debug("", exc_info=True)
        return 1080, 1920  # Default fallback resolution


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
        logger.error(f"Error extracting text from image: {e}")
        logger.debug("", exc_info=True)
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

            vertical_overlap = (
                min(box_bottom, current_bottom) - max(box_top, current_top))

            if vertical_overlap > 0 or abs(box_y - current_y) <= y_threshold:
                # Box is aligned with current line
                current_line.append(box)
                # Update current line bounds
                current_y = min(current_y, box_y)
                current_height = max(
                    current_height, box_y + box_height - current_y)
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
    try:
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
                safe_text = box['text'].encode(
                    'ascii', 'replace').decode('ascii')
                draw.text((x, y - 15), safe_text, fill='gray', font=font)
            except Exception as e:
                logger.warning(f"Could not draw text '{box['text']}': {e}")

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
                max_x = max(box['box'][0] + box['box'][2]
                            for box in para['boxes'])
                max_y = max(box['box'][1] + box['box'][3]
                            for box in para['boxes'])
                draw.rectangle([min_x, min_y, max_x, max_y],
                               outline='green', width=2)

        # Draw tap target circle if provided
        if tap_target:
            tap_x, tap_y = tap_target
            radius = 30
            draw.ellipse([tap_x - radius, tap_y - radius, tap_x +
                         radius, tap_y + radius], outline='red', width=3)

        # Save the visualization
        img.save(output_path)
        return img
    except Exception as e:
        logger.error(f"Error creating visual debug overlay: {e}")
        logger.debug("", exc_info=True)
        return None


def generate_joke_from_screenshots(screenshots, format_txt_path, prompts_txt_path, captions_txt_path, polls_txt_path, locations_txt_path):
    """
    Generate a contextually appropriate joke response for a Hinge profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to format.txt describing profile structure
        prompts_txt_path: Path to prompts.txt containing available prompts
        captions_txt_path: Path to captions.txt containing possible captions
        polls_txt_path: Path to polls.txt containing available polls
        locations_txt_path: Path to locations.txt containing Madison, WI establishments

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
        with open(locations_txt_path, 'r') as f:
            locations_content = f.read()
    except Exception as e:
        logger.error(f"Error reading context files: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
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
            logger.error(
                f"Error processing screenshot {i} ({screenshot_path}): {e}")
            # Log full traceback at debug level
            logger.debug("", exc_info=True)
            return None

    # System prompt containing all the structural information
    system_prompt = f"""You are a witty and natural conversationalist on a dating app. Your task is to analyze Hinge profiles and generate engaging conversation starters based on the prompts and responses in a woman's profile. Since she's already expressed interest by looking at your profile, balance natural conversation with clear intent - keeping it light while being specific enough for text.

PROFILE STRUCTURE:
You will receive exactly 7 screenshots of a Hinge profile in order (index 0 to 6). Each profile will contain the following guaranteed elements:
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
3. Examine each screenshot in order (0 to 6) and identify all elements
4. Consult locations.txt to understand the types of establishments that exist in Madison, Wisconsin

STEP 2: IDENTIFY THE MAIN PERSON
1. Analyze all photos to identify the main person whose profile this is:
   - If a photo has just one person, that is likely the profile owner
   - If photos have multiple people, identify who appears most consistently across photos
   - Photos with no people are interest photos and should be ignored for identity
   - The main person should appear in the majority of photos
   - Note any distinguishing features of the main person for later analysis
2. Remember this person's identity throughout the rest of the analysis

STEP 3: ORGANIZE ELEMENTS INTO BUCKETS
For each screenshot, group elements into these buckets:
a. Photos (6 total)
   - May have a caption
   - If captioned, understand she chose that caption to describe the photo
   - Focus on photos containing the main person for her characteristics
   - Use photos with other people to understand her social circle
   - Use photos without people to understand her interests/hobbies
b. Prompts and Responses (3 total)
   - Understand she chose the prompt and provided her own response
   - These directly reflect her personality and preferences
   - CRITICAL: Verify these are actual prompts from prompts.txt and not captions
   - A prompt will NEVER have a photo or image as the response
   - If you see a response that describes a photo (e.g., "[Photo showing person posing with friend]"), this is a caption, not a prompt
c. Profile Basics (1 total)
   - All bullet-points she provided to describe herself
   - These are her self-reported characteristics
d. Voice Prompt and/or Poll Prompt (0-2 total)
   - Voice prompt: Cannot understand the recording, only see the prompt
   - Poll prompt: Can read her provided options for engagement

STEP 4: CHECK FOR UNDESIRABLE TRAITS
Analyze ONLY the main person (not other people in photos) for these traits:

1. Body Type Analysis (CRITICAL - Mark as undesirable if ANY of these are true):
   a. Body Shape Indicators:
      - Body width appears wider than typical fit/skinny woman
      - Visible stomach protrusion in any photo
      - Face shape is rounder with less defined jawline
      - Full-body photos show larger body size than average fit/skinny woman
      - Arms/legs show significant fat accumulation compared to wrists/ankles
      - No visible muscle or bone definition in arms/legs
      - Only face/high angle photos (hiding body)
   
   b. Specific Measurements (if visible):
      - Waist-to-hip ratio appears greater than 0.8
      - Arms appear wider than wrists by more than 2x
      - Legs appear wider than ankles by more than 2x
      - Face width appears greater than 1.5x face height
   
   c. Photo Analysis Guidelines:
      - If ANY full-body photo shows these traits, mark as undesirable
      - If only face photos, assume undesirable unless face shows very defined jawline
      - If photos are at angles that hide body shape, assume undesirable
      - If wearing baggy clothes in all photos, assume undesirable
      - If any photo shows clear stomach protrusion, mark as undesirable
      - If arms/legs show no muscle definition, mark as undesirable

2. Low Quality Photos:
   - Any ONE of these qualifies as low quality photos:
     * Majority of photos are blurry/pixelated
     * Heavy filters or editing in majority of photos
     * Screenshots or reposts in multiple photos

3. Male Features:
   - Any of these qualifies as male features:
     * Adam's apple visible in the throat
     * Facial hair (mustache or beard)
     * Flat chest with no boob growth visible

If ANY of these trait categories are met, return an empty response:
{{
    "prompt": "",
    "response": "",
    "conversation_starter": "",
    "screenshot_index": -1
}}

Otherwise, continue with the following steps.

STEP 5: ANALYZE EACH STORY'S IMPLICATIONS
For each story (element), determine what it reveals about the main person:
1. Photos of the main person: Consider what she's doing and how she made it happen
   - Example: "Playing uno in forest" → brought cards to wilderness
2. Photos with others: Consider why she is with other people in this photo. 
   They are probably people she likes. She might be engaging in an activity 
   that is more fun with others, or perhaps they are all doing something new 
   which is done easiest as a group.
3. Photos without people: Consider what this reveals about her interests. 
   There will be things like photos of food or nature and we can easily 
   interpret this as her interests.
4. Prompt/Response: Consider her history and preferences
   - Example: "Bar trivia and salty snacks" → enjoys trivia as hobby
   - CRITICAL: Always analyze the implicit meaning behind responses:
     * Responses often contain unstated but obvious information
     * Never ask redundant questions about things they've already stated
     * Look for the deeper context in their answers
     * Example 1: "My simple pleasures - the spicy potato soft taco at taco bell"
       - Implicit: They've tried it and enjoy it
       - Don't ask: "Have you tried the spicy potato soft taco?"
       - Instead ask: "What's your go-to order at Taco Bell?"
     * Example 2: "I geek out on - astrophysics"
       - Implicit: They have knowledge/interest in the topic
       - Don't ask: "Do you know about astrophysics?"
       - Instead ask: "What's the most fascinating thing you've learned about astrophysics?"
5. Profile Basics: Take her descriptions at face value
6. Voice Prompt: Unknown content, only prompt visible
7. Poll Prompt: Shows topics she'd discuss

STEP 6: COMPILE CHARACTERISTICS
1. List all characteristics derived from stories about the main person
2. Organize by confidence level. Also determine what her characteristics are based 
   on all of the analysis of the stories we just did.
3. Note when multiple stories support same characteristic
   - Example: Two stories about exercise → high confidence in active lifestyle
4. Read this list out loud to yourself. Does this sound like a reasonable person? 
   Do they have hobbies that seem realistic? If there are characteristics of her 
   that seem vague or exaggerated, be critical of them. Such as, "she's an adventurous 
   person" - it's hard to prove this characteristic because it is vague. It's also hard 
   to prove because it's exaggerated. Imagine you are going to represent this list of 
   characteristics to the woman. Would she be flattered? Shocked? Disinterested? Feel 
   awkward? Focus on things that are highly likely to be true as a few good truths are 
   better than many assumptions.

STEP 7: GENERATE CONVERSATION STARTERS
For each prompt/response pair:
1. Use characteristics and stories as context
2. Create exactly THREE unique conversation starters that follow this three-step approach:
   a. First, give a direct, simple acknowledgment that shows you've understood something visible from her profile
      - Focus on concrete, observable things:
        * Physical appearance (only if obvious and tasteful)
          - "You have a great smile" (when smiling in photos)
          - "That's a cool jacket" (when wearing something distinctive)
        * Activities she's doing in photos
          - "You look comfortable on that surfboard" (when surfing)
          - "You seem at home in that kitchen" (when cooking)
        * Places she's shown in photos
          - "That beach looks amazing" (when at beach)
          - "That coffee shop looks cozy" (when in cafe)
        * Specific items or objects in photos
          - "That's a nice camera" (when holding camera)
          - "Cool hiking boots" (when wearing hiking gear)
      - AVOID vague or personality-based compliments like:
        * "You look like you know how to have fun"
        * "You seem like a fun person"
        * "You look adventurous"
        * "You seem really cool"
        * "You look like you have a great personality"
      - Keep it straightforward and obvious
      - Make it specific to what you see in her profile
      - The compliment should be something you can point to in a photo or prompt response
   b. Next, provide a brief connection (5-7 words) between her prompt/response and your suggested date activity
      - Explain why you're suggesting the particular activity
      - Reference something specific from her prompt or response
      - Keep it concise but clear why this activity relates to what she shared
      - This creates context for your suggestion and shows you paid attention
   c. Finally, suggest a shared activity or experience related to the prompt
      - Express that you'd like to share in an activity she mentioned
      - Make it clear you want to participate with her, not just discuss the topic
      - Suggest something fun and specific that connects to her interests
      - CRITICAL: When suggesting a specific location in Madison, Wisconsin:
         * First check the locations.txt file to see if the type of establishment exists in Madison
         * If suggesting a specific named establishment (e.g., "Barriques", "Eno Vino"), ONLY use establishments listed in locations.txt
         * If the specific type of establishment isn't in locations.txt, just use the generic type (e.g., "a cafe downtown" instead of naming one)
         * NEVER suggest a specific named establishment that isn't in locations.txt
         * It's always safe to suggest general areas like "downtown", "near the Capitol", or "by the lake" without naming a specific establishment
      - CRITICAL: The suggested activity MUST be suitable for a date context:
         * Takes between 30-90 minutes (half hour to hour and a half)
         * Fun, casual and low-stakes (avoid high-pressure or complex activities)
         * No larger goals that overshadow the natural connection (avoid activities focused on "mastering" or "perfecting" something)
         * Should allow conversation to flow naturally while doing something enjoyable
         * Examples of good date activities: coffee shop visit, wine tasting, casual walk, museum visit, dessert spot, casual game
         * Examples of poor date activities: full sporting events, multi-day projects, competitive activities, professional networking
      * Each of the THREE starters should differ in either:
        - The aspect/item being acknowledged (e.g., different visible elements)
        - OR a different shared activity suggestion for the same topic
      * Aim to create inviting scenarios that feel natural, fun and time-limited

   * Uses natural, everyday language - how you'd actually talk to someone
   * Avoids flowery metaphors or dramatic language
   * Never uses marketing-style words like "perfect", "amazing", "journey", "adventure"
   * Shows interest through specificity and shared experiences
   * Aims for brevity - 8-10 words for part one (acknowledgment), 5-7 words for part two (connection), and 8-10 words for part three (activity suggestion)
   * Easy to respond to over text
   * Creates a pleasant shared experience when possible

CRITICAL: People don't narrate their lives with dramatic language - they just express interest in things they want to do together.

CRITICAL: Your conversation starters must be rooted in reality and suggest real shared activities based on her interests.

CRITICAL: Shorter messages are significantly more understandable over text. Keep messages brief, direct, and immediately clear - avoid any unnecessary words.

Examples of good vs bad conversation starters:

GOOD:
- Prompt: "My Love Language is"
  Response: "Words of affirmation and cheese"
  Conversation Starter: "Cute beanie in your hiking photo. Cheese lovers unite for snacking. Wine and cheese at Eno Vino?" 
  (Simple acknowledgment, connection to her interest, date-appropriate shared activity)
- Prompt: "I geek out on"
  Response: "NBA, specifically golden state warriors" 
  Conversation Starter: "Great smile in your concert photo. Warriors fans need sports bars. Warriors game at Coopers Tavern?" 
  (Specific acknowledgment, connection explaining why bar, casual viewing activity)
- Prompt: "Two truths and a lie..."
  Response: "1) I have a scar on my hand from being bitten by a hamster 2) I wrote and directed a short film 
          3) I won a county-wide math competition in school"
  Conversation Starter: "Nice kayak in your lake photo. Filmmakers deserve coffee breaks. Tell me about your film at Barriques?" 
  (Visible object acknowledgment, connection to her filmmaking, casual date activity)
- Prompt: "I take pride in"
  Response: "my ability to make friends with anyone, anywhere. I can strike up a conversation with a wall."
  Conversation Starter: "Love your colorful scarf. Social skills shine in cafes. Test your conversation skills at Capitol cafes?" 
  (Specific compliment, connection to her social ability, fun casual activity)
- Prompt: "What I order for the table"
  Response: "Carbs, cheese and wine"
  Conversation Starter: "Cool sunglasses in your beach photo. Wine pairs with good company. Cheese board at downtown wine bar?" 
  (Specific visible element, connection to wine interest, specific time-limited activity)
        - More Good Examples:
        * Prompt: "My simple pleasures"
          Response: "Carbs, cheese, and wine"
          Conversation Starter: "Great earrings in your dinner photo. Carbs and wine need proper venue. Let's enjoy a charcuterie board downtown."
        * Prompt: "The dorkiest thing about me is"
          Response: "I'm currently studying all of world history to pass a teaching exam, help me out!"
          Conversation Starter: "Love your smile in profile pic. History buffs need study breaks. Let's quiz each other over coffee at Barriques."  
        * Prompt: "I geek out on"
          Response: "Recreating dishes from my favorite restaurants"
          Conversation Starter: "Cool hiking boots in mountain photo. Food recreation deserves audience. Would love to try your dishes sometime."

BAD:
- Prompt: "A random fact I love is"
  Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight. And twilight is a fanfic of 
          My Chemical Romance, and if 9/11 didn't happen, they wouldn't exist. It's weird I know"
  Conversation Starter: "That's a wild fanfic connection! Pop culture creates unexpected links. What other rabbit holes do you enjoy falling into?" 
  (Just asks a question instead of suggesting a shared activity)
- Prompt: "The quickest way to my heart is"
  Response: "Talking over ice cream/coffee"
  Conversation Starter: "That coffee shop looks cozy. Conversations need the right ambiance. What's your go-to coffee order for good talks?" 
  (Asks about preferences rather than suggesting "Let's meet at that cozy coffee shop in your photo")
- Prompt: "My biggest date fail"
  Response: "Going swimming in Lake Mendota and then cutting my foot on a piece of metal... 
          the date ended because I had to go to Urgent Care for a tetanus shot 😜"
  Conversation Starter: "That's a memorable date story. Lake accidents happen to everyone. Do tetanus shots make your top 5 worst dates?" 
  (Asks a question instead of suggesting "Let's try kayaking on Mendota without the injury")
- Prompt: "My best celebrity impression is..."
  Response: "\"Why hello there,\" in my best Obi-Wan Kenobi voice and my friends hate it."
  Conversation Starter: "That's a solid Obi-Wan reference. Star Wars fans understand the appeal. Do you have other impressions in your arsenal?" 
  (Asks for more content instead of suggesting "Let's watch Star Wars and trade our best impressions")
- Prompt: "The dorkiest thing about me is"
  Response: "I'm currently studying all of world history to pass a teaching exam, help me out!"
  BAD Conversation Starter: "Your dedication is impressive. History exams are challenging tasks. How long have you been studying for it?" 
  (Just asks a question without offering to help)

STEP 8: SIMULATE CONVERSATION
For each of the NINE starters (3 per prompt/response pair):
1. Conduct a complete simulated conversation:
   a. Message 1: Her prompt/response as the first message
   b. Message 2: Your conversation starter as the response
   c. Message 3: Imagine her most likely reply based on her profile characteristics
   d. Message 4: Your natural follow-up to continue the conversation
   e. Message 5: Her second response
2. Evaluate based on these criteria:
   - How natural does this conversation flow?
   - Would this conversation be enjoyable for both parties?
   - Does it offer a clear shared activity or experience?
   - Does it avoid sounding like marketing copy?
   - Is it free of unnecessary qualifiers and drama?
   - Is it easy for her to respond without requiring complex thinking?
   - Does it show genuine interest in shared experiences?
   - Does it respect her intelligence without being pretentious?
3. Rank each conversation on a scale of 1-10 for overall quality
4. Note which conversation feels most natural and has the best flow

STEP 9: SELECT BEST STARTER
1. Choose the starter that best:
   - Uses the most natural, everyday language
   - Gets straight to the point
   - Shows interest through specificity, not enthusiasm
   - Creates easy conversation flow
   - Respects the complexity of her personality
   - Creates positive imagery or stories in her mind
   - Requires minimal mental effort to respond
   - Has the highest-ranked simulated conversation
   - CRITICAL: If a starter suggests a specific named establishment (e.g., "Barriques", "Eno Vino"), verify it exists in locations.txt
   - IMMEDIATELY DISQUALIFY any starter that mentions a specific named establishment not found in locations.txt
2. Note which prompt/response pair of the woman's profile matches the chosen starter
3. Reference prompts.txt and separate the prompt/response pair into the prompt and the response. Take note of the prompt distinctly from the response.

STEP 10: IDENTIFY SCREENSHOT
1. Note which screenshot (0-6) contains the prompt/response pair text of the woman's profile that matches the chosen starter
2. If prompt/response pair is cut off or spans multiple screenshots, note which screenshot contains the majority of the prompt/response pair text

Return the chosen prompt, response, your conversation starter, and the screenshot index in this JSON format exactly. Do not return any other text or comments beyond the JSON.
{{
    "prompt": "The exact prompt text the woman chose",
    "response": "The woman's response to the prompt",
    "conversation_starter": "Your natural conversation starter",
    "screenshot_index": index_of_screenshot_containing_prompt_response  # 0-based index (0-6)
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
        logger.info("Making API call to Claude...")
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=1000,
            temperature=1,
            system=system_prompt,
            messages=messages
        )
        logger.info("Claude API call successful")

        # Parse the response
        try:
            # The response should be in JSON format as requested
            import json

            # Extract just the JSON portion from the response
            response_text = response.content[0].text
            logger.debug(f"Raw response from Claude: {response_text[:500]}...")

            # Find the JSON part by locating the first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                result = json.loads(json_text)
                logger.info("Successfully parsed Claude response")
                return {
                    "prompt": result.get("prompt", ""),
                    "response": result.get("response", ""),
                    "conversation_starter": result.get("conversation_starter", ""),
                    "screenshot_index": result.get("screenshot_index", 0)
                }
            else:
                logger.error("No valid JSON object found in response")
                logger.error(f"Raw response: {response_text}")
                return None
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            logger.error(f"Raw response: {response.content[0].text[:1000]}...")
            # Log full traceback at debug level
            logger.debug("", exc_info=True)
            return None

    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
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
    try:
        # Take a fresh screenshot at this position
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_prompt_detection")

        # Extract text and group into paragraphs
        boxes = extract_text_from_image_with_boxes(screenshot_path)
        if not boxes:
            logger.warning("No text boxes found in screenshot")
            return False, None

        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # First try to find the target prompt with high confidence
        logger.debug("\nComparing target prompt against OCR paragraphs:")
        logger.debug(f"Target prompt: '{target_prompt}'")
        logger.debug("\nOCR paragraphs found:")

        best_prompt_match = None
        best_prompt_ratio = 0.0
        best_response_match = None
        best_response_ratio = 0.0

        for i, para in enumerate(paragraphs):
            # Check for prompt match
            is_prompt_match, prompt_ratio, matched_text = fuzzy_match_text(
                target_prompt, para['text'], threshold=0.8)
            logger.debug(f"Paragraph {i+1}:")
            logger.debug(f"  Text: '{para['text']}'")
            logger.debug(f"  Prompt match ratio: {prompt_ratio:.2f}")

            # Also check for response match with lower threshold
            is_response_match, response_ratio, _ = fuzzy_match_text(
                target_response, para['text'], threshold=0.7)
            logger.debug(f"  Response match ratio: {response_ratio:.2f}")
            logger.debug("")

            if is_prompt_match and prompt_ratio > best_prompt_ratio:
                best_prompt_match = para
                best_prompt_ratio = prompt_ratio
            elif is_response_match and response_ratio > best_response_ratio:
                best_response_match = para
                best_response_ratio = response_ratio

        # Use prompt match if found, otherwise use response match
        best_match = best_prompt_match if best_prompt_match else best_response_match
        if best_match:
            match_type = 'prompt' if best_prompt_match else 'response'
            match_ratio = max(best_prompt_ratio, best_response_ratio)
            logger.info(
                f"Found {match_type} match with ratio {match_ratio:.2f}")

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
            # Increased from 1.0 to 2.0 seconds to wait for response interface to open
            time.sleep(2.0)

            return True, (tap_x, tap_y)
        else:
            logger.warning(
                "No matching prompt or response found in screenshot")
            # Create visualization without tap target
            create_visual_debug_overlay(
                screenshot_path,
                boxes=boxes,
                lines=lines,
                paragraphs=paragraphs,
                output_path=f"images/profile_{profile_num}_prompt_detection_visual.png"
            )

            # Fallback: Scroll to bottom and double-click center
            logger.info(
                "\nFallback: Scrolling to bottom and double-clicking center...")

            # Calculate remaining scrolls (we've already done screenshot_index scrolls)
            # 6 is max scrolls (7 screenshots total, 0-6)
            remaining_scrolls = 6 - screenshot_index

            # Scroll the remaining distance to bottom
            for i in range(remaining_scrolls):
                logger.info(f"Fallback scroll #{i+1}")
                swipe(device, "down")
                time.sleep(1)  # Wait for scroll to complete

            # Get screen dimensions for center tap
            width, height = get_screen_resolution(device)
            center_x = width // 2
            center_y = height // 2

            logger.info(
                f"Double-clicking center of screen at ({center_x}, {center_y})")
            tap(device, center_x, center_y, double_tap=True)
            time.sleep(1)  # Wait for response interface to open

            # Return True since we executed the fallback
            return True, (center_x, center_y)

    except Exception as e:
        logger.error(f"Error in detect_prompt_in_screenshot: {e}")
        logger.debug("", exc_info=True)
        return False, None


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
    logger.info("\nPhase 1: Locating comment box...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase1")

    # Extract text and boxes
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        logger.warning("No text boxes found in initial screenshot")
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
        logger.warning("Could not find comment box")
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
    min_x = min(box['box'][0] for box in comment_boxes)
    max_x = max(box['box'][0] + box['box'][2] for box in comment_boxes)
    min_y = min(box['box'][1] for box in comment_boxes)
    max_y = max(box['box'][1] + box['box'][3] for box in comment_boxes)

    comment_x = (min_x + max_x) // 2
    comment_y = (min_y + max_y) // 2

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
    time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds

    # Clear any existing text in the comment box by sending 5 backspace keys
    for _ in range(5):
        device.shell("input keyevent KEYCODE_DEL")
    time.sleep(0.5)  # Wait after clearing text

    # Now input the conversation starter
    input_text(device, conversation_starter)
    time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds

    # Close keyboard
    device.shell('input keyevent 4')  # KEYCODE_BACK
    time.sleep(2.0)  # Increased from 1.0 to 2.0 seconds

    # PHASE 2: Find and click Send Priority Like button in new layout
    logger.info("\nPhase 2: Locating Send Priority Like button...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase2")

    # Extract text and boxes again for new layout
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        logger.warning("No text boxes found in post-input screenshot")
        return False

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Find "Send Priority Like" button in new layout
    send_button = None
    send_ratio = 0.0
    for para in paragraphs:
        is_match, ratio, _ = fuzzy_match_text(
            "Send Priority Like", para['text'], threshold=0.7)
        if is_match and ratio > send_ratio:
            send_button = para
            send_ratio = ratio

    if not send_button:
        logger.warning("Could not find Send Priority Like button")
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
    min_x = min(box['box'][0] for box in send_boxes)
    max_x = max(box['box'][0] + box['box'][2] for box in send_boxes)
    min_y = min(box['box'][1] for box in send_boxes)
    max_y = max(box['box'][1] + box['box'][3] for box in send_boxes)

    send_x = (min_x + max_x) // 2
    send_y = (min_y + max_y) // 2

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


def save_profile_results(profile_num, screenshots, ai_response):
    """Save profile screenshots and AI response in an organized folder structure.

    Args:
        profile_num: The profile number
        screenshots: List of screenshot paths
        ai_response: The AI response dictionary (can be None)

    Returns:
        str: Path to the profile's results directory
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create desktop directory path with timestamp for easy access from host
    desktop_dir = f"/app/desktop/PitchPerfect_Results_{RUN_TIMESTAMP}"
    if not os.path.exists(desktop_dir):
        os.makedirs(desktop_dir)

    # Create profile-specific directory in both locations
    profile_dir = os.path.join(results_dir, f"profile_{profile_num}")
    desktop_profile_dir = os.path.join(desktop_dir, f"profile_{profile_num}")

    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    if not os.path.exists(desktop_profile_dir):
        os.makedirs(desktop_profile_dir)

    # Create screenshots subdirectory in both locations
    screenshots_dir = os.path.join(profile_dir, "screenshots")
    desktop_screenshots_dir = os.path.join(desktop_profile_dir, "screenshots")

    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    if not os.path.exists(desktop_screenshots_dir):
        os.makedirs(desktop_screenshots_dir)

    # Copy screenshots to both profile directories
    for screenshot in screenshots:
        filename = os.path.basename(screenshot)
        # Save to container results directory
        dest_path = os.path.join(screenshots_dir, filename)
        shutil.copy2(screenshot, dest_path)
        # Also save to desktop directory
        desktop_dest_path = os.path.join(desktop_screenshots_dir, filename)
        shutil.copy2(screenshot, desktop_dest_path)

    # Save AI response as JSON with timestamp, but only to desktop
    desktop_response_path = os.path.join(desktop_profile_dir, "response.json")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Handle None response by creating an empty response with timestamp
    response_data = ai_response if ai_response is not None else {}
    response_data['timestamp'] = timestamp

    # Save only to desktop directory
    with open(desktop_response_path, 'w') as f:
        json.dump(response_data, f, indent=2)

    logger.info(f"Screenshots saved to container path: {profile_dir}")
    logger.info(
        f"Results and screenshots saved to desktop path: {desktop_profile_dir}")

    return profile_dir


def check_for_end_of_profiles(device, profile_num):
    """Check if we've reached the end of available profiles.

    Args:
        device: The ADB device
        profile_num: Current profile number for debugging

    Returns:
        tuple: (bool, str) - (True if end reached, message that was matched)
    """
    # Take a screenshot to check for the message
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_end_check")

    # Extract text and group into paragraphs
    boxes = extract_text_from_image_with_boxes(screenshot_path)
    if not boxes:
        return False, ""

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Messages to check for
    end_messages = [
        "You've seen everyone for now",
        "Try changing your filters so more people match your criteria - or check again later!",
        "Change filters",
        "Review skipped profiles"
    ]

    # Check each paragraph against each message
    for para in paragraphs:
        for message in end_messages:
            is_match, ratio, _ = fuzzy_match_text(
                message, para['text'], threshold=0.8)
            if is_match:
                logger.info(
                    f"Found end message: '{message}' with confidence {ratio:.2f}")
                # Create visualization of the match
                create_visual_debug_overlay(
                    screenshot_path,
                    boxes=boxes,
                    lines=lines,
                    paragraphs=paragraphs,
                    output_path=f"images/profile_{profile_num}_end_detected_visual.png"
                )
                return True, message

    return False, ""
