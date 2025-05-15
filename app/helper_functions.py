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
import random
import re
import pytz

load_dotenv()
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Generate a unique timestamp for this run using Central Time zone
central_tz = pytz.timezone('US/Central')
RUN_TIMESTAMP = datetime.now(central_tz).strftime("%m-%d_%H-%M-%S")

# Custom log filter to truncate long base64 strings


class Base64TruncateFilter(logging.Filter):
    """Filter to truncate base64 strings in log messages to prevent log file bloat"""

    def __init__(self, max_length=200):
        self.max_length = max_length
        super().__init__()

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = self._truncate_base64(record.msg)
        if record.args:
            record.args = tuple(
                self._truncate_base64(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True

    def _truncate_base64(self, message):
        # For strings like "data:image/png;base64,iVBORw0K..." or "data: iVBORw0K..."
        if isinstance(message, str):
            # Base64 data URI pattern
            base64_pattern = r'(data:.*?base64,)([A-Za-z0-9+/=]{20,})'
            result = re.sub(
                base64_pattern,
                lambda m: f"{m.group(1)}{m.group(2)[:self.max_length]}...[truncated]",
                message
            )

            # Plain base64 pattern (not in a data URI)
            plain_base64_pattern = r'([A-Za-z0-9+/=]{100,})'
            result = re.sub(
                plain_base64_pattern,
                lambda m: f"{m.group(1)[:self.max_length]}...[truncated]",
                result
            )

            return result
        return message


class PNGStreamFilter(logging.Filter):
    """Filter to remove noisy PNG stream debug messages"""

    def filter(self, record):
        if record.levelno == logging.DEBUG:
            # Filter out messages about PNG chunks
            if isinstance(record.msg, str) and ("STREAM b'IDAT'" in record.msg or "STREAM b'IHDR'" in record.msg):
                return False
        return True


# Set up logging to file and console


def setup_logging(app_name=""):
    """Set up logging to file and console.

    Args:
        app_name: Optional app name (no longer affects log directory structure)

    Returns:
        Logger object
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create main desktop directory for the entire run
    desktop_dir_name = f"pp_{RUN_TIMESTAMP}"
    desktop_dir = f"/app/desktop/{desktop_dir_name}"

    if not os.path.exists(desktop_dir):
        os.makedirs(desktop_dir)

    # Single log file in the main directory
    log_file_name = f"pp_{RUN_TIMESTAMP}.log"
    desktop_log_file = os.path.join(desktop_dir, log_file_name)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

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

    # Add the base64 truncate filter to both handlers
    base64_filter = Base64TruncateFilter(max_length=100)
    desktop_file_handler.addFilter(base64_filter)
    console_handler.addFilter(base64_filter)

    # Add the PNG stream filter to both handlers
    png_filter = PNGStreamFilter()
    desktop_file_handler.addFilter(png_filter)
    console_handler.addFilter(png_filter)

    # Set specific loggers to a higher level
    # These are likely sources of the noisy PNG-related logs
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

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


def tap(device, x, y, double_tap=False, with_additional_swipe=True):
    """Execute a tap at the specified coordinates.

    Args:
        device: The ADB device
        x: X coordinate to tap
        y: Y coordinate to tap
        double_tap: Whether to execute a double tap (default: False)
        with_additional_swipe: Whether to perform additional swipe to ensure tap registers (default: True)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
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
            # Additional swipe to ensure tap registers (optional)
            if with_additional_swipe:
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


def extract_text_from_image_with_boxes(image_path, app_type=None):
    """Extract text and bounding boxes from an image using OCR

    Args:
        image_path: Path to the screenshot image
        app_type: Optional app type ('hinge' or 'bumble') to optimize OCR settings

    Returns:
        List of text boxes with 'text' and 'box' fields
    """
    try:
        image = Image.open(image_path)

        # Set OCR configuration based on app type
        config = '--oem 1 -l eng --dpi 429'

        # Run OCR with appropriate config
        ocr_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config=config)

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


def group_boxes_into_lines(boxes, y_threshold=15, max_horizontal_gap=300):
    """Group text boxes into lines based on vertical alignment.

    Args:
        boxes: List of text boxes with 'text' and 'box' fields
        y_threshold: Maximum vertical distance for boxes to be on same line
        max_horizontal_gap: Maximum horizontal distance between boxes to be considered same line

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

            # Check horizontal distance to closest box in current line
            if current_line:
                # Find the closest box horizontally
                box_left = box['box'][0]
                box_right = box['box'][0] + box['box'][2]

                min_distance = float('inf')
                for existing_box in current_line:
                    existing_left = existing_box['box'][0]
                    existing_right = existing_box['box'][0] + \
                        existing_box['box'][2]

                    # Check for horizontal overlap
                    if box_right >= existing_left and box_left <= existing_right:
                        # Boxes overlap horizontally
                        min_distance = 0
                        break

                    # Distance between boxes
                    distance = min(abs(box_left - existing_right),
                                   abs(existing_left - box_right))
                    min_distance = min(min_distance, distance)

                # Check if horizontal distance is within limits
                horizontal_ok = min_distance <= max_horizontal_gap
            else:
                horizontal_ok = True

            if (vertical_overlap > 0 or abs(box_y - current_y) <= y_threshold) and horizontal_ok:
                # Box is aligned with current line and within horizontal distance
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


def group_lines_into_paragraphs(lines, paragraph_spacing=30):
    """Group lines into paragraphs based on vertical spacing and horizontal overlap.

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
    last_line_left = None
    last_line_right = None

    for line in lines:
        # Get vertical bounds of this line
        line_top = min(box['box'][1] for box in line)
        line_bottom = max(box['box'][1] + box['box'][3] for box in line)

        # Get horizontal bounds of this line
        line_left = min(box['box'][0] for box in line)
        line_right = max(box['box'][0] + box['box'][2] for box in line)

        if last_line_bottom is None:
            # First line
            current_para['lines'].append(line)
            current_para['boxes'].extend(line)
            last_line_bottom = line_bottom
            last_line_left = line_left
            last_line_right = line_right
        else:
            # Check vertical spacing
            vertical_ok = (line_top - last_line_bottom <= paragraph_spacing)

            # Check horizontal overlap
            horizontal_overlap = (
                min(line_right, last_line_right) -
                max(line_left, last_line_left)
            )
            horizontal_ok = (horizontal_overlap > 0)

            if vertical_ok and horizontal_ok:
                # Line is close enough to previous line and has horizontal overlap - same paragraph
                current_para['lines'].append(line)
                current_para['boxes'].extend(line)
                last_line_bottom = line_bottom
                last_line_left = min(last_line_left, line_left)
                last_line_right = max(last_line_right, line_right)
            else:
                # Line is far from previous line or has no horizontal overlap - start new paragraph
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
                last_line_left = line_left
                last_line_right = line_right

    # Add the last paragraph
    if current_para['boxes']:
        current_para['text'] = ' '.join(box['text']
                                        for box in current_para['boxes'])
        paragraphs.append(current_para)

    return paragraphs


def fuzzy_match_text(target_text, text_to_match, threshold=0.7):
    """Perform fuzzy matching between two text strings.

    Args:
        target_text: The text we're looking for
        text_to_match: The text we're comparing against
        threshold: Minimum similarity ratio to consider a match (0.0 to 1.0), default is 0.7

    Returns:
        tuple: (is_match, similarity_ratio, matched_text)
    """
    # Convert to lowercase for case-insensitive matching
    target_lower = target_text.lower().strip()
    match_lower = text_to_match.lower().strip()

    # Only fire substring match if the target appears as a contiguous substring (with spaces preserved)
    if target_lower and target_lower in match_lower:
        return True, 0.85, text_to_match

    # Get base similarity ratio using sequence matcher
    ratio = difflib.SequenceMatcher(None, target_lower, match_lower).ratio()

    # Apply length penalty if strings are very different in length
    len_ratio = min(len(target_lower), len(match_lower)) / \
        max(len(target_lower), len(match_lower))
    ratio = ratio * len_ratio

    # Only consider it a match if the ratio is above threshold
    is_match = ratio >= threshold
    return is_match, ratio, text_to_match


def create_visual_debug_overlay(image_path, boxes, lines=None, paragraphs=None, output_path=None, tap_target=None, profile_num=None, app_name=None):
    """Create a visual debugging overlay showing text boxes, lines, and paragraphs.

    Args:
        image_path: Path to the original screenshot
        boxes: List of text boxes with 'text' and 'box' fields
        lines: Optional list of line groupings
        paragraphs: Optional list of paragraph groupings
        output_path: Optional path to save the visualization
        tap_target: Optional (x, y) coordinates for tap target
        profile_num: Optional profile number to organize outputs
        app_name: Optional app name for organizing outputs

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

        # Flag to determine if we should save the file
        should_save = False

        # If profile_num is provided, save to the analysis folder (desktop only)
        if profile_num is not None:
            try:
                # Determine if we need to save to app-specific folder
                desktop_dir_name = f"pp_{RUN_TIMESTAMP}"
                desktop_dir = f"/app/desktop/{desktop_dir_name}"

                if app_name:
                    desktop_dir = os.path.join(
                        desktop_dir, app_name.capitalize())

                # Get the profile's analysis directory
                desktop_profile_dir = os.path.join(
                    desktop_dir, f"profile_{profile_num}")
                analysis_dir = os.path.join(desktop_profile_dir, "analysis")

                # Create it if it doesn't exist
                if not os.path.exists(analysis_dir):
                    os.makedirs(analysis_dir)

                # Extract just the filename from the output_path
                filename = os.path.basename(output_path)
                # Update output path to save in analysis folder
                output_path = os.path.join(analysis_dir, filename)
                should_save = True
                logger.debug(
                    f"Will save debug visualization to desktop analysis folder: {output_path}")
            except Exception as e:
                logger.warning(
                    f"Could not save visualization to analysis folder: {e}")
                should_save = False

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
            radius = 20
            draw.ellipse([tap_x - radius, tap_y - radius, tap_x +
                         radius, tap_y + radius], outline='purple', width=3)

        # Save the visualization only if should_save is True
        if should_save:
            img.save(output_path)
            logger.debug(f"Saved debug visualization to: {output_path}")

        return img
    except Exception as e:
        logger.error(f"Error creating visual debug overlay: {e}")
        logger.debug("", exc_info=True)
        return None


def read_context_files(file_paths):
    """
    Read the content of multiple context files.

    Args:
        file_paths: Dict of {name: path} for each file to read

    Returns:
        Dict containing the content of each file with the same keys as file_paths,
        or None if there was an error reading any file
    """
    try:
        result = {}
        for name, path in file_paths.items():
            with open(path, 'r') as f:
                result[name] = f.read()
        return result
    except Exception as e:
        logger.error(f"Error reading context files: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return None


def prepare_screenshots_for_api(screenshots):
    """
    Convert screenshots to base64 for API transmission.

    Args:
        screenshots: List of paths to screenshot images

    Returns:
        List of dicts with base64-encoded image data, or None if error
    """
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
    return screenshot_data


def call_claude_api(system_prompt, user_message, screenshots_data, model="claude-3-7-sonnet-latest", max_tokens=1000, temperature=1):
    """
    Make an API call to Claude with retry logic.

    Args:
        system_prompt: System prompt for Claude
        user_message: Message to send to Claude
        screenshots_data: List of screenshot data objects to include
        model: Claude model to use
        max_tokens: Maximum tokens in response
        temperature: Temperature parameter for generation

    Returns:
        Response from Claude API or None if error
    """
    try:
        # Create the message for Claude
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    *screenshots_data  # Add all screenshots as subsequent content items
                ]
            }
        ]

        # Completely disable logging of API call details
        # Save current log levels
        root_logger = logging.getLogger()
        original_root_level = root_logger.level

        anthropic_logger = logging.getLogger('anthropic')
        original_anthropic_level = anthropic_logger.level if anthropic_logger.level else original_root_level

        urllib3_logger = logging.getLogger('urllib3')
        original_urllib3_level = urllib3_logger.level if urllib3_logger.level else original_root_level

        requests_logger = logging.getLogger('requests')
        original_requests_level = requests_logger.level if requests_logger.level else original_root_level

        # Temporarily increase log levels to suppress detailed logs
        root_logger.setLevel(logging.WARNING)
        if anthropic_logger:
            anthropic_logger.setLevel(logging.WARNING)
        if urllib3_logger:
            urllib3_logger.setLevel(logging.WARNING)
        if requests_logger:
            requests_logger.setLevel(logging.WARNING)

        # Implement retry with exponential backoff
        max_retries = 5
        retry_count = 0
        base_delay = 2  # Start with 2 seconds delay

        while retry_count <= max_retries:
            try:
                # Make the actual API call
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                )

                # If we get here, the API call was successful
                break

            except Exception as api_error:
                # Check if this is a retryable error
                should_retry = False
                error_message = str(api_error)

                # Check for specific error types that warrant retry
                if "529" in error_message or "overloaded_error" in error_message:
                    should_retry = True
                    logger.warning(f"Claude API overloaded (429). Retrying...")
                elif "503" in error_message or "502" in error_message:
                    should_retry = True
                    logger.warning(f"Claude API server error. Retrying...")
                elif "429" in error_message or "rate_limit" in error_message:
                    should_retry = True
                    logger.warning(
                        f"Claude API rate limit reached. Retrying...")

                # If no more retries or not a retryable error, raise the exception
                if retry_count >= max_retries or not should_retry:
                    raise

                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                logger.info(
                    f"Retrying Claude API call in {delay:.1f} seconds (attempt {retry_count+1}/{max_retries})")
                time.sleep(delay)
                retry_count += 1

        # Restore original log levels
        root_logger.setLevel(original_root_level)
        if anthropic_logger:
            anthropic_logger.setLevel(original_anthropic_level)
        if urllib3_logger:
            urllib3_logger.setLevel(original_urllib3_level)
        if requests_logger:
            requests_logger.setLevel(original_requests_level)

        logger.info("Claude API call successful")
        return response

    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        logger.error("Critical error in API call - stopping application")
        logger.debug("", exc_info=True)  # Log full traceback at debug level

        # Exit the application with non-zero status code
        logger.info("Exiting application due to Claude API error")
        sys.exit(1)  # Use exit code 1 to indicate an error occurred


def parse_claude_json_response(response):
    """
    Parse a JSON response from Claude.

    Args:
        response: Response object from Claude API

    Returns:
        Parsed JSON dict or None if error
    """
    try:
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
            return result
        else:
            logger.error("No valid JSON object found in response")
            logger.error(f"Raw response: {response_text}")
            return None
    except Exception as e:
        logger.error(f"Error parsing Claude response: {e}")
        logger.error(f"Raw response: {response.content[0].text[:1000]}...")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return None


# Shared prompt components
DATING_APP_INTRO = """You are a witty and natural conversationalist on a dating app. Your task is to analyze profiles and generate engaging conversation starters based on the prompts and responses in a woman's profile. Since she's already expressed interest by looking at your profile, balance natural conversation with clear intent - keeping it light while being specific enough for text."""

IDENTIFY_MAIN_PERSON = """STEP 2: IDENTIFY THE MAIN PERSON
1. Examine all photos to identify the main person (profile owner)
2. Look for recurring individuals across multiple photos
3. Pay attention to photo composition (center framing usually indicates main person)
4. If there are multiple people in group photos, identify common patterns:
   - Who appears most frequently?
   - Who is emphasized in composition?
   - Is anyone tagged or highlighted?
5. For photos without people, note what these reveal about interests
6. Focus analysis on the main person, not friends/family in photos"""

UNDESIRABLE_TRAITS_CHECK = """STEP 4: CHECK FOR UNDESIRABLE TRAITS
Analyze ONLY the main person (not other people in photos) for these traits:

1. Body Type Analysis (CRITICAL - Mark as undesirable if ANY of these are true):
   a. Body Shape Indicators:
      - Body width appears wider than typical fit/skinny woman
      - Visible stomach protrusion that extends beyond the waistline (bulging out)
      - Face shape is rounder with less defined jawline
      - Full-body photos show larger body size than average fit/skinny woman
      - Arms/legs show significant fat accumulation compared to wrists/ankles
      - No visible muscle or bone definition in arms/legs
      - Only face/high angle photos (hiding body)

   b. Specific Measurements (if visible):
      - Waist-to-hip ratio appears greater than 1.0
      - Arms appear wider than wrists by more than 2.2x
      - Legs appear wider than ankles by more than 2.2x
      - Face width appears greater than 1.5x face height

   c. Photo Analysis Guidelines:
      - If ANY full-body photo shows these traits, mark as undesirable
      - If only face photos, assume undesirable unless face shows very defined jawline
      - If all photos are at angles that hide body shape, assume undesirable
      - If wearing baggy clothes in all photos, assume undesirable
      - If any photo shows clear stomach protrusion that extends beyond the waistline (bulging out), mark as undesirable
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

4. Advertisement Profile:
   - Any ONE of these indicates the profile is actually an advertisement:
     * Multiple photos showing the same exact image/scene repeatedly
     * Text in the profile containing words like "Ad", "Advertisement", or "Sponsored"
     * Product promotion is evident in photos or bio text (showing specific products with branding)
     * Profile seems focused on promoting a service, product, or business rather than dating
     * Profile contains website URLs or social media handles to follow

If ANY of these trait categories are met, return the below response exactly, where prompt, response, and conversation_starter are empty strings, and screenshot_index is -1:
{
    "prompt": "",
    "response": "",
    "conversation_starter": "",
    "screenshot_index": -1
}

Otherwise, continue with the following steps."""

STORY_ANALYSIS_STEPS = """STEP 5: ANALYZE EACH STORY'S IMPLICATIONS
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
7. Poll Prompt: Shows topics she'd discuss"""

CONVERSATION_STARTER_GUIDE = """STEP 7: GENERATE CONVERSATION STARTERS
For each prompt/response pair:
1. Use characteristics and stories as context
2. Confirm that the prompt of the prompt/response pair is NOT a poll prompt:
    a. CRITICAL: Immediately check if the prompt appears in hingePolls.txt
    b. If the prompt is found in hingePolls.txt, it is a poll prompt and must be DISQUALIFIED for conversation starters
    c. Only generate conversation starters for the three actual prompt/response pairs where the prompt is NOT in hingePolls.txt
3. Generate three conversation starters per prompt/response pair (nine total) that follow this three-step approach:
    a. First, think of a date activity (eight to ten words) that is relevant to her prompt/response.
       - CRITICAL: While it's ideal to suggest activities you see her doing in photos, this isn't always possible
       - You'll often need to base date activities on prompt/response pairs when photos don't show suitable activities
       - When using information from her photos:
         * Choose activities you've seen her actually doing
         * Keep the suggestion simple and time-limited (30-90 minutes)
       - When using information from prompt/response pairs:
         * Focus on concrete preferences mentioned (foods, drinks, interests)
         * Avoid using personal stories, goals, or aspirations
       - CRITICAL: Always use this exact format for date suggestions:
         * Start with the VERB + at venue + with me/together + question mark
         * VERB must be the first word - no introductory phrases like "Want to" or "Would you like to"
         * Examples: "Play monopoly at Role Play Lounge with me?" or "Drink at The Old Fashioned together?"
         * Bad examples: "Do you want to play monopoly at Role Play Lounge?" or "Let's grab drinks downtown?"
         * Use specific, direct verbs instead of generic ones like "try" or "sample" (e.g., "Eat", "Drink", "Climb", "Play", "Browse", "Hike")
         * Must be a complete grammatical sentence
         * Must include a specific action verb
         * Must include the venue name
         * Must end with "with me" or "together" followed by a comma and her name, then a question mark
       - CRITICAL: The suggested activity MUST:
         * Take between 30-90 minutes (half hour to hour and a half)
         * Be fun, casual and low-stakes (avoid high-pressure or complex activities)
         * Have no larger goals that overshadow the natural connection
         * Allow conversation to flow naturally while doing something enjoyable
         * Prioritize interactive experiences where you can do something together rather than just talking
         * Be between eight to ten words in length
         * Only suggest bowling/ice skating/climbing/kayaking if she shows specific interest in these activities
         * Examples of poor date activities: full sporting events, multi-day projects, competitive activities, professional networking
    b. Second, create a bridge (6-9 words) between her prompt/response and the suggested date activity
       - CRITICAL: The bridge must create a logical connection that explains WHY this activity relates to her interests
       - CRITICAL: The bridge should articulate a genuine reason why someone with her interest would enjoy this activity
       - Reference something specific from her prompt or response
       - Keep it concise but make the connection crystal clear
       - This creates context for your suggestion and shows you paid attention
       - IMPORTANT: If your date activity suggestion is based on an activity seen in her photos (preferred approach), create a bridge that connects from the prompt/response to that activity
         * Example: If her prompt is about loving dogs but photos show her kayaking, you might say "Dog people love exploring nature together" to bridge to kayaking
         * The connection should feel natural even when connecting different topics
       CRITICAL: The bridge must be:
       1. Specific to her prompt/response
       2. Clear and obvious to understand
       3. Natural lead-in to the date activity
       4. Between 6-9 words exactly
       5. Shows you paid attention to her profile
       6. Explains WHY the activity is relevant to her interest
       7. Suggests a genuine benefit or logical relationship
    c. Third, add her name after a comma after the activity suggestion
       - This personalizes the message and shows you paid attention to her profile
       - For example: "... with me, Sarah?" or "... together, Jessica?"
       - Always include her name at the end preceded by a comma
       - Keep it natural and conversational
    d. Verify the location of the establishment is in Madison, Wisconsin
       - CRITICAL: When suggesting a specific location in Madison, Wisconsin:
          * First check the locations.txt file to see if the type of establishment exists in Madison
          * If suggesting a specific named establishment (e.g., "Barriques", "Eno Vino"), ONLY use establishments listed in locations.txt - READ THE ENTIRE LIST CAREFULLY
          * If the specific type of establishment isn't in locations.txt, just use the generic type (e.g., "a cafe downtown" instead of naming one)
          * NEVER suggest a specific named establishment that isn't in locations.txt
          * ONLY mention establishments that are EXACTLY as they appear in locations.txt - no similar names or variations
          * Double-check the spelling and exact name of any establishment you mention
          * Do not suggest places that have permanently closed - if you're not 100% certain a place is still open, use a generic location instead
          * It's always safe to suggest general areas like "downtown", "near the Capitol", or "by the lake" without naming a specific establishment
    e. For each of the three prompt/response pairs, generate THREE distinct date activity suggestions:
       * Each with a unique bridge connecting it to the prompt/response
       * Each with a distinct activity that would be enjoyable based on her profile
       * Each should use a different venue or location
       * Each should follow all guidelines above

    - The conversation starter tuples should follow these general guidelines:
        * Aim to create inviting scenarios that feel natural, fun and time-limited
        * Use natural, everyday language - how you'd actually talk to someone
        * Avoid flowery metaphors or dramatic language
        * Never use marketing-style words like "perfect", "amazing", "journey", "adventure"
        * Show interest through specificity and shared experiences
        * Aim for brevity - 6-9 words for the bridge, 8-10 words for the activity suggestion with name
        * Keep it easy to respond to over text
        * Create a pleasant shared experience
   
   This will result in 9 total conversation starter tuples (3 per prompt/response pair). Count the number of conversation starter tuples. Are there 9? If not, you need to go back and generate three conversation starters per prompt/response pair for a total of 9.

    Examples of good vs bad DATE SUGGESTION TUPLES:

    GOOD DATE SUGGESTION TUPLES:
    - Prompt: "My Love Language is"
      Response: "Words of affirmation and cheese"
      Photo context: Playing board games in one photo
      Good Tuple: "Cheese appreciation calls for proper tasting atmosphere. Eat the cheese board at Fromagination together, Emma?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects cheese interest to a tasting experience
      - Activity (verb-first): ✓ Specific action verb "Eat" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Natural flow from bridge to activity suggestion
      
      Alternative Good Tuple: "Board games pair perfectly with specialty cheeses. Play Catan at Roll Play Lounge, Emma?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects both cheese interest and board game activity
      - Activity (verb-first): ✓ Specific action verb "Play" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Incorporates both her interests in one cohesive suggestion

    - Prompt: "I geek out on"
      Response: "NBA, specifically golden state warriors" 
      Photo context: Rock climbing at a gym
      Good Tuple: "Sports fans understand physical challenges and competition. Climb some beginner routes at Boulders Climbing Gym with me, Jessica?"
      Analysis:
      - Bridge (6-9 words): ✓ Links sports interest to physical activity challenges
      - Activity (verb-first): ✓ Specific action verb "Climb" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Natural flow connecting interest to relevant activity
      
      Alternative Good Tuple: "Basketball enthusiasts appreciate analysis. Watch the Bucks game at Sconnie Bar together, Jessica?"
      Analysis:
      - Bridge (6-9 words): ✓ Links basketball interest to athletic appreciation
      - Activity (verb-first): ✓ Specific action verb "Watch" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Focuses directly on her stated basketball interest

    - Prompt: "Two truths and a lie..."
      Response: "1) I have a scar on my hand from being bitten by a hamster 2) I wrote and directed a short film 3) I won a county-wide math competition in school"
      Photo context: Kayaking on a lake
      Good Tuple: "Creative people appreciate nature's peaceful inspiration too. Paddle a tandem kayak along Lake Mendota together, Sophia?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects creative identity to nature appreciation
      - Activity (verb-first): ✓ Specific action verb "Paddle" with proper location
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Natural connection between creativity and peaceful outdoor activity
      
      Alternative Good Tuple: "Filmmaking requires observant eyes for detail. Explore film exhibits at Chazen Museum of Art with me, Sophia?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects to her filmmaking experience
      - Activity (verb-first): ✓ Specific action verb "Explore" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Focuses on her creative interest mentioned in her response

    - Prompt: "What I order for the table"
      Response: "Carbs, cheese and wine"
      Photo context: Drinking at a bar
      Good Tuple: "Wine enthusiasts deserve quality local venues for tasting. Drink a flight of reds at Eno Vino together, Mia?"
      Analysis:
      - Bridge (6-9 words): ✓ Links wine interest to quality tasting experience
      - Activity (verb-first): ✓ Specific action verb "Drink" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Direct connection between her preference and activity
      
      Alternative Good Tuple: "Cheese and wine create perfect flavor pairings. Sample charcuterie boards at Fromagination with me, Mia?"
      Analysis:
      - Bridge (6-9 words): ✓ Links cheese and wine interests together
      - Activity (verb-first): ✓ Specific action verb "Sample" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Focuses on multiple aspects of her food preferences

    - Prompt: "My simple pleasures"
      Response: "Carbs, cheese, and wine"
      Photo context: Playing board games
      Good Tuple: "Wine perfectly complements friendly competition nights. Play Clue at Gamer's Library with me, Emily?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects wine to game night in logical way
      - Activity (verb-first): ✓ Specific action verb "Play" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Logical flow between interests and activity
      
      Alternative Good Tuple: "Carbs fuel long evenings of strategic thinking. Eat wood-fired pizza at Salvatore's Tomato Pies together, Emily?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects carb interest to strategic activities
      - Activity (verb-first): ✓ Specific action verb "Eat" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Connects her food interest to a relevant dining experience

    - Prompt: "The dorkiest thing about me is"
      Response: "I'm currently studying all of world history to pass a teaching exam, help me out!"
      Photo context: Hiking on a trail
      Good Tuple: "History buffs need peaceful breaks from intensive studying. Hike at Picnic Point together, Ashley?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects history studying to need for breaks
      - Activity (verb-first): ✓ Specific action verb "Hike" with proper location
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Logical connection between mental activity and physical break
      
      Alternative Good Tuple: "History comes alive through museum exhibits. Explore the Wisconsin Historical Museum with me, Ashley?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects history interest to interactive learning
      - Activity (verb-first): ✓ Specific action verb "Explore" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Directly relevant to her current studies in an engaging way

    - Prompt: "The quickest way to my heart is"
      Response: "Making me laugh and ice cream"
      Photo context: At a park with friends
      Good Tuple: "Ice cream is great for the summer. Taste different flavors at Chocolate Shoppe on State with me, Zoe?"
      Analysis:
      - Bridge (6-9 words): ✓ Links ice cream interest to summer memory creation
      - Activity (verb-first): ✓ Specific action verb "Taste" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Direct connection to her mentioned interest
      
      Alternative Good Tuple: "Comedy shows guarantee laughter and good vibes. Watch standup at Comedy Club on State together, Zoe?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects to her interest in laughter
      - Activity (verb-first): ✓ Specific action verb "Watch" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Focuses on her other stated interest effectively

    - Prompt: "I'm weirdly attracted to"
      Response: "Guys who can beat me at Mario Kart"
      Photo context: At a party with friends
      Good Tuple: "Gaming gets better with friendly competition and stakes. Race Mario Kart at Double Tap Beercade together, Megan?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects gaming to enhanced competition with stakes
      - Activity (verb-first): ✓ Specific action verb "Race" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Perfect activity match for her stated interest
      
      Alternative Good Tuple: "Competitive gaming requires proper fuel and atmosphere. Drink craft beers at I/O Arcade Bar with me, Megan?"
      Analysis:
      - Bridge (6-9 words): ✓ Links gaming competition to gaming venue atmosphere
      - Activity (verb-first): ✓ Specific action verb "Drink" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Gaming-adjacent activity that complements her interest

    - Prompt: "A non-negotiable for me is"
      Response: "Having a good book on the go at all times"
      Photo context: Reading at a cafe
      Good Tuple: "Book lovers love cozy literary spaces. Browse the fiction section at Lake City Books together, Lily?"
      Analysis:
      - Bridge (6-9 words): ✓ Links reading interest to literary environment value
      - Activity (verb-first): ✓ Specific action verb "Browse" with proper venue
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Natural extension of her reading interest
      
      Alternative Good Tuple: "Reading pairs perfectly with artisanal coffee flavors. Sip pour-overs at Colectivo Coffee with me, Lily?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects reading to complementary coffee experience
      - Activity (verb-first): ✓ Specific action verb "Sip" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Builds on context of reading at cafe from her photo

    - Prompt: "The one thing I want to know about you is"
      Response: "If you have a favorite hiking trail"
      Photo context: Yoga on a beach
      Good Tuple: "Nature lovers appreciate different outdoor experiences. Explore a trail at Devil's Lake State Park together, Hannah?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects hiking interest to broader outdoor experiences
      - Activity (verb-first): ✓ Specific action verb "Explore" with proper location
      - Name: ✓ Name after "together" with comma and question mark
      - Overall: ✓ Direct response to her explicit interest in hiking trails
      
      Alternative Good Tuple: "Hiking enthusiasm translates well to yoga practice. Join outdoor yoga class at Olbrich Gardens with me, Hannah?"
      Analysis:
      - Bridge (6-9 words): ✓ Connects hiking interest to yoga activity
      - Activity (verb-first): ✓ Specific action verb "Join" with proper venue
      - Name: ✓ Name after "with me" with comma and question mark
      - Overall: ✓ Incorporates both her stated interest and photo activity

    BAD DATE SUGGESTION TUPLES:
    - Prompt: "A random fact I love is"
      Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight. And twilight is a fanfic of My Chemical Romance, and if 9/11 didn't happen, they wouldn't exist. It's weird I know"
      Photo context: Woman at a bookstore
      Bad Tuple: "Literary connections show your intellectual curiosity. Would you like to explore this fascinating domino effect at a bookstore with me, Kate?"
      Analysis:
      - Bridge: ✗ Too generic about "literary connections" without specificity
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Vague connection and indirect verb structure

    - Prompt: "A random fact I love is"
      Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight. And twilight is a fanfic of My Chemical Romance, and if 9/11 didn't happen, they wouldn't exist. It's weird I know"
      Photo context: Woman at a bookstore
      Bad Tuple: "Literary connections show your intellectual curiosity. Let's explore this fascinating domino effect at a bookstore together, Kate?"
      Analysis:
      - Bridge: ✗ Too generic about "literary connections" without specificity
      - Activity: ✗ Doesn't start with verb - uses "Let's" instead of direct verb
      - Name: ✓ Name included with comma and question mark
      - Overall: ✗ Vague connection and indirect verb structure

    - Prompt: "The quickest way to my heart is"
      Response: "Talking over ice cream/coffee"
      Photo context: Woman at a cafe
      Bad Tuple: "Deep conversations are better with sweet treats. Want to get some ice cream at Memorial Union and chat with me, Madison?"
      Analysis:
      - Bridge: ✓ Decent bridge connecting conversation and treats
      - Activity: ✗ Doesn't start with verb - uses "Want to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure makes it less concise and direct

    - Prompt: "The quickest way to my heart is"
      Response: "Talking over ice cream/coffee"
      Photo context: Woman at a cafe
      Bad Tuple: "Deep conversations are better with sweet treats. Let's get some ice cream at Memorial Union and chat, Madison?"
      Analysis:
      - Bridge: ✓ Decent bridge connecting conversation and treats
      - Activity: ✗ Doesn't start with verb - uses "Let's" instead of direct verb
      - Name: ✓ Name included with comma and question mark
      - Overall: ✗ Indirect verb structure makes it less concise and direct

    - Prompt: "My biggest date fail"
      Response: "Going swimming in Lake Mendota and then cutting my foot on a piece of metal... the date ended because I had to go to Urgent Care for a tetanus shot 😜"
      Photo context: Woman at a lake
      Bad Tuple: "Adventure mishaps make for great stories to share. Would you like to try swimming in Lake Mendota again but safely together, Rachel?"
      Analysis:
      - Bridge: ✗ Frames negative experience as positive without clear justification
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Suggests recreating a negative experience from her profile

    - Prompt: "My best celebrity impression is..."
      Response: "\"Why hello there,\" in my best Obi-Wan Kenobi voice and my friends hate it."
      Photo context: Woman with friends
      Bad Tuple: "Star Wars references reveal your fun pop culture side. Want to have a Star Wars marathon and practice impressions together, Sarah?"
      Analysis:
      - Bridge: ✗ Too generic about "pop culture" without specificity
      - Activity: ✗ Doesn't start with verb - uses "Want to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Generic activity that's too long (movie marathon exceeds 90 minutes)

    - Prompt: "My best celebrity impression is..."
      Response: "\"Why hello there,\" in my best Obi-Wan Kenobi voice and my friends hate it."
      Photo context: Woman with friends
      Bad Tuple: "Star Wars references reveal your fun pop culture side. Let's have a Star Wars marathon and practice impressions together, Sarah?"
      Analysis:
      - Bridge: ✗ Too generic about "pop culture" without specificity
      - Activity: ✗ Doesn't start with verb - uses "Let's" instead of direct verb
      - Name: ✓ Name included with comma and question mark
      - Overall: ✗ Activity is too long and non-specific

    - Prompt: "A random fact I love is"
      Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight. And twilight is a fanfic of My Chemical Romance, and if 9/11 didn't happen, they wouldn't exist. It's weird I know"
      Photo context: Woman at a bookstore
      Bad Tuple: "Literary connections show your intellectual curiosity. Would you like to explore this fascinating domino effect at a bookstore together, Kate?"
      Analysis:
      - Bridge: ✗ Too generic about "literary connections" without specificity
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Vague connection and activity not well-defined

    - Prompt: "The quickest way to my heart is"
      Response: "Talking over ice cream/coffee"
      Photo context: Woman at a cafe
      Bad Tuple: "Deep conversations are better with sweet treats. Want to get some ice cream at Memorial Union together, Madison?"
      Analysis:
      - Bridge: ✓ Decent bridge between conversation and treats
      - Activity: ✗ Doesn't start with verb - uses "Want to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure makes it less concise

    - Prompt: "I geek out on"
      Response: "NBA, specifically golden state warriors" 
      Photo context: Rock climbing at a gym
      Bad Tuple: "Sports fans appreciate good competition and teamwork. Would you like to try some beginner routes at Boulders Climbing Gym with me, Jessica?"
      Analysis:
      - Bridge: ✓ Links sports interest to physical activity
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure makes it less concise and direct

    - Prompt: "My biggest date fail"
      Response: "Going swimming in Lake Mendota and then cutting my foot on a piece of metal... the date ended because I had to go to Urgent Care for a tetanus shot 😜"
      Photo context: Woman at a lake
      Bad Tuple: "Everyone has awkward date stories to share. Would you like to grab coffee at Colectivo together, Rachel?"
      Analysis:
      - Bridge: ✓ Reasonable bridge about shared experiences
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure and generic connection

    - Prompt: "What I order for the table"
      Response: "Carbs, cheese and wine"
      Photo context: Woman at a bar
      Bad Tuple: "Wine enthusiasts deserve quality local venues for tasting. Would you like to try a flight of reds at Eno Vino together, Mia?"
      Analysis:
      - Bridge: ✓ Links wine interest to quality tasting experience
      - Activity: ✗ Doesn't start with verb - uses "Would you like to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure makes it less concise

    - Prompt: "A non-negotiable for me is"
      Response: "Having a good book on the go at all times"
      Photo context: Woman reading at a cafe
      Bad Tuple: "Book lovers appreciate thoughtfully curated collections. Want to browse the fiction section at Lake City Books together, Lily?"
      Analysis:
      - Bridge: ✓ Links reading interest to literary environment value
      - Activity: ✗ Doesn't start with verb - uses "Want to" instead of direct verb
      - Name: ✓ Name included with comma
      - Overall: ✗ Indirect verb structure makes it less direct and concise"""

COMPLIMENT_GENERATION = """STEP 8: GENERATE COMPLIMENTS
For each photo (there will be exactly 6 photos):
1. Generate THREE unique compliments (twelve to fifteen words each) based on something visibly attractive in each photo:
   a. First, find something that stands out and is visibly attractive from the profile photo.
      - Focus on concrete, observable things in her photos:
        * Physical appearance (only if obvious and tasteful)
           - A genuine smile in a photo
           - A distinctive hairstyle or color
           - A stylish outfit or specific piece of clothing
           - Her tattoos (take note of ink color, size, placement, and design)
           - Good posture in activities (yoga, surfing, dancing)
           - Athletic form when engaged in sports
           - Balanced movements captured in action shots
        * Activities she's doing in photos
          - Good posture while doing an activity
          - Muscle definition in activities (rock climbing, weightlifting, etc.)
          - Calm expression while cooking or focused on something
        * Specific items or objects in photos
          - Unique jewelry or accessories
          - Interesting shoes or hat
      - AVOID vague or personality-based observations like:
        * seeming like she knows how to have fun
        * seeming like a fun person
        * looking adventurous
        * seeming really cool
        * looking like she has a great personality
      - AVOID complimenting items she won't bring on a date:
        * Yoga mats, sports equipment, pets, scenery
        * Temporary items specific to the activity in the photo
      - The goal is to find something natural or intrinsic about her that is attractive to us
      - People like to be noticed for specific qualities they have in photos
      - Keep it straightforward and honestly observable
      - The attractive quality must be something you could literally point to in a photo
      - No marketing speak or over-the-top descriptions
   b. Turn this observation into a direct compliment
      - Express that you want to experience this attractive trait in person with fewer words
      - Format: "I want to see that [specific attractive trait] from your [specific photo context] in person" or "Please show me that [specific attractive trait] from your [specific photo context] firsthand"
      - Examples:
        * "I want to see that straight posture from your surfing photo in person"
        * "I want to see that short bob haircut from your cooking photo firsthand"
        * "Please show me that bright smile from your beach photo in person"
        * "Please show me those silver hoop earrings from your profile photo firsthand"
      - CRITICAL: The compliment MUST be:
        * Based on something specifically visible in the photo
        * Simple, direct, and straightforward - no unnecessary words
        * Expressed with minimal wording (use "I want to see" or "Please show me")
        * Always reference the specific photo context
        * Between 12-15 words in total length
        * Selfish but genuine (what YOU want to experience about her)
   
2. For each compliment, make sure it's:
   * COMPLETELY distinct from other compliments for the same photo
   * Based on different visual elements in the same photo
   * Varies in wording and structure
   * Easy to understand and unambiguous
   * Flattering without being excessive
   * Reflective of something tangible in her profile

3. This will result in 18 total compliments (3 for each of the 6 photos). Count the number of compliments. Are there 18? If not, you need to go back and generate three compliments per photo for a total of 18.

Examples of good vs bad COMPLIMENTS:

GOOD COMPLIMENTS:
- Prompt: "My Love Language is"
  Response: "Words of affirmation and cheese"
  Photo context: Woman playing board games with friends
  Good Compliment: "I want to see that bright smile from your board game photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her smile)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Genuine, specific, and tied to visible attribute
  
  Alternative Good Compliment: "Please show me that relaxed posture from your board game photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her posture)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Observes a different visual element from the same photo

- Prompt: "I geek out on"
  Response: "NBA, specifically golden state warriors" 
  Photo context: Woman rock climbing at a gym
  Good Compliment: "Please show me that strong jawline from your climbing photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her jawline)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Specific to her physical appearance
  
  Alternative Good Compliment: "I want to see those muscles from your climbing photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her eyes)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Comments on a physical feature that will be present on the date

- Prompt: "Two truths and a lie..."
  Response: "1) I have a scar on my hand from being bitten by a hamster 2) I wrote and directed a short film 3) I won a county-wide math competition in school"
  Photo context: Woman at the beach wearing a hat
  Good Compliment: "I want to see that wide-brimmed sun hat from your beach photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her hat)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Specific to an item she chose to wear
  
  Alternative Good Compliment: "Please show me that long wavy hair from your beach photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her hair)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Comments on a physical feature that will be present on the date

- Prompt: "What I order for the table"
  Response: "Carbs, cheese and wine"
  Photo context: Woman with curly hair in profile photo
  Good Compliment: "Please show me that curly hair from your profile photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible and distinctive (her hair)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Personal, distinctive, and tied to a natural feature
  
  Alternative Good Compliment: "I want to see that warm smile from your profile photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her smile)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Highlights a physical feature she'll definitely have on a date

- Prompt: "My simple pleasures"
  Response: "Carbs, cheese, and wine"
  Photo context: Woman with tattoo visible in board game photo
  Good Compliment: "I want to see that black geometric tattoo from your board game photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her tattoo)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Specific to a permanent feature with details
  
  Alternative Good Compliment: "Please show me those brown bangs from your board game photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her eyes)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Comments on a physical feature she'll definitely have on a date

- Prompt: "The dorkiest thing about me is"
  Response: "I'm currently studying all of world history to pass a teaching exam, help me out!"
  Photo context: Woman hiking on a trail
  Good Compliment: "I want to see that good posture from your trail photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her posture)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Specific to her physical appearance in activity
  
  Alternative Good Compliment: "Please show me that hiking outfit from your trail photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her outfit)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Comments on clothing she might wear again

- Prompt: "The quickest way to my heart is"
  Response: "Making me laugh and ice cream"
  Photo context: Woman with tattoo at beach
  Good Compliment: "Please show me that blue mermaid tattoo from your beach photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her tattoo)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Specific to a distinctive personal feature
  
  Alternative Good Compliment: "I want to see that confident posture from your beach photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her posture)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Highlights a physical attribute she'll bring to the date

- Prompt: "I'm weirdly attracted to"
  Response: "Guys who can beat me at Mario Kart"
  Photo context: Woman at a party with glasses
  Good Compliment: "I want to see those round glasses from your party photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her glasses)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Specific to an accessory she chose to wear
  
  Alternative Good Compliment: "Please show me that cute hairstyle from your party photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her hairstyle)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Highlights a physical feature she'll bring to the date

- Prompt: "A non-negotiable for me is"
  Response: "Having a good book on the go at all times"
  Photo context: Woman at cafe wearing earrings
  Good Compliment: "Please show me those dangling earrings from your cafe photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her earrings)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Specific to a distinctive accessory
  
  Alternative Good Compliment: "I want to see that relaxed demeanor from your cafe photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her demeanor/expression)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Comments on a visible attribute she'll bring to the date

- Prompt: "The one thing I want to know about you is"
  Response: "If you have a favorite hiking trail"
  Photo context: Woman doing yoga on a beach
  Good Compliment: "I want to see that good posture from your beach photo in person."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her posture)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "I want to see"
  - Overall: ✓ Specific to her physical form during activity
  
  Alternative Good Compliment: "Please show me that vibrant hair color from your beach yoga photo firsthand."
  Analysis:
  - Observable attribute: ✓ Focuses on something visible (her hair color)
  - Photo context: ✓ References specific photo context
  - Wording: ✓ Direct and clear with "Please show me"
  - Overall: ✓ Highlights a physical feature she'll bring to the date

BAD COMPLIMENTS:
- Prompt: "A random fact I love is"
  Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight..."
  Photo context: Woman at a bookstore
  Bad Compliment: "You seem like someone who enjoys deep conversations."
  Analysis:
  - Observable attribute: ✗ Focuses on a personality trait, not something visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Completely misses the "I want to see" or "Please show me" format
  - Overall: ✗ Generic personality assumption not tied to anything visible

- Prompt: "A random fact I love is"
  Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight..."
  Photo context: Woman at a bookstore
  Bad Compliment: "Your determination would make this activity better."
  Analysis:
  - Observable attribute: ✗ Focuses on a personality trait, not something visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Completely misses the "I want to see" or "Please show me" format
  - Overall: ✗ Vague personality assumption not tied to anything visible

- Prompt: "The quickest way to my heart is"
  Response: "Talking over ice cream/coffee"
  Photo context: Woman at a cafe
  Bad Compliment: "Your profile suggests you're a great conversationalist."
  Analysis:
  - Observable attribute: ✗ Focuses on a personality trait, not something visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Completely misses the "I want to see" or "Please show me" format
  - Overall: ✗ Vague inference not based on anything visible

- Prompt: "The quickest way to my heart is"
  Response: "Talking over ice cream/coffee"
  Photo context: Woman at a park with friends
  Bad Compliment: "I'd love to see your adventurous spirit in real life."
  Analysis:
  - Observable attribute: ✗ Focuses on a personality trait, not something visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Uses wordy "I'd love to see" and "in real life" instead of "in person"
  - Overall: ✗ Vague, assumption-based, not tied to visible attribute

- Prompt: "My biggest date fail"
  Response: "Going swimming in Lake Mendota and then cutting my foot on a piece of metal..."
  Photo context: Woman at a lake
  Bad Compliment: "You seem adventurous and willing to try things again."
  Analysis:
  - Observable attribute: ✗ Makes personality assumption rather than commenting on visible attribute
  - Photo context: ✗ Vaguely references the outdoor setting but not specific details
  - Wording: ✗ Completely misses the "I want to see" or "Please show me" format
  - Overall: ✗ Generic personality assessment not tied to anything visible

- Prompt: "My best celebrity impression is..."
  Response: "\"Why hello there,\" in my best Obi-Wan Kenobi voice and my friends hate it."
  Photo context: Woman with friends
  Bad Compliment: "I would love to see you bring entertainment to any situation."
  Analysis:
  - Observable attribute: ✗ Focuses on a personality trait, not something visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Uses wordy "I would love to see" format
  - Overall: ✗ Generic personality assumption not tied to anything visible

- Prompt: "My best celebrity impression is..."
  Response: "\"Why hello there,\" in my best Obi-Wan Kenobi voice and my friends hate it."
  Photo context: Woman with friends
  Bad Compliment: "I'd love to see your intelligence in real life."
  Analysis:
  - Observable attribute: ✗ Focuses on a non-physical quality that isn't directly observable
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Uses wordy "I'd love to see" and "in real life" instead of "in person"
  - Overall: ✗ Comments on an intellectual quality not visible in photos

- Prompt: "A random fact I love is"
  Response: "My favorite domino effect is that Fifty Shades of Grey is a fanfic of twilight..."
  Photo context: Woman at a bookstore
  Bad Compliment: "I'd love to see your knowledge of literary history in real life."
  Analysis:
  - Observable attribute: ✗ Focuses on an intellectual quality that isn't directly observable
  - Photo context: ✗ Vaguely references literary interest but not visual details
  - Wording: ✗ Uses wordy "I'd love to see" and "in real life" instead of "in person"
  - Overall: ✗ Comments on knowledge (not visible) and relates to prompt content

- Prompt: "The quickest way to my heart is"
  Response: "Talking over ice cream/coffee"
  Photo context: Woman at a cafe
  Bad Compliment: "I'd love to see your focused expression when you're listening."
  Analysis:
  - Observable attribute: ✗ Uses subjective qualifier "focused" rather than describing what is visible
  - Photo context: ✗ References the cafe but not specific visible details
  - Wording: ✗ Uses wordy "I'd love to see" format
  - Overall: ✗ Comments on a subjective quality using marketing-like language

- Prompt: "I geek out on"
  Response: "NBA, specifically golden state warriors" 
  Photo context: Woman rock climbing at a gym
  Bad Compliment: "I'd love to see how your impressive grip strength translates to handshakes."
  Analysis:
  - Observable attribute: ✗ Uses subjective qualifier "impressive" rather than describing what is visible
  - Photo context: ✗ References climbing but uses subjective judgment
  - Wording: ✗ Uses wordy "I'd love to see how" format
  - Overall: ✗ Comments on a quality using unnecessary qualifiers

- Prompt: "My biggest date fail"
  Response: "Going swimming in Lake Mendota and then cutting my foot on a piece of metal..."
  Photo context: Woman at a lake
  Bad Compliment: "I'd love to see your genuine smile in person sometime."
  Analysis:
  - Observable attribute: ✗ Uses subjective qualifier "genuine" rather than describing what is visible
  - Photo context: ✗ Doesn't reference specific photo context
  - Wording: ✗ Uses wordy "I'd love to see" format with "sometime"
  - Overall: ✗ Uses marketing-like language with unnecessary qualifier

- Prompt: "What I order for the table"
  Response: "Carbs, cheese and wine"
  Photo context: Woman at a bar
  Bad Compliment: "I'd love to see your colorful yoga mat on our next outing."
  Analysis:
  - Observable attribute: ✗ Focuses on an item (yoga mat) she won't bring on a date
  - Photo context: ✗ Doesn't reference the actual photo context (bar)
  - Wording: ✗ Uses wordy "I'd love to see" format
  - Overall: ✗ Comments on an item that won't be present for the date

- Prompt: "A non-negotiable for me is"
  Response: "Having a good book on the go at all times"
  Photo context: Woman reading at a cafe
  Bad Compliment: "I'd love to see how being well-read influences your browsing."
  Analysis:
  - Observable attribute: ✗ Focuses on an intellectual quality that isn't directly observable
  - Photo context: ✗ Vaguely references reading but not specific visual details
  - Wording: ✗ Uses wordy "I'd love to see how" format
  - Overall: ✗ Comments on being well-read (not visible) and relates to prompt and activity"""

EVALUATE_DATE_TUPLES = """STEP 9: EVALUATE DATE SUGGESTION TUPLES
For each of the NINE tuples (3 per prompt/response pair):
1. Conduct a simulated conversation:
   a. Message 1: Her prompt/response as the first message
   b. Message 2: Your tuple (bridge + activity + name) as the response
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
3. Rank each conversation on a scale of 1-10 for overall quality. Keep in mind if the conversation starter:
   - Uses the most natural, everyday language
   - Gets straight to the point
   - Shows interest through specificity, not enthusiasm
   - Creates easy conversation flow
   - Respects the complexity of her personality
   - Creates positive imagery or stories in her mind
   Is between eight to ten words (wordy date suggestions are penalized).
4. Select the single best conversation starter tuple ranked highest by you.
5. Re-validate the location of the establishment and if it is responding to a poll prompt in your conversation starter tuple:
   - CRITICAL: Location validation - FOLLOW THESE STEPS CAREFULLY:
      * Carefully scan your conversation starter for ANY specific venue or business name (e.g., "Brocach", "Barriques", "Eno Vino")
      * If you find one, check if it appears EXACTLY as written in the locations.txt file
      * Search through the ENTIRE locations.txt file - check every category
      * If the exact name is not in locations.txt, IMMEDIATELY DISQUALIFY this starter
      * Replace it with a starter that either uses a generic venue type (e.g., "coffee shop downtown") or uses a venue from locations.txt
      * Do NOT use similar spellings or variations - match EXACTLY what's in locations.txt
      * This validation is MANDATORY - never return a starter with an unverified location
   - CRITICAL: POLL VALIDATION CHECK - FOLLOW THESE STEPS CAREFULLY:
      * Review the prompt you selected for your best conversation starter
      * Check if this prompt appears in hingePolls.txt
      * If it does, this is a poll prompt and MUST NOT be used for conversation starters
      * If you selected a poll prompt, immediately disqualify it and select a different one from the three valid prompt/response pairs
      * A valid prompt must NOT be in hingePolls.txt
      * There are always exactly three non-poll prompts in every profile - ensure you've selected one of these"""

EVALUATE_COMPLIMENTS = """STEP 10: EVALUATE COMPLIMENTS
For each of the 18 compliments (3 per photo):
1. Conduct a simulated conversation:
   a. Imagine adding this compliment to a standard date suggestion
   b. Message 1: Her receiving the full message with this compliment
   c. Message 2: Imagine her most likely reply based on profile characteristics
   d. Message 3: Your natural follow-up based on her reply
2. Evaluate based on these criteria:
   - Is the compliment easy to understand immediately?
   - Is it flattering without being excessive?
   - Is it unambiguous in meaning?
   - Does it clearly reference something visible in her photos?
   - Does it avoid sounding generic or like it could apply to anyone?
   - Would it make her feel noticed for something specific about her?
   - Does it avoid objectifying her inappropriately?
   - Would it be received positively by most women?
3. Rank each compliment on a scale of 1-10 for overall quality. Make sure that the compliment:
   - Makes her feel genuinely noticed
   - References something unique or distinctive about her
   - Is direct and clear without requiring interpretation
   - Uses simple, straightforward language
   - Avoids being overly sexual or inappropriate
   - Creates a positive feeling when read
   - Penalize compliments that are overly wordy/too long (more than twelve 15 words)
4. Select the single best compliment ranked highest by you.
   """

COMBINE_COMPONENTS = """STEP 11: COMBINE TUPLE AND COMPLIMENT
1. Take the winning tuple from Step 9
   - This includes the bridge, activity suggestion, and name with question mark
2. Take the winning compliment from Step 10
   - This includes the direct statement about something attractive in her photo
3. Combine them in this exact format:
   "[Bridge. Activity suggestion with me, NAME? [Compliment]."
4. Verify that the combined message flows naturally
5. Ensure the compliment remains completely separate from the activity suggestion
6. The final conversation starter should be approximately 25-35 words in total"""

IDENTIFY_SCREENSHOT = """STEP 12: IDENTIFY SCREENSHOT
1. Note which screenshot contains the prompt/response pair text of the woman's profile that matches the chosen starter
2. If prompt/response pair is cut off or spans multiple screenshots, note which screenshot contains the majority of the prompt/response pair text"""

FINAL_VALIDATION_CHECK = """STEP 13: FINAL VALIDATION CHECK
Before returning your result, perform one last critical check:
1. Review your conversation_starter one more time
2. If it mentions ANY specific venue name:
   - Verify that this EXACT venue name appears in locations.txt
   - If it doesn't match EXACTLY (including spelling), modify your starter to use a generic venue type instead
   - This is your LAST chance to avoid mentioning venues not in locations.txt
3. Double-check that your selected starter follows all the guidelines and passes all validation checks"""

# New shared prompt component for desirability-only path
ENDING_DESIREABILITY_ONLY = """If the woman is desirable (doesn't meet any undesirable traits in Step 4), return the following JSON exactly, where prompt, response, and conversation_starter are empty strings, and screenshot_index is 0:
{
    "prompt": "",
    "response": "",
    "conversation_starter": "",
    "screenshot_index": 0
}"""


def generate_bumble_reply_from_screenshots(screenshots, format_txt_path, prompts_txt_path, interests_txt_path, metadata_txt_path, compliments_available=True):
    """
    Generate a contextually appropriate reply for a Bumble profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to bumbleFormat.txt describing profile structure
        prompts_txt_path: Path to bumblePrompts.txt containing available prompts
        interests_txt_path: Path to bumbleInterests.txt containing interest categories
        metadata_txt_path: Path to bumbleMetadata.txt containing profile fields
        compliments_available: Whether "send a compliment" is available on this profile

    Returns:
        dict: Contains the prompt-response pair, generated response, and screenshot index:
        {
            "prompt": str,      # The exact prompt text being responded to
            "response": str,    # The user's response to the prompt
            "conversation_starter": str,  # The generated conversation starter
            "screenshot_index": int,  # 0-based index of screenshot containing prompt/response
        }
    """
    # Read the content of our context files
    context_files = read_context_files({
        'format': format_txt_path,
        'prompts': prompts_txt_path,
        'interests': interests_txt_path,
        'metadata': metadata_txt_path
    })

    if not context_files:
        return None

    # Convert screenshots to base64 for API transmission
    screenshot_data = prepare_screenshots_for_api(screenshots)
    if not screenshot_data:
        return None

    # System prompt containing all the structural information
    system_prompt = f"""{DATING_APP_INTRO}

    PROFILE STRUCTURE:
    You will analyze 7 screenshots of a Bumble profile. Each profile may contain the following elements (following the exact order described in bumbleFormat.txt):
    1. Photos (0-7 total)
    2. About Me section (optional)
    3. Basic Information attributes (optional)
    4. "I'm looking for" section (optional)
    5. "My interests" section (optional, 0-5 selections)
    6. Prompt/Response pairs (optional, 0-3 total)
    7. "My causes and communities" section (optional, 0-3 selections)
    8. Location information
    
    Each of these elements tells you something about the woman. Your goal is to analyze her profile comprehensively to see if she is desireable/undesireable.

    REFERENCE FILES CONTENT:
    format.txt:
    {context_files['format']}

    prompts.txt:
    {context_files['prompts']}

    interests.txt:
    {context_files['interests']}

    metadata.txt:
    {context_files['metadata']}

    STEP 1: READ AND UNDERSTAND THE CONTEXT
    1. Read format.txt to understand the profile layout
    2. Consult prompts.txt to understand available prompts
    3. Examine interests.txt to understand interest categories
    4. Review metadata.txt to understand profile attribute meanings
    5. Examine each screenshot to identify all profile elements

    {IDENTIFY_MAIN_PERSON}

    STEP 3: ORGANIZE ELEMENTS INTO BUCKETS
    For each screenshot, group elements into these buckets:
    a. Photos
       - Is the main person in this photo?
       - What is she doing? With whom?
       - What does this reveal about her?
    b. About Me text
       - What does she explicitly state about herself?
       - What does she value enough to mention?
    c. Attributes (basic info)
       - What factual information does she share?
       - How do these fit together to create a lifestyle?
    d. Looking For
       - What relationship type is she seeking?
       - What qualities matter to her?
    e. Interests
       - What activities does she enjoy?
       - Are there themes among interests?
    f. Prompt/Response pairs
       - Which prompts did she choose?
       - What do her responses reveal?
    g. Causes
       - What societal issues matter to her?
       - How important are these to her identity?

    {UNDESIRABLE_TRAITS_CHECK}
    """

    # Add different paths based on compliments_available
    if compliments_available:
        system_prompt += f"""
    {STORY_ANALYSIS_STEPS}

    {COMPILE_CHARACTERISTICS}

    {CONVERSATION_STARTER_GUIDE}

    {SIMULATE_CONVERSATION}

    {SELECT_BEST_STARTER}

    {IDENTIFY_SCREENSHOT}

    {FINAL_VALIDATION_CHECK}
    
    Return the chosen prompt (if applicable), response (if applicable), your conversation starter, and the screenshot index in this JSON format exactly. Do not include any other message.
    {{
        "prompt": "The exact prompt text if replying to a prompt, otherwise empty string",
        "response": "The user's response to the prompt if replying to a prompt, otherwise empty string",
        "conversation_starter": "Your natural conversation starter",
        "screenshot_index": index_of_screenshot_containing_target_element  # 0-based index
    }}"""
    else:
        system_prompt += f"""
    {ENDING_DESIREABILITY_ONLY}"""

    # User message - just the specific task
    user_message = """Please analyze these Bumble profile screenshots and return the requested data as instructed."""

    # Make the API call to Claude
    logger.info("Making API call to Claude...")

    response = call_claude_api(
        system_prompt=system_prompt,
        user_message=user_message,
        screenshots_data=screenshot_data
    )

    if not response:
        return None

    # Parse the response
    result = parse_claude_json_response(response)
    if not result:
        return None

    return {
        "prompt": result.get("prompt", ""),
        "response": result.get("response", ""),
        "conversation_starter": result.get("conversation_starter", ""),
        "screenshot_index": result.get("screenshot_index", 0)
    }


def generate_hinge_reply_from_screenshots(screenshots, format_txt_path, prompts_txt_path, captions_txt_path, polls_txt_path, locations_txt_path):
    """
    Generate a contextually appropriate response for a Hinge profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to hingeFormat.txt describing profile structure
        prompts_txt_path: Path to hingePrompts.txt containing available prompts
        captions_txt_path: Path to hingeCaptions.txt containing possible captions
        polls_txt_path: Path to hingePolls.txt containing available polls
        locations_txt_path: Path to locations.txt containing Madison, WI establishments

    Returns:
        dict: Contains the prompt-response pair, generated conversation starter, and screenshot index:
        {
            "prompt": str,      # The exact prompt text being responded to
            "response": str,    # The user's response to the prompt
            "conversation_starter": str,  # The generated conversation starter
            "screenshot_index": int,  # 0-based index of screenshot containing prompt/response
        }
    """
    # Read the content of our context files
    context_files = read_context_files({
        'format': format_txt_path,
        'prompts': prompts_txt_path,
        'captions': captions_txt_path,
        'polls': polls_txt_path,
        'locations': locations_txt_path
    })

    if not context_files:
        return None

    # Convert screenshots to base64 for API transmission
    screenshot_data = prepare_screenshots_for_api(screenshots)
    if not screenshot_data:
        return None

    # System prompt containing all the structural information
    system_prompt = f"""{DATING_APP_INTRO}

    PROFILE STRUCTURE:
    You will receive exactly 7 screenshots of a Hinge profile in order (index 0 to 6). Each profile will contain the following guaranteed elements:
    1. Exactly 6 photos (may have captions)
    2. Exactly 3 prompt/response pairs
    3. One section of profile basics
    4. Optionally one voice prompt
    5. Optionally one poll prompt

    Each of these elements is a "story" about the woman - something she has chosen to share about herself. There will always be at least 10 stories (6 photos + 3 prompts + 1 basics) and up to 12 stories if she includes a voice prompt and poll prompt.

    REFERENCE FILES CONTENT:
    format.txt:
    {context_files['format']}

    prompts.txt:
    {context_files['prompts']}

    captions.txt:
    {context_files['captions']}

    polls.txt:
    {context_files['polls']}

    locations.txt:
    {context_files['locations']}

    STEP 1: READ AND UNDERSTAND THE CONTEXT
    1. Read format.txt to understand the profile layout
    2. Consult prompts.txt, captions.txt, and polls.txt to understand possible elements
    3. Examine each screenshot in order (0 to 6) and identify all elements
    4. Consult locations.txt to understand the types of establishments that exist in Madison, Wisconsin

    {IDENTIFY_MAIN_PERSON}

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
       - CRITICAL: Check if each prompt appears in hingePolls.txt - if it does, it's a poll prompt, NOT a standard prompt
       - Poll prompts have a distinctive visual identifier: each of the three response options has a black heart with a white center
       - Poll responses are hypothetical scenarios created by the woman and reflect her creativity, not actual experiences
       - NEVER select poll prompts (found in hingePolls.txt) for conversation starters
    c. Profile Basics (1 total)
       - All bullet-points she provided to describe herself
       - These are her self-reported characteristics
    d. Voice Prompt (0-1 total)
       - Cannot understand the recording, only see the prompt
       - Always appears as the first prompt when present
       - Content of voice note is NOT accessible and should NOT be used for analysis
       - Prompt text itself can still be used for personality analysis
       - Voice notes are NOT valid targets for joke responses
    e. Poll Prompt (0-1 total)
       - A distinct type of prompt found in hingePolls.txt
       - Shows a question with exactly three creative response options provided by the woman
       - Each option has a black heart with white center (key visual identifier)
       - CRITICAL: Must NEVER be used for conversation starters - only for personality analysis
       - IMPORTANT: Poll responses are hypothetical scenarios the woman created and say nothing about what she has actually done
       - They only reflect a moment in which she was creative, not actual experiences
       - Always check prompts against hingePolls.txt to identify polls
    f. Special Case - "Two truths and a lie":
       - This prompt appears in TWO possible formats:
         1. Standard poll format: Three distinct options, each with a black heart with white center
         2. Paragraph format: All three options in a single paragraph without heart icons
       - CRITICAL: Regardless of format, "Two truths and a lie" is ALWAYS treated as a poll prompt
       - It must NEVER be selected for conversation starter generation
       - When in paragraph format, the profile will only have two other prompt/response pairs available
       - Even when it doesn't have the visual poll identifiers, it IS still a poll prompt
       - Always check specifically for this prompt text in addition to checking hingePolls.txt

    {UNDESIRABLE_TRAITS_CHECK}

    {STORY_ANALYSIS_STEPS}

    {COMPILE_CHARACTERISTICS}

    {CONVERSATION_STARTER_GUIDE}

    {COMPLIMENT_GENERATION}

    {EVALUATE_DATE_TUPLES}

    {EVALUATE_COMPLIMENTS}

    {COMBINE_COMPONENTS}

    {IDENTIFY_SCREENSHOT}

    {FINAL_VALIDATION_CHECK}
    
    Return the chosen prompt, response, your conversation starter, and the screenshot index in this JSON format exactly. Do not return any other text or comments beyond the JSON.
    {{
        "prompt": "The exact prompt text the woman chose",
        "response": "The woman's response to the prompt",
        "conversation_starter": "Your natural conversation starter",
        "screenshot_index": index_of_screenshot_containing_prompt_response  # 0-based index (0-6)
    }}"""

    # User message - just the specific task
    user_message = """Please analyze these profile screenshots and return the requested data as instructed."""

    # Make the API call to Claude
    logger.info("Making API call to Claude...")

    response = call_claude_api(
        system_prompt=system_prompt,
        user_message=user_message,
        screenshots_data=screenshot_data
    )

    if not response:
        return None

    # Parse the response
    result = parse_claude_json_response(response)
    if not result:
        return None

    return {
        "prompt": result.get("prompt", ""),
        "response": result.get("response", ""),
        "conversation_starter": result.get("conversation_starter", ""),
        "screenshot_index": result.get("screenshot_index", 0)
    }


def find_prompt_response_match(screenshot_path, target_prompt, target_response, profile_num, suffix="", dating_app='hinge'):
    """Find prompt or response match in a screenshot.

    Args:
        screenshot_path: Path to the screenshot image
        target_prompt: The prompt text to look for
        target_response: The response text to look for
        profile_num: Current profile number for debugging
        suffix: Suffix for debug image filename (e.g. "_up" for scrolled up screenshot)
        dating_app: The dating app type ('hinge' or 'bumble') to optimize OCR settings

    Returns:
        tuple: (best_match, tap_coordinates, visualization_path) where:
            - best_match: The paragraph object that matched, or None if no match
            - tap_coordinates: (x, y) coordinates to tap if found, None if not found
            - visualization_path: Path to the created visualization image
    """
    # Extract text and group into paragraphs
    boxes = extract_text_from_image_with_boxes(
        screenshot_path, app_type=dating_app)
    if not boxes:
        logger.warning(f"No text boxes found in screenshot{suffix}")
        return None, None, None

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Try to find the target prompt with high confidence
    logger.debug(f"\nComparing target prompt against OCR paragraphs{suffix}:")
    logger.debug(f"Target prompt: '{target_prompt}'")
    logger.debug("\nOCR paragraphs found:")

    best_prompt_match = None
    best_prompt_ratio = 0.0
    best_response_match = None
    best_response_ratio = 0.0

    for i, para in enumerate(paragraphs):
        # Check for prompt match
        is_prompt_match, prompt_ratio, matched_text = fuzzy_match_text(
            target_prompt, para['text'], threshold=0.7)
        logger.debug(f"Paragraph {i+1}:")
        logger.debug(f"  Text: '{para['text']}'")
        logger.debug(f"  Prompt match ratio: {prompt_ratio:.2f}")

        # Also check for response match with same threshold
        is_response_match, response_ratio, _ = fuzzy_match_text(
            target_response, para['text'], threshold=0.7)
        logger.debug(f"  Response match ratio: {response_ratio:.2f}")

        # Update best matches if we found better ones
        if is_prompt_match and prompt_ratio > best_prompt_ratio:
            best_prompt_match = para.copy()  # Make a copy of the paragraph
            best_prompt_match['match_type'] = 'prompt'  # Add match type
            best_prompt_ratio = prompt_ratio

        if is_response_match and response_ratio > best_response_ratio:
            best_response_match = para.copy()  # Make a copy of the paragraph
            best_response_match['match_type'] = 'response'  # Add match type
            best_response_ratio = response_ratio

    # Use prompt match if found, otherwise use response match
    best_match = best_prompt_match if best_prompt_match else best_response_match
    visualization_path = f"images/profile_{profile_num}_prompt_detection{suffix}_visual.png"

    if best_match:
        match_type = best_match['match_type']
        match_ratio = best_prompt_ratio if match_type == 'prompt' else best_response_ratio
        logger.info(
            f"Found {match_type} match{suffix} with ratio {match_ratio:.2f}")

        # Calculate tap coordinates (center of the paragraph)
        boxes_match = best_match['boxes']
        min_x = min(box['box'][0] for box in boxes_match)
        max_x = max(box['box'][0] + box['box'][2] for box in boxes_match)
        min_y = min(box['box'][1] for box in boxes_match)
        max_y = max(box['box'][1] + box['box'][3] for box in boxes_match)

        tap_x = (min_x + max_x) // 2
        tap_y = (min_y + max_y) // 2

        # Create visualization with tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=visualization_path,
            tap_target=(tap_x, tap_y),
            profile_num=profile_num,
            app_name=dating_app
        )

        return best_match, (tap_x, tap_y), visualization_path
    else:
        logger.warning(
            f"No matching prompt or response found in screenshot{suffix}")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=visualization_path,
            profile_num=profile_num,
            app_name=dating_app
        )

        return None, None, visualization_path


def detect_prompt_in_screenshot(device, target_prompt, target_response, screenshot_index, profile_num, dating_app='hinge'):
    """Detect and visualize the target prompt or response in a screenshot.

    Args:
        device: The ADB device
        target_prompt: The prompt text we're looking for
        target_response: The response text we're looking for
        screenshot_index: Index of the screenshot to analyze
        profile_num: Current profile number
        dating_app: The dating app type ('hinge' or 'bumble') to optimize OCR settings

    Returns:
        tuple: (found, tap_coordinates, found_prompt_match, found_response_match) where:
            - found: bool indicating if prompt/response was found
            - tap_coordinates: (x,y) coordinates to tap if found, None if not found
            - found_prompt_match: bool indicating if the prompt was matched
            - found_response_match: bool indicating if the response was matched
    """
    try:
        # Check if we're dealing with "Two truths and a lie" special case
        is_two_truths = False
        response_versions = []

        if target_prompt and "two truths and a lie" in target_prompt.lower():
            logger.info("Special case: Detected 'Two truths and a lie' prompt")
            is_two_truths = True

            # Add the original response as first version
            response_versions.append(target_response)

            # Try to split by newlines (2-4 newlines)
            newline_splits = target_response.split('\n')
            if len(newline_splits) >= 3 and len(newline_splits) <= 5:
                # Filter out empty strings
                newline_splits = [s for s in newline_splits if s.strip()]
                if len(newline_splits) >= 3:
                    logger.info(
                        f"Found {len(newline_splits)} statements split by newlines")
                    # Take up to first 3 non-empty splits
                    response_versions.extend(newline_splits[:3])

            # If newline splitting didn't work, try splitting by periods
            if len(response_versions) < 3:
                period_splits = []
                # Split by period followed by space
                for split in target_response.split('. '):
                    if split.endswith('.'):
                        period_splits.append(split)
                    else:
                        period_splits.append(split + '.')

                # If we got 2-4 statements, add them
                if 2 <= len(period_splits) <= 4:
                    logger.info(
                        f"Found {len(period_splits)} statements split by periods")
                    # Take up to first 3 splits
                    response_versions.extend(period_splits[:3])

            logger.info(
                f"Created {len(response_versions)} response versions for matching")
            for i, v in enumerate(response_versions):
                logger.debug(f"Response version {i+1}: '{v[:50]}...'")

        # Take a fresh screenshot at this position
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_prompt_detection")

        # Variables to track whether prompt or response was matched
        found_prompt_match = False
        found_response_match = False

        # For two truths and a lie, try all response versions
        if is_two_truths and response_versions:
            # First try with all response versions
            for response_version in response_versions:
                best_match, tap_coordinates, _ = find_prompt_response_match(
                    screenshot_path, None, response_version, profile_num, dating_app=dating_app)

                if best_match and tap_coordinates:
                    logger.info(
                        f"Found match with response version: '{response_version[:30]}...'")
                    tap_x, tap_y = tap_coordinates
                    tap(device, tap_x, tap_y, double_tap=True)
                    time.sleep(2.0)
                    found_response_match = True
                    return True, tap_coordinates, False, True
        else:
            # Standard flow - check for match with original prompt/response
            best_match, tap_coordinates, _ = find_prompt_response_match(
                screenshot_path, target_prompt, target_response, profile_num, dating_app=dating_app)

            if best_match and tap_coordinates:
                tap_x, tap_y = tap_coordinates
                tap(device, tap_x, tap_y, double_tap=True)
                time.sleep(2.0)

                # Determine if prompt or response was matched
                if best_match.get('match_type') == 'prompt':
                    found_prompt_match = True
                else:
                    found_response_match = True

                return True, tap_coordinates, found_prompt_match, found_response_match

        # If no match found, try scrolling up
        logger.info("\nAttempting to scroll up once to find prompt/response...")
        swipe(device, "up")  # Scroll up
        time.sleep(0.5)  # Wait for scroll to complete

        # Take another screenshot after scrolling up
        screenshot_path_up = capture_screenshot(
            device, f"profile_{profile_num}_prompt_detection_up")

        # For two truths and a lie, try all response versions after scrolling up
        if is_two_truths and response_versions:
            # Try with all response versions
            for response_version in response_versions:
                best_match_up, tap_coordinates_up, _ = find_prompt_response_match(
                    screenshot_path_up, None, response_version, profile_num, suffix="_up", dating_app=dating_app)

                if best_match_up and tap_coordinates_up:
                    logger.info(
                        f"Found match after scrolling up with response version: '{response_version[:30]}...'")
                    tap_x, tap_y = tap_coordinates_up
                    tap(device, tap_x, tap_y, double_tap=True)
                    time.sleep(2.0)
                    found_response_match = True
                    return True, tap_coordinates_up, False, True
        else:
            # Standard flow - check for match with original prompt/response
            best_match_up, tap_coordinates_up, _ = find_prompt_response_match(
                screenshot_path_up, target_prompt, target_response, profile_num, suffix="_up", dating_app=dating_app)

            if best_match_up and tap_coordinates_up:
                tap_x, tap_y = tap_coordinates_up
                tap(device, tap_x, tap_y, double_tap=True)
                time.sleep(2.0)

                # Determine if prompt or response was matched
                if best_match_up.get('match_type') == 'prompt':
                    found_prompt_match = True
                else:
                    found_response_match = True

                return True, tap_coordinates_up, found_prompt_match, found_response_match

        # Scroll back down to original position
        logger.info("Scrolling back down to original position...")
        swipe(device, "down")
        time.sleep(0.5)

        # For two truths and a lie, try matching with the prompt if no response matches found
        if is_two_truths:
            # Take a fresh screenshot after scrolling back
            screenshot_path = capture_screenshot(
                device, f"profile_{profile_num}_prompt_detection_prompt")

            # Try matching with the prompt text
            best_match_prompt, tap_coordinates_prompt, _ = find_prompt_response_match(
                screenshot_path, target_prompt, None, profile_num, suffix="_prompt", dating_app=dating_app)

            if best_match_prompt and tap_coordinates_prompt:
                logger.info(
                    f"Found match with prompt text: '{target_prompt[:30]}...'")
                tap_x, tap_y = tap_coordinates_prompt
                tap(device, tap_x, tap_y, double_tap=True)
                time.sleep(2.0)
                found_prompt_match = True
                return True, tap_coordinates_prompt, True, False

            # Try scrolling up one last time to match the prompt
            logger.info(
                "\nAttempting to scroll up one last time to find prompt text...")
            swipe(device, "up")  # Scroll up
            time.sleep(0.5)  # Wait for scroll to complete

            # Take another screenshot after scrolling up
            screenshot_path_up_final = capture_screenshot(
                device, f"profile_{profile_num}_prompt_detection_up_final")

            # Try matching with the prompt text
            best_match_prompt_up, tap_coordinates_prompt_up, _ = find_prompt_response_match(
                screenshot_path_up_final, target_prompt, None, profile_num, suffix="_up_final", dating_app=dating_app)

            if best_match_prompt_up and tap_coordinates_prompt_up:
                logger.info(
                    f"Found match with prompt text after scrolling up one last time: '{target_prompt[:30]}...'")
                tap_x, tap_y = tap_coordinates_prompt_up
                tap(device, tap_x, tap_y, double_tap=True)
                time.sleep(2.0)
                found_prompt_match = True
                return True, tap_coordinates_prompt_up, True, False

            # Scroll back down to original position
            logger.info(
                "Scrolling back down to original position one last time...")
            swipe(device, "down")

        # No match found after all attempts
        logger.warning("No matching prompt or response found")
        return False, None, False, False

    except Exception as e:
        logger.error(f"Error in detect_prompt_in_screenshot: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level
        return False, None, False, False


def dislike_profile(device, dating_app='hinge'):
    """Execute a dislike action by tapping the X button or swiping left.

    Args:
        device: The ADB device
        dating_app: The dating app ('hinge', 'bumble', or 'tinder') to adjust coordinates/actions

    Returns:
        None
    """
    if dating_app == 'bumble':
        # For Bumble, swipe left to dislike
        width, height = get_screen_resolution(device)
        swipe_start_x = int(width * 0.8)  # Start at 80% of screen width
        swipe_end_x = int(width * 0.2)    # End at 20% of screen width
        swipe_y = int(height * 0.5)       # Middle of screen height

        # Execute the left swipe
        device.shell(
            f"input swipe {swipe_start_x} {swipe_y} {swipe_end_x} {swipe_y} 300")
    elif dating_app == 'hinge':
        # Hinge dislike button coordinates
        tap(device, 140, 2075, with_additional_swipe=False)
    elif dating_app == 'tinder':
        # Tinder dislike button coordinates
        tap(device, 330, 2050, with_additional_swipe=False)

    # No need to wait here as we'll wait at the beginning of the next profile processing


def like_profile(device, dating_app):
    """Execute a like action by tapping the like button or swiping right.

    Args:
        device: The ADB device
        dating_app: The dating app ('bumble' or 'tinder') to determine action
                    Note: Not used for Hinge since it uses OCR to find the button

    Returns:
        None
    """
    if dating_app == 'bumble':
        # For Bumble, swipe right to like
        width, height = get_screen_resolution(device)
        swipe_start_x = int(width * 0.2)  # Start at 20% of screen width
        swipe_end_x = int(width * 0.8)    # End at 80% of screen width
        swipe_y = int(height * 0.5)       # Middle of screen height

        # Execute the right swipe
        device.shell(
            f"input swipe {swipe_start_x} {swipe_y} {swipe_end_x} {swipe_y} 300")
    elif dating_app == 'tinder':
        # Tinder like button coordinates
        tap(device, 750, 2050, with_additional_swipe=False)

    # No need to wait here as we'll wait at the beginning of the next profile processing


def send_response_to_story(device, conversation_starter, profile_num, dating_app='hinge', response_match_found=False, response_tap_coordinates=None):
    """Handle the flow of responding to an opened story.

    Args:
        device: The ADB device
        conversation_starter: The text to send as a response
        profile_num: Current profile number for debugging visualization
        dating_app: Optional app type ('hinge' or 'bumble') to optimize OCR settings
        response_match_found: Whether a response match was found during prompt detection
        response_tap_coordinates: Coordinates of the response match if found

    Returns:
        bool: True if response was sent successfully, False otherwise
    """
    # Add delay to ensure UI is fully loaded before looking for comment box
    logger.info("Waiting 1 second for UI to stabilize...")
    time.sleep(1.0)

    # PHASE 1: Find and click comment box
    logger.info("\nPhase 1: Locating comment box...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase1")

    # Extract text and boxes with app-specific OCR settings
    boxes = extract_text_from_image_with_boxes(
        screenshot_path, app_type=dating_app)
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

        # If we have a response match from previous detection, try tapping it
        if response_match_found and response_tap_coordinates:
            logger.info(
                "No comment box found, but we have a response match. Trying to tap on response...")

            # Create visualization without tap target for the current state
            create_visual_debug_overlay(
                screenshot_path,
                boxes=boxes,
                lines=lines,
                paragraphs=paragraphs,
                output_path=f"images/profile_{profile_num}_response_phase1_visual_before_response_tap.png",
                profile_num=profile_num,
                app_name=dating_app
            )

            # Tap on the response match coordinates
            tap_x, tap_y = response_tap_coordinates
            logger.info(
                f"Tapping response match at coordinates: ({tap_x}, {tap_y})")
            tap(device, tap_x, tap_y, double_tap=True)

            # Wait for UI to update after tapping response
            time.sleep(1.5)

            # Take another screenshot and check if comment box appears now
            retry_screenshot_path = capture_screenshot(
                device, f"profile_{profile_num}_response_phase1_retry")

            retry_boxes = extract_text_from_image_with_boxes(
                retry_screenshot_path, app_type=dating_app)

            if not retry_boxes:
                logger.warning("No text boxes found in retry screenshot")
                return False

            retry_lines = group_boxes_into_lines(retry_boxes)
            retry_paragraphs = group_lines_into_paragraphs(retry_lines)

            # Look for comment box again
            retry_comment_box = None
            retry_comment_ratio = 0.0

            for para in retry_paragraphs:
                is_match, ratio, _ = fuzzy_match_text(
                    "Add a comment", para['text'])
                if is_match and ratio > retry_comment_ratio:
                    retry_comment_box = para
                    retry_comment_ratio = ratio

            if not retry_comment_box:
                logger.warning(
                    "Still could not find comment box after tapping response")
                # Create visualization without tap target
                create_visual_debug_overlay(
                    retry_screenshot_path,
                    boxes=retry_boxes,
                    lines=retry_lines,
                    paragraphs=retry_paragraphs,
                    output_path=f"images/profile_{profile_num}_response_phase1_retry_visual.png",
                    profile_num=profile_num,
                    app_name=dating_app
                )
                return False

            # Update our variables to continue with the found comment box
            comment_box = retry_comment_box
            comment_ratio = retry_comment_ratio
            boxes = retry_boxes
            lines = retry_lines
            paragraphs = retry_paragraphs
            screenshot_path = retry_screenshot_path
            logger.info(
                f"Found comment box after tapping response with ratio {comment_ratio:.2f}")
        else:
            # No response match to try, create visualization without tap target
            create_visual_debug_overlay(
                screenshot_path,
                boxes=boxes,
                lines=lines,
                paragraphs=paragraphs,
                output_path=f"images/profile_{profile_num}_response_phase1_visual.png",
                profile_num=profile_num,
                app_name=dating_app
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
        tap_target=(comment_x, comment_y),
        profile_num=profile_num,
        app_name=dating_app
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

    # PHASE 2: Find and click Send Like button in new layout
    logger.info("\nPhase 2: Locating Send Like button...")
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_response_phase2")

    # Extract text and boxes again for new layout with app-specific OCR settings
    boxes = extract_text_from_image_with_boxes(
        screenshot_path, app_type=dating_app)
    if not boxes:
        logger.warning("No text boxes found in post-input screenshot")
        return False

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Find "Send Like" button in new layout
    send_button = None
    send_ratio = 0.0
    for para in paragraphs:
        is_match, ratio, _ = fuzzy_match_text(
            "Send Like", para['text'], threshold=0.7)
        if is_match and ratio > send_ratio:
            send_button = para
            send_ratio = ratio

    if not send_button:
        logger.warning("Could not find Send Like button")
        # Create visualization without tap target
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_response_phase2_visual.png",
            profile_num=profile_num,
            app_name=dating_app
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
        tap_target=(send_x, send_y),
        profile_num=profile_num,
        app_name=dating_app
    )

    # Click Send Like button
    tap(device, send_x, send_y)

    # No need to wait here as we'll wait at the beginning of the next profile processing
    return True


def save_profile_results(profile_num, screenshots, ai_response, add_timestamp=False, app_name=""):
    """Save profile screenshots and AI response in an organized folder structure.

    Args:
        profile_num: The profile number
        screenshots: List of screenshot paths
        ai_response: The AI response dictionary (can be None)
        add_timestamp: Whether to add a timestamp to the response (default: False)
        app_name: Name of the app (hinge, bumble) to include in directory name for separate results

    Returns:
        str: Path to the profile's results directory
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create desktop directory path with timestamp for easy access from host
    # Single parent folder structure with app-specific subfolders
    desktop_dir_name = f"pp_{RUN_TIMESTAMP}"
    desktop_dir = f"/app/desktop/{desktop_dir_name}"

    # Create app-specific subfolder if provided
    if app_name:
        desktop_dir = os.path.join(desktop_dir, app_name.capitalize())

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

    # Create analysis subdirectory ONLY in desktop location for OCR visualizations
    desktop_analysis_dir = os.path.join(desktop_profile_dir, "analysis")

    if not os.path.exists(desktop_analysis_dir):
        os.makedirs(desktop_analysis_dir)

    # Copy screenshots to both profile directories
    for screenshot in screenshots:
        filename = os.path.basename(screenshot)
        # Save to container results directory
        dest_path = os.path.join(screenshots_dir, filename)
        shutil.copy2(screenshot, dest_path)
        # Also save to desktop directory
        desktop_dest_path = os.path.join(desktop_screenshots_dir, filename)
        shutil.copy2(screenshot, desktop_dest_path)

    # Save AI response as JSON, but only to desktop
    desktop_response_path = os.path.join(desktop_profile_dir, "response.json")

    # Copy the AI response to avoid modifying the original
    response_data = ai_response.copy() if ai_response is not None else {}

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_data['timestamp'] = timestamp

    # Save only to desktop directory
    with open(desktop_response_path, 'w') as f:
        json.dump(response_data, f, indent=2)

    logger.info(f"Screenshots saved to container path: {profile_dir}")
    logger.info(
        f"Analysis directory for visualizations: {desktop_analysis_dir}")
    logger.info(
        f"Results and screenshots saved to desktop path: {desktop_profile_dir}")

    return profile_dir


def check_for_end_of_profiles(device, profile_num, dating_app=None):
    """Check if we've reached the end of available profiles.

    Args:
        device: The ADB device
        profile_num: Current profile number for debugging
        dating_app: Optional app type ('hinge', 'bumble', or 'tinder') to optimize OCR settings

    Returns:
        tuple: (bool, str) - (True if end reached, message that was matched)
    """
    # Take a screenshot to check for the message
    screenshot_path = capture_screenshot(
        device, f"profile_{profile_num}_end_check")

    # Extract text and group into paragraphs with app-specific OCR settings
    boxes = extract_text_from_image_with_boxes(
        screenshot_path, app_type=dating_app)
    if not boxes:
        return False, ""

    lines = group_boxes_into_lines(boxes)
    paragraphs = group_lines_into_paragraphs(lines)

    # Messages to check for
    end_messages = [
        "You've seen everyone for now",
        "Try changing your filters so more people match your criteria - or check again later!",
        "Change filters",
        "Review skipped profiles",
        "You're out of free likes for today",
        "You're out of free",
        "likes for today",
        "Hinge+",
        "Send unlimited likes*",
        "See everyone who likes you",
        "Set more dating preferences",
        "Sort all incoming likes",
        "Browse by who's new ön nearby",
        "Get 3 months for $89.99",
        "What a match!",
        "Now you have 24 hours to start chatting.",
        # Tinder-specific end messages
        "There's no one around you",
        "There's no one new around you",
        "Check back later",
        "Something went wrong",
        "It seems we're having technical difficulties",
        "You've reached the end",
        "No one new around you",
        "Upgrade to Tinder Plus",
        "Get Tinder Plus",
        "You've got a secret admirer!"
    ]

    # Check each paragraph against each message
    for para in paragraphs:
        for message in end_messages:
            is_match, ratio, _ = fuzzy_match_text(
                message, para['text'], threshold=0.7)
            if is_match:
                logger.info(
                    f"Found end message: '{message}' with confidence {ratio:.2f}")
                # Create visualization of the match
                create_visual_debug_overlay(
                    screenshot_path,
                    boxes=boxes,
                    lines=lines,
                    paragraphs=paragraphs,
                    output_path=f"images/profile_{profile_num}_end_detected_visual.png",
                    profile_num=profile_num,
                    app_name=dating_app
                )
                return True, message

    return False, ""


def launch_app(device, package_name, app_name):
    """Launch an app using its package name.

    Args:
        device: The ADB device
        package_name: The package name of the app
        app_name: The human-readable name of the app for logging

    Returns:
        None
    """
    try:
        logger.info(f"Launching {app_name} app")
        device.shell(f"monkey -p {package_name} 1")
        time.sleep(2)  # Reduced wait time from 5 to 2 seconds
        logger.info(f"{app_name} app launched successfully")
    except Exception as e:
        logger.error(f"Error launching {app_name} app: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level


def close_app(device, package_name, app_name):
    """Close an app using its package name.

    Args:
        device: The ADB device
        package_name: The package name of the app
        app_name: The human-readable name of the app for logging

    Returns:
        None
    """
    try:
        logger.info(f"Closing {app_name} app")
        device.shell(f"am force-stop {package_name}")
        time.sleep(2)  # Reduced wait time from 5 to 2 seconds
        logger.info(f"{app_name} app closed successfully")
    except Exception as e:
        logger.error(f"Error closing {app_name} app: {e}")
        logger.debug("", exc_info=True)  # Log full traceback at debug level


def open_hinge(device):
    """Open the Hinge dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    launch_app(device, "co.hinge.app", "Hinge")


def open_bumble(device):
    """Open the Bumble dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    launch_app(device, "com.bumble.app", "Bumble")


def open_tinder(device):
    """Open the Tinder dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    launch_app(device, "com.tinder", "Tinder")


def close_hinge(device):
    """Close the Hinge dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    close_app(device, "co.hinge.app", "Hinge")


def close_bumble(device):
    """Close the Bumble dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    close_app(device, "com.bumble.app", "Bumble")


def close_tinder(device):
    """Close the Tinder dating app.

    Args:
        device: The ADB device

    Returns:
        None
    """
    close_app(device, "com.tinder", "Tinder")


def check_for_bumble_advertisement(device, profile_num):
    """Check if there's a Bumble advertisement that requires dismissal.

    Detects advertisements by looking for specific text phrases and dismisses them:
    - For regular ads (premium features, etc.), swipes right to dismiss
    - For specific ads about likes or "second time's a charm", taps the X button

    Args:
        device: The ADB device
        profile_num: Current profile number for debugging

    Returns:
        bool: True if advertisement was detected and dismissed, False otherwise
    """
    try:
        # Take a screenshot to check for the advertisement
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_ad_check")

        # Extract text and group into paragraphs
        boxes = extract_text_from_image_with_boxes(
            screenshot_path, app_type='bumble')
        if not boxes:
            return False

        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Regular advertisement indicators - to be dismissed with swipe right
        regular_ad_indicators = [
            "Like what you're seeing?",
            "Bumble premium",
            "Premium features",
            "Upgrade now",
            "Upgrade to Bumble Premium"
        ]

        # X-button advertisement indicators - to be dismissed with tap on X button
        x_button_ad_indicators = [
            "You've already liked these bees, why not use Compliments to help stand out?",
            "Compliments to help stand out?",
            "You've already liked these bees, why not use",
            "Second time's a charm",
            "Be seen more for less, with 1 spotlight for $0.99",
            "For 30 minutes we'll bump you to the front of the linew, here more eyes will be on you. Own it.",
            "Get Spotlight for $0.99"
        ]

        # Check for X-button advertisements first
        x_button_ad_detected = False
        matched_x_button_indicator = ""

        for para in paragraphs:
            for indicator in x_button_ad_indicators:
                is_match, ratio, _ = fuzzy_match_text(
                    indicator, para['text'], threshold=0.7)
                if is_match:
                    x_button_ad_detected = True
                    matched_x_button_indicator = indicator
                    logger.info(
                        f"Detected X-button Bumble advertisement: '{matched_x_button_indicator}' with confidence {ratio:.2f}")

                    # Create visualization of the match
                    create_visual_debug_overlay(
                        screenshot_path,
                        boxes=boxes,
                        lines=lines,
                        paragraphs=paragraphs,
                        output_path=f"images/profile_{profile_num}_ad_detected_visual.png",
                        profile_num=profile_num,
                        app_name="bumble"
                    )
                    break

            if x_button_ad_detected:
                break

        # Check for regular advertisements if no X-button ad detected
        regular_ad_detected = False
        matched_regular_indicator = ""

        if not x_button_ad_detected:
            for para in paragraphs:
                for indicator in regular_ad_indicators:
                    is_match, ratio, _ = fuzzy_match_text(
                        indicator, para['text'], threshold=0.7)
                    if is_match:
                        regular_ad_detected = True
                        matched_regular_indicator = indicator
                        logger.info(
                            f"Detected regular Bumble advertisement: '{matched_regular_indicator}' with confidence {ratio:.2f}")

                        # Create visualization of the match
                        create_visual_debug_overlay(
                            screenshot_path,
                            boxes=boxes,
                            lines=lines,
                            paragraphs=paragraphs,
                            output_path=f"images/profile_{profile_num}_ad_detected_visual.png",
                            profile_num=profile_num,
                            app_name="bumble"
                        )
                        break

                if regular_ad_detected:
                    break

        # Handle X-button advertisements by tapping the X button at (70, 200)
        if x_button_ad_detected:
            logger.info(
                f"Dismissing Bumble advertisement by tapping X button at (70, 200)")
            tap(device, 70, 200)

            # Wait for the next profile to load
            time.sleep(2.0)
            return True

        # Handle regular advertisements with a swipe left gesture (changed from right)
        elif regular_ad_detected:
            logger.info(f"Dismissing Bumble advertisement by swiping left")
            width, height = get_screen_resolution(device)

            # Horizontal swipe from right to left (changed from left to right)
            swipe_start_x = int(width * 0.8)  # Start at 80% of screen width
            swipe_end_x = int(width * 0.2)    # End at 20% of screen width
            swipe_y = int(height * 0.5)       # Middle of screen height

            # Execute the left swipe
            device.shell(
                f"input swipe {swipe_start_x} {swipe_y} {swipe_end_x} {swipe_y} 300")

            # Give time for the ad to be dismissed
            time.sleep(2)
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking for Bumble advertisement: {e}")
        logger.debug("", exc_info=True)
        return False


def generate_tinder_reply_from_screenshots(screenshots, format_txt_path):
    """
    Generate a contextually appropriate analysis for a Tinder profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to tinderFormat.txt describing profile structure

    Returns:
        dict: Contains the prompt-response pair, generated response, and screenshot index:
        {
            "prompt": str,      # The exact prompt text being responded to
            "response": str,    # The user's response to the prompt
            "conversation_starter": str,  # The generated conversation starter
            "screenshot_index": int,  # 0-based index of screenshot containing prompt/response
        }
    """
    # Read the content of our context files
    context_files = read_context_files({
        'format': format_txt_path
    })

    if not context_files:
        return None

    # Convert screenshots to base64 for API transmission
    screenshot_data = prepare_screenshots_for_api(screenshots)
    if not screenshot_data:
        return None

    system_prompt = f"""{DATING_APP_INTRO}
    You are analyzing a set of 9 screenshots of a Tinder dating profile which can contain:
    - Profile photos
    - Name and age
    - Location information
    - Bio text
    - Interests/passions
    - Other optional profile information
    
    Each of these elements tells you something about the woman. Your goal is to analyze her profile comprehensively to see if she is desirable or undesirable.

    REFERENCE FILES CONTENT:
    format.txt:
    {context_files['format']}

    STEP 1: READ AND UNDERSTAND THE CONTEXT
    Read format.txt to understand the profile layout

    {IDENTIFY_MAIN_PERSON}

    STEP 3: ORGANIZE ELEMENTS INTO BUCKETS
    For each screenshot, group elements into these buckets:
    a. Photos
       - Is the main person in this photo?
       - What is she doing? With whom?
       - What does this reveal about her?
    b. About Me text
       - What does she explicitly state about herself?
       
    c. Attributes (basic info)
       - What factual information does she share?
       - How do these fit together to create a lifestyle?
    d. Looking For
       - What relationship type is she seeking?
       - What qualities matter to her?
    e. Interests
       - What activities does she enjoy?
       - Are there themes among interests?
    f. Prompt/Response pairs
       - Which prompts did she choose?
       - What do her responses reveal?

    {UNDESIRABLE_TRAITS_CHECK}
    
    {ENDING_DESIREABILITY_ONLY}"""

    # User message - just the specific task
    user_message = """Please analyze these Tinder profile screenshots and return the requested data as instructed."""

    # Make the API call to Claude
    logger.info("Making API call to Claude...")

    response = call_claude_api(
        system_prompt=system_prompt,
        user_message=user_message,
        screenshots_data=screenshot_data
    )

    if not response:
        return None

    # Parse the response
    result = parse_claude_json_response(response)
    if not result:
        return None

    return {
        "prompt": result.get("prompt", ""),
        "response": result.get("response", ""),
        "conversation_starter": result.get("conversation_starter", ""),
        "screenshot_index": result.get("screenshot_index", 0)
    }


def check_for_super_like_popup(device, profile_num, dating_app=None):
    """Check if a 'Send Super Like' popup appeared after liking a profile.

    Args:
        device: The ADB device
        profile_num: Current profile number for debugging
        dating_app: Optional app type for OCR settings

    Returns:
        bool: True if popup was detected and dismissed, False otherwise
    """
    try:
        # Wait 1.0 second for popup to appear
        logger.info("Waiting 1.0 second for possible Super Like popup")
        time.sleep(1.0)

        # Take a screenshot to check for the popup
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_super_like_check")

        # Extract text using OCR
        boxes = extract_text_from_image_with_boxes(
            screenshot_path, app_type=dating_app)
        if not boxes:
            logger.info("No text found in screenshot, no popup detected")
            return False

        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Create visualization for debugging
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_super_like_check_visual.png"
        )

        # Check for "Send Super Like" text in paragraphs
        super_like_detected = False
        no_thanks_button = None

        # Search for "Send Super Like" and "No thanks" text
        for para in paragraphs:
            # Check for "Send Super Like" text
            is_match_super, ratio_super, _ = fuzzy_match_text(
                "Send Super Like", para['text'], threshold=0.7)
            if is_match_super:
                logger.info(
                    f"Detected 'Send Super Like' popup with ratio {ratio_super:.2f}")
                super_like_detected = True

            # Check for "No thanks" button
            is_match_no, ratio_no, _ = fuzzy_match_text(
                "No thanks", para['text'], threshold=0.7)
            if is_match_no:
                logger.info(
                    f"Found 'No thanks' button with ratio {ratio_no:.2f}")
                no_thanks_button = para

        # If Super Like popup detected and No thanks button found, click it
        if super_like_detected and no_thanks_button:
            logger.info(
                "Super Like popup detected, clicking 'No thanks' button")

            # Calculate tap coordinates for the No thanks button
            boxes_no = no_thanks_button['boxes']
            min_x = min(box['box'][0] for box in boxes_no)
            max_x = max(box['box'][0] + box['box'][2] for box in boxes_no)
            min_y = min(box['box'][1] for box in boxes_no)
            max_y = max(box['box'][1] + box['box'][3] for box in boxes_no)

            no_thanks_x = (min_x + max_x) // 2
            no_thanks_y = (min_y + max_y) // 2

            # Create visualization with tap target
            create_visual_debug_overlay(
                screenshot_path,
                boxes=boxes,
                lines=lines,
                paragraphs=paragraphs,
                output_path=f"images/profile_{profile_num}_super_like_dismiss_visual.png",
                tap_target=(no_thanks_x, no_thanks_y)
            )

            # Tap the No thanks button
            tap(device, no_thanks_x, no_thanks_y)
            time.sleep(1.0)  # Wait for popup to dismiss
            return True

        # Alternative approach: if we detect Super Like but not the button,
        # try a generic bottom-left tap which is typically where "No thanks" is
        if super_like_detected and not no_thanks_button:
            logger.info(
                "Super Like popup detected but 'No thanks' button not found, trying generic location")
            # Tap at a common location for "No thanks" (bottom-left area)
            tap(device, 300, 1600)
            time.sleep(1.0)
            return True

        logger.info("No Super Like popup detected")
        return False

    except Exception as e:
        logger.error(f"Error checking for Super Like popup: {e}")
        logger.debug("", exc_info=True)
        return False


def check_for_tinder_advertisement(device, profile_num):
    """Check if there's a Tinder advertisement about profile boosting that requires dismissal.

    Detects specific advertisements and dismisses them appropriately:
    - 'Your new pic is everything' ad: dismissed by swiping left
    - 'Be Seen' boosts ad: dismissed by tapping X button at top left
    - 'Passion in common' ad: dismissed by tapping at 550x1675
    """
    try:
        # Take a screenshot to check for the advertisement
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_tinder_ad_check")

        # Extract text and group into paragraphs
        boxes = extract_text_from_image_with_boxes(
            screenshot_path, app_type='tinder')
        if not boxes:
            return False

        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Create visualization for debugging
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_tinder_ad_check_visual.png",
            profile_num=profile_num,
            app_name="tinder"
        )

        # X-button advertisement indicators - to be dismissed with tap on X button
        x_button_ad_indicators = [
            "Be Seen",
            "Be a top profile in your area for 30 minutes to get more matches",
            "Be a top profile in your area for 30 minutes"
            "Get more matches"
        ]

        # Regular advertisement indicators - to be dismissed with swipe left
        regular_ad_indicators = [
            "Your new pic is everything",
            "Make sure your next crush date or boo sees it",
            "Make sure your next crush, date or boo sees it - Boost your profile to increase your chances of meeting them.",
            "BOOST ME"
        ]

        # Passion in common ad indicators - to be dismissed with tap at 550x1675
        passion_ad_indicators = [
            "You have a Passion in common with",
            "Send a Super Like to",
            "increase your chance to match",
            "Send Super Like",
            "No Thanks"
        ]

        # First check for X-button advertisements
        x_button_ad_detected = False
        matched_x_button_indicator = ""

        for para in paragraphs:
            for indicator in x_button_ad_indicators:
                is_match, ratio, _ = fuzzy_match_text(
                    indicator, para['text'], threshold=0.7)
                if is_match:
                    logger.info(
                        f"Detected Tinder X-button ad indicator '{indicator}' with ratio {ratio:.2f}")
                    x_button_ad_detected = True
                    matched_x_button_indicator = indicator
                    break
            if x_button_ad_detected:
                break

        # Check for regular advertisements if no X-button ad detected
        regular_ad_detected = False
        matched_regular_indicator = ""

        if not x_button_ad_detected:
            for para in paragraphs:
                for indicator in regular_ad_indicators:
                    is_match, ratio, _ = fuzzy_match_text(
                        indicator, para['text'], threshold=0.7)
                    if is_match:
                        logger.info(
                            f"Detected Tinder regular ad indicator '{indicator}' with ratio {ratio:.2f}")
                        regular_ad_detected = True
                        matched_regular_indicator = indicator
                        break
                if regular_ad_detected:
                    break

        # Check for passion in common ad if no other ads detected
        passion_ad_detected = False
        matched_passion_indicator = ""

        if not x_button_ad_detected and not regular_ad_detected:
            for para in paragraphs:
                for indicator in passion_ad_indicators:
                    is_match, ratio, _ = fuzzy_match_text(
                        indicator, para['text'], threshold=0.7)
                    if is_match:
                        logger.info(
                            f"Detected Tinder passion ad indicator '{indicator}' with ratio {ratio:.2f}")
                        passion_ad_detected = True
                        matched_passion_indicator = indicator
                        break
                if passion_ad_detected:
                    break

        # Handle X-button advertisements by tapping the X button at (75, 190)
        if x_button_ad_detected:
            logger.info(
                f"Dismissing Tinder 'Be Seen' advertisement by tapping X button at (75, 190)")
            tap(device, 75, 190)

            # Wait for the ad to be dismissed
            time.sleep(2.0)
            return True

        # Handle regular advertisements with a swipe left
        elif regular_ad_detected:
            logger.info(
                "Tinder 'Boost' advertisement detected - swiping left to dismiss")
            # Swipe left to dismiss advertisement
            # Start from mid-right of screen and swipe to mid-left
            width, height = get_screen_resolution(device)
            start_x = int(width * 0.8)
            end_x = int(width * 0.2)
            mid_y = int(height * 0.5)

            # Execute the swipe
            device.shell(f"input swipe {start_x} {mid_y} {end_x} {mid_y} 300")

            # Wait for the ad to be dismissed
            time.sleep(2.0)
            return True

        # Handle passion in common ad by tapping at 550x1675
        elif passion_ad_detected:
            logger.info(
                "Tinder 'Passion in common' advertisement detected - tapping at (550, 1675) to dismiss")
            tap(device, 550, 1675)

            # Wait for the ad to be dismissed
            time.sleep(2.0)
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking for Tinder advertisement: {e}")
        logger.debug("", exc_info=True)
        return False


def check_for_hinge_rose_like_popup(device, profile_num):
    """Check if a 'Send Rose' / 'Send Like Anyway' popup appeared after a profile loads on Hinge.

    Args:
        device: The ADB device
        profile_num: Current profile number for debugging

    Returns:
        bool: True if popup was detected and 'Send Like Anyway' was clicked, False otherwise
    """
    try:
        # Take a screenshot to check for the popup
        screenshot_path = capture_screenshot(
            device, f"profile_{profile_num}_rose_like_check")

        # Extract text using OCR
        boxes = extract_text_from_image_with_boxes(
            screenshot_path, app_type='hinge')
        if not boxes:
            logger.info("No text found in screenshot, no popup detected")
            return False

        lines = group_boxes_into_lines(boxes)
        paragraphs = group_lines_into_paragraphs(lines)

        # Create visualization for debugging
        create_visual_debug_overlay(
            screenshot_path,
            boxes=boxes,
            lines=lines,
            paragraphs=paragraphs,
            output_path=f"images/profile_{profile_num}_rose_like_check_visual.png",
            profile_num=profile_num,
            app_name="hinge"
        )

        # Check for "Send Rose" or "Send Like Anyway" text in paragraphs
        rose_detected = False
        like_anyway_button = None

        # Search for both buttons in the text
        for para in paragraphs:
            # Check for "Send Rose" text
            is_match_rose, ratio_rose, _ = fuzzy_match_text(
                "Send Rose", para['text'], threshold=0.7)
            if is_match_rose:
                logger.info(
                    f"Detected 'Send Rose' button with ratio {ratio_rose:.2f}")
                rose_detected = True

            # Check for "Send Like Anyway" button
            is_match_anyway, ratio_anyway, _ = fuzzy_match_text(
                "Send Like Anyway", para['text'], threshold=0.7)
            if is_match_anyway:
                logger.info(
                    f"Found 'Send Like Anyway' button with ratio {ratio_anyway:.2f}")
                like_anyway_button = para

        # If we see 'Send Rose' and 'Send Like Anyway', click 'Send Like Anyway'
        if (rose_detected or like_anyway_button) and like_anyway_button:
            logger.info(
                "Rose/Like popup detected, clicking 'Send Like Anyway' button")

            # Calculate tap coordinates for the 'Send Like Anyway' button
            boxes_anyway = like_anyway_button['boxes']
            min_x = min(box['box'][0] for box in boxes_anyway)
            max_x = max(box['box'][0] + box['box'][2] for box in boxes_anyway)
            min_y = min(box['box'][1] for box in boxes_anyway)
            max_y = max(box['box'][1] + box['box'][3] for box in boxes_anyway)

            like_anyway_x = (min_x + max_x) // 2
            like_anyway_y = (min_y + max_y) // 2

            # Create visualization with tap target
            create_visual_debug_overlay(
                screenshot_path,
                boxes=boxes,
                lines=lines,
                paragraphs=paragraphs,
                output_path=f"images/profile_{profile_num}_rose_like_click_visual.png",
                tap_target=(like_anyway_x, like_anyway_y),
                profile_num=profile_num,
                app_name="hinge"
            )

            # Tap the 'Send Like Anyway' button
            tap(device, like_anyway_x, like_anyway_y)

            # Wait 3 seconds for the next profile to load
            logger.info("Waiting 3.0 seconds for next profile to load...")
            time.sleep(3.0)

            return True

        return False
    except Exception as e:
        logger.error(f"Error in check_for_hinge_rose_like_popup: {e}")
        logger.debug("", exc_info=True)
        return False
