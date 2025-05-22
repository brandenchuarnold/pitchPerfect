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


def call_claude_api(system_prompt, user_message, screenshots_data, model="claude-4-sonnet-latest", max_tokens=1000, temperature=1):
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

COMPILE_CHARACTERISTICS = """STEP 6: COMPILE CHARACTERISTICS
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
   better than many assumptions."""

STORY_BASED_EXPLANATIONS = """STEP 7: GENERATE STORY-BASED EXPLANATIONS
For each prompt/response pair (there will be exactly 3 pairs):
1. For each of the 6 photos:
   a. Identify THREE distinct elements in the photo:
      - Setting/location (where the photo was taken)
      - Activity/action (what she's doing in the photo)
      - Notable items (specific clothes, accessories, equipment, etc.)
   b. For each distinct element, create a story explanation that:
      - Treats the prompt/response as representing a real story from her life
      - Uses the photo element as a clue to understand this story
      - CONSISTS OF TWO PARTS: 
         1. A QUESTION/STATEMENT (8-10 WORDS ONLY)
         2. CONTEXT EXPLANATION (10-15 WORDS ONLY)
      - ALWAYS include her name immediately after referencing her photo in the context explanation (e.g., "I see in your hiking photo, Emma, that you enjoy...")
      - IMPORTANT: The question/statement MUST COME FIRST, followed by the context explanation
      - Format: "[Question/statement about the story behind her prompt/response?] [Brief explanation connecting to the photo element, followed by her name]"
      - CRITICAL: The question/statement MUST be able to stand completely on its own without requiring the context explanation to be understood
      - CRITICAL: NEVER use information from one prompt to explain another prompt - each prompt is a completely separate story
      - Examples:
        * Prompt: "Together we could"
          Response: "Play Civ6 although not good at it" 
          Photo element: Woman at a concert
          Story explanation: "Do you agree that gaming and concerts both hype you up? I see that you went to a concert in your photo, Sarah."
          WHY THIS WORKS: This makes a thoughtful connection between gaming (Civ6 from her prompt) and concerts (from her photo) by identifying the shared element of "hype" or excitement they both generate. Rather than just noting indoor vs. outdoor activities, it identifies a specific emotional experience common to both. The question stands on its own and would make sense even without context, while the second part simply provides evidence that you've looked at her photos. The statement is 10 words, and the context is 12 words, making it concise yet meaningful.
        
        * "Do exercise and relaxation create perfect weekends? I notice you're running in your beach photo, Jessica, which creates nice balance."
          WHY THIS WORKS: This takes the concrete element of her running in the photo and connects it to her prompt about a Sunday routine involving exercise and relaxation. It suggests her routine isn't random but a deliberate lifestyle choice reflecting her understanding that balance is important. The photo provides evidence that she actually does run regularly, making this a genuine rather than assumed connection. The statement is 7 words, and the context is 13 words.
        
        * "Does proper gear make all outdoor activities better? I spotted quality running shoes in your photo, Emma, they look well-used."
          WHY THIS WORKS: This focuses on a specific item (running shoes) and connects it to her routine that includes running. Instead of making generic assumptions about her personality, it creates a story where she's thoughtful about her equipment choices, suggesting she's experienced enough as a runner to care about proper gear. The worn shoes indicate this isn't just a casual hobby but something she does regularly. The statement is 8 words, and the context is 13 words.
        
        * "Is the beach your favorite recharging spot? You seem comfortable by the ocean in your photos, Alexis."
          WHY THIS WORKS: This observation connects the setting of her photo (beach) with the concept of relaxation/recharging, which ties to many prompt responses about personal time or self-care. Rather than assuming she loves beaches, it frames it as a genuine question about whether this is one of multiple places she goes to unwind, creating space for her to share more about her relaxation preferences. The statement is 6 words, and the context is 11 words.
        
        * Prompt: "I get along best with people who" 
          Response: "Are kind and adventurous"
          Photo element: Woman hiking on a mountain trail
          Story explanation: "Do hikers value kindness and new experiences? You're hiking on trails in your mountain photos, Nicole."
          WHY THIS WORKS: This makes a logical connection between her stated values (kindness and adventure) and her visible activity (hiking). It suggests her preference for these traits in others stems from her own experiences in hiking communities, where people tend to help each other and embrace new experiences. Instead of making abstract assumptions about her personality, it ties her stated values to concrete experiences visible in her photos. The statement is 7 words, and the context is 12 words.
        
        * Prompt: "Typical Saturday" 
          Response: "You can find me at a farmer's market"
          Photo element: Woman dressed up for a night out
          Story explanation: "Are farmers markets and nights out both special occasions? You make time for dressing up to go out in your photo, Rachel."
          WHY THIS WORKS: This creates a thoughtful connection between two activities - farmer's market visits and getting dressed up for nights out - viewing both as special occasions she values. Instead of contrasting them, it identifies a pattern of making time for different types of enjoyable experiences, suggesting she's someone who deliberately structures her lifestyle around meaningful activities. The observation about "special occasions" creates a story about how she prioritizes experiences in her life. The statement is 9 words, and the context is 14 words.

2. For each prompt/response pair, you will need to generate story explanations based on elements identified in her photos.
   
3. Each story explanation MUST:
   - FOLLOW THE EXACT STRUCTURE: Question/statement (8-10 words) FIRST, then context explanation (10-15 words)
   - Ensure the question/statement STANDS COMPLETELY ON ITS OWN - it must be fully understandable without the context explanation
   - The context explanation should only provide background on how you connected to the photo, not essential information needed to understand the question
   - Be based on a concrete, visible element in the photo (not assumptions about her personality)
   - Connect logically to her prompt/response for THIS PROMPT ONLY (never use other prompts as context)
   - Use natural, conversational language (avoid marketing-speak or overly formal language)
   - Invite her to share more about her experiences or thoughts
   - Include a specific reference to the photo element that inspired your question/statement
   - ALWAYS include her name immediately after referencing her photo (this creates a personal connection)
   - Be specific enough to show you paid attention to details
   - Create a flattering but realistic image of her life
   - Avoid making assumptions that aren't supported by the photos or prompt/response
   - CONTAIN GENUINE CONTRAST when discussing "balance" (e.g., indoor/outdoor, work/play, active/relaxing)

4. When creating story explanations:
   - Focus on her real experiences visible in the photos
   - Avoid generic personality traits like "adventurous" or "fun-loving"
   - Don't focus on her appearance (focus on what she's doing, not how she looks)
   - Create connections that feel natural, not forced
   - Invite her to elaborate on the story behind her prompt/response
   - Use the photo elements as clues to understand the context of her life
   - Always use her name when referencing her photos (this psychologically increases her interest)
   - Imagine each prompt/response represents a real experience she's had
   - Treat the conversation as a chance to learn more about her stories
   - Make her feel understood and interesting, not analyzed or judged
   - NEVER use information from one prompt to explain another prompt

5. CRITICAL: Understanding the creative process behind good story explanations:
   - The best story explanations identify patterns between the prompt and photo element that reveal something about her real life
   - For example, connecting beer in a bar photo with a prompt about ordering margaritas reveals that she likely tailors her drink choices to different occasions
   - Look for contrasts that tell a story: if she mentions running on Sundays but is dressed up in a photo, this suggests she balances different aspects of her life
   - Notice specific items (running shoes, specific clothing, equipment) that indicate regular participation in activities mentioned in her prompts
   - The goal is to imagine the actual story behind each prompt - not just what she wrote, but the real-life experiences that led her to write it
   - Use visible elements in photos as evidence to support your interpretation of these stories
   - Create explanations that feel like natural observations rather than reaching or forced connections
   - When you connect photo elements to prompts, explain why that connection makes sense in a way that feels conversational and insightful
   - BAD EXAMPLE: "I notice you're into gaming based on your Sunday routine and Civ6 mention." - This is bad because it combines information from multiple prompts instead of treating each prompt as its own story

6. FOR THE FIRST PROMPT/RESPONSE PAIR:
   a. For each of the 6 photos, identify 3 DISTINCT ELEMENTS (distinct from each other within this photo)
   b. For each element, create a story explanation following the format above
   c. This will generate 18 story explanations for this prompt (3 elements × 6 photos = 18)
   d. Count the number of story explanations for this first prompt. Do you have 18? If not, go back and continue finding elements and creating story explanations until you reach 18.

7. FOR THE SECOND PROMPT/RESPONSE PAIR:
   a. For each of the 6 photos, identify 3 DISTINCT ELEMENTS (distinct from each other within this photo)
   b. NOTE: These elements CAN be the same ones you used for the first prompt - they only need to be distinct within each photo's analysis for this prompt
   c. For each element, create a story explanation following the format above
   d. This will generate 18 story explanations for this prompt (3 elements × 6 photos = 18)
   e. Count the number of story explanations for this second prompt. Do you have 18? If not, go back and continue finding elements and creating story explanations until you reach 18.

8. FOR THE THIRD PROMPT/RESPONSE PAIR:
   a. For each of the 6 photos, identify 3 DISTINCT ELEMENTS (distinct from each other within this photo)
   b. NOTE: These elements CAN be the same ones you used for the first or second prompts - they only need to be distinct within each photo's analysis for this prompt
   c. For each element, create a story explanation following the format above
   d. This will generate 18 story explanations for this prompt (3 elements × 6 photos = 18)
   e. Count the number of story explanations for this third prompt. Do you have 18? If not, go back and continue finding elements and creating story explanations until you reach 18.

9. You should now have:
   - 18 story explanations for the first prompt
   - 18 story explanations for the second prompt
   - 18 story explanations for the third prompt
   - TOTAL: 54 story explanations

10. Review all story explanations to ensure:
   - Each one is exactly 8-10 words for the question/statement
   - Each one is exactly 10-15 words for the context explanation
   - The question/statement ALWAYS comes FIRST
   - The context explanation ALWAYS comes SECOND
   - No information from other prompts is used to explain the current prompt"""

EVALUATE_STORY_EXPLANATIONS = """STEP 8: EVALUATE STORY EXPLANATIONS
For each of the 54 story explanations (18 per prompt/response pair):
1. Conduct a simulated conversation:
   a. Message 1: Her prompt/response as the first message
   b. Message 2: Your story explanation as the response
   c. Message 3: Imagine her most likely reply based on her profile characteristics
   d. Message 4: Your natural follow-up to continue the conversation
   e. Message 5: Her second response
2. Evaluate based on these criteria:
   - Is the structure correct? (Question/statement FIRST, context explanation SECOND)
   - Can the question/statement STAND COMPLETELY ON ITS OWN without requiring the context explanation?
   - Is the question/statement exactly 8-10 words?
   - Is the context explanation exactly 10-15 words?
   - Does the context explanation only provide background, not essential information?
   - Does it avoid using information from other prompts?
   - How natural does this conversation flow?
   - Would this conversation be enjoyable for both parties?
   - Does it invite her to share her experiences or thoughts?
   - Does it avoid sounding like marketing copy?
   - Is it free of unnecessary qualifiers and drama?
   - Is it easy for her to respond without requiring complex thinking?
   - Does it show genuine interest in her stories?
   - Does it respect her intelligence without being pretentious?
   - Does it include her name after referencing her photo?
   - Does it present genuine contrast when discussing balance (e.g., indoor/outdoor activities)?
3. Rank each conversation on a scale of 1-10 for overall quality. Keep in mind if the story explanation:
   - Has a question/statement that STANDS COMPLETELY ON ITS OWN without requiring context
   - Uses the most natural, everyday language
   - Gets straight to the point
   - Shows interest through specificity, not enthusiasm
   - Creates easy conversation flow
   - Respects the complexity of her personality
   - Creates positive imagery or stories in her mind
   - Feels like a natural reaction to her prompt/response
   - Makes a connection that feels genuine, not forced
   - Includes her name after photo references
   - Avoids sounding "sales-y" or overly enthusiastic
   - Avoids making unfounded assumptions about her
   - Has the CORRECT STRUCTURE (question/statement first, context second)
   - Has the CORRECT WORD COUNT (8-10 words for question/statement, 10-15 words for context)
   - DOES NOT use information from other prompts
4. Count the number of ratings you have. Are there 54? If not, you need to go back and rate all the story explanations you already have.
5. For each prompt/response pair, select the single best story explanation ranked highest by you (3 total, 1 per prompt).

Examples of good story explanation evaluations:

GOOD STORY EXPLANATION EVALUATION:
- Prompt: "What I order for the table"
  Response: "Margaritas"
  Photo element: Woman drinking a beer at a bar
  Story explanation: "Do you enjoy different drinks for different occasions? I see you enjoying a beer in your bar photo, Emma."
  Simulated conversation:
  * Her: "What I order for the table: Margaritas"
  * You: "Do you enjoy different drinks for different occasions? I see you enjoying a beer in your bar photo, Emma."
  * Her: "Absolutely! Margaritas when I'm out with friends, beer for casual nights."
  * You: "I'm the same way! What's your go-to spot for margaritas in Madison?"
  * Her: "I love Canteen for margaritas! Their spicy ones are amazing. Have you been there?"
  Analysis:
  - Structure: ✓ Question first (8 words), context second (10 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✓ Conversation develops organically
  - Enjoyable: ✓ Light-hearted topic about preferences
  - Invites sharing: ✓ She elaborates on her drinking habits
  - Language: ✓ Casual, everyday speech patterns
  - Name usage: ✓ Includes her name after photo reference
  - Ease of response: ✓ Simple topic she can easily engage with
  - Question stands alone: ✓ The question about enjoying different drinks makes complete sense without the context
  - Genuine interest: ✓ Shows interest in her preferences
  - Overall: 9/10 - Creates a natural conversation about her real habits

- Prompt: "Typical Sunday"
  Response: "Coffee, then run, then couch"
  Photo element: Woman running in a race
  Story explanation: "Do exercise and relaxation create perfect weekends? I notice you're running in your race photo, Sarah, which creates nice balance."
  Simulated conversation:
  * Her: "Typical Sunday: Coffee, then run, then couch"
  * You: "Do exercise and relaxation create perfect weekends? I notice you're running in your race photo, Sarah, which creates nice balance."
  * Her: "Absolutely! I need that balance - the run gives me energy and the couch time helps me recharge for the week."
  * You: "I find that balance important too. Do you have a favorite running route in Madison?"
  * Her: "I love running around the Arboretum! It's beautiful year-round. Do you run too?"
  Analysis:
  - Structure: ✓ Question first (7 words), context second (13 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✓ Builds on her routine in a thoughtful way
  - Enjoyable: ✓ Positive topic about lifestyle choices
  - Invites sharing: ✓ She explains her philosophy about balance
  - Language: ✓ Conversational and relatable
  - Name usage: ✓ Includes her name after photo reference
  - Ease of response: ✓ Connects to something she clearly values
  - Genuine interest: ✓ Shows understanding of her lifestyle
  - Overall: 8/10 - Creates meaningful conversation about her routine

- Prompt: "I get along best with people who" 
  Response: "Are kind and adventurous"
  Photo element: Woman hiking on a mountain trail
  Story explanation: "Do hikers value kindness and new experiences? You're hiking on trails in your mountain photos, Nicole."
  Simulated conversation:
  * Her: "I get along best with people who: Are kind and adventurous"
  * You: "Do hikers value kindness and new experiences? You're hiking on trails in your mountain photos, Nicole."
  * Her: "That's actually spot on! Most of my hiking friends are the kindest people I know, and we're always planning new adventures together."
  * You: "Have you done any hiking around Madison? I've heard Devil's Lake has some great trails."
  * Her: "Yes! Devil's Lake is one of my favorites - I try to go at least once each season. The East Bluff trail has the best views. Do you hike too?"
  Analysis:
  - Structure: ✓ Question first (7 words), context second (12 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✓ Makes a concrete observation without exaggeration
  - Enjoyable: ✓ Positive but realistic connection between activities and values
  - Invites sharing: ✓ She confirms the observation and adds personal context
  - Language: ✓ Simple, everyday speech without marketing terms
  - Name usage: ✓ Includes her name after photo reference
  - Ease of response: ✓ Straightforward observation she can easily respond to
  - Genuine interest: ✓ Shows understanding of how activities reflect values
  - Overall: 9/10 - Creates natural conversation about real interests and values

- Prompt: "Typical Saturday" 
  Response: "You can find me at a farmer's market"
  Photo element: Woman dressed up for a night out
  Story explanation: "Are farmers markets and nights out both special occasions? You make time for dressing up to go out in your photo, Rachel."
  Simulated conversation:
  * Her: "Typical Saturday: You can find me at a farmer's market"
  * You: "Are farmers markets and nights out both special occasions? You make time for dressing up to go out in your photo, Rachel."
  * Her: "Yes! I'm big on having regular things to look forward to - whether it's picking up fresh produce or getting dressed up for dinner with friends."
  * You: "Which farmer's market do you usually go to? I've been to the one on the Square a few times."
  * Her: "The Capitol Square one is my favorite! I love getting Stella's spicy cheese bread first thing - have you tried it? It's a Madison must-have."
  Analysis:
  - Structure: ✓ Question first (9 words), context second (13 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✓ Makes a direct observation about her activities
  - Enjoyable: ✓ Positive focus on her lifestyle choices without exaggeration
  - Invites sharing: ✓ She elaborates on her philosophy about special occasions
  - Language: ✓ Straightforward and conversational
  - Name usage: ✓ Includes her name after photo reference
  - Ease of response: ✓ Simple observation she can easily confirm or add to
  - Genuine interest: ✓ Shows understanding of her priorities
  - Overall: 8/10 - Creates meaningful conversation about her regular activities

BAD STORY EXPLANATION EVALUATION:
- Prompt: "Together we could"
  Response: "Play Civ6 although not good at it"
  Photo element: Woman at a concert
  Story explanation: "I notice you're into gaming based on your Sunday routine and Civ6 mention. I see in your concert photo, Shuying, that you enjoy going out for music events too. Is that the perfect balance - gaming at home and occasional outings for special events?"
  Simulated conversation:
  * Her: "Together we could: Play Civ6 although not good at it"
  * You: "I notice you're into gaming based on your Sunday routine and Civ6 mention. I see in your concert photo, Shuying, that you enjoy going out for music events too. Is that the perfect balance - gaming at home and occasional outings for special events?"
  * Her: "Um, yeah I guess so..."
  * You: "What other games do you enjoy besides Civ6?"
  * Her: "Just that one really. I'm still learning."
  Analysis:
  - Structure: ✗ Question comes last instead of first
  - Word count: ✗ Far exceeds word limits for both parts
  - No cross-prompt info: ✗ References "Sunday routine" from a different prompt
  - Natural flow: ✗ Conversation feels awkward due to too much information
  - Enjoyable: ✗ Too much analysis makes it feel like an interrogation
  - Invites sharing: ✗ She gives short replies due to overwhelm
  - Language: ✗ Too analytical rather than conversational
  - Overall: 2/10 - Creates awkward conversation based on mismatched assumptions

- Prompt: "Typical Sunday"
  Response: "Coffee, then run, then couch"
  Photo element: Woman running
  Story explanation: "Wow, I'm so impressed by your dedication to fitness! You must be incredibly disciplined to maintain such a perfect balance in life!"
  Simulated conversation:
  * Her: "Typical Sunday: Coffee, then run, then couch"
  * You: "Wow, I'm so impressed by your dedication to fitness! You must be incredibly disciplined to maintain such a perfect balance in life!"
  * Her: "Thanks, I just enjoy running."
  * You: "That's amazing! What other fitness activities are you passionate about?"
  * Her: "Just running mostly."
  Analysis:
  - Structure: ✗ No clear question/statement first, just enthusiastic reactions
  - Word count: ✗ Doesn't follow the required structure or word count
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✗ Overly enthusiastic and assumptive
  - Enjoyable: ✗ Puts her on a pedestal rather than connecting
  - Invites sharing: ✗ Leads to short, defensive responses
  - Language: ✗ Uses marketing-style language ("impressed," "perfect")
  - Name usage: ✗ Doesn't reference her photo specifically or include her name
  - Ease of response: ✗ Creates awkward dynamic
  - Genuine interest: ✗ Focuses on admiration, not understanding
  - Overall: 2/10 - Creates uncomfortable conversation with poor flow

- Prompt: "I get along best with people who" 
  Response: "Are kind and adventurous"
  Photo element: Woman hiking on a mountain trail
  Story explanation: "Wow! You must be such an incredibly adventurous spirit with a heart of gold! Your amazing hiking photo shows you're someone who truly embodies the perfect balance of daring and compassion!"
  Simulated conversation:
  * Her: "I get along best with people who: Are kind and adventurous"
  * You: "Wow! You must be such an incredibly adventurous spirit with a heart of gold! Your amazing hiking photo shows you're someone who truly embodies the perfect balance of daring and compassion!"
  * Her: "Haha, thanks. I just enjoy hiking and try to be nice to people."
  * You: "That's so amazing! What other absolutely incredible adventures have you conquered? I bet you have the most inspirational stories!"
  * Her: "Just normal stuff really. Some camping, occasional day hikes."
  Analysis:
  - Structure: ✗ No clear question/statement first, just enthusiastic reactions
  - Word count: ✗ Doesn't follow the required structure or word count
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✗ Overly enthusiastic and exaggerated
  - Enjoyable: ✗ Creates discomfort through excessive praise
  - Invites sharing: ✗ Leads to downplaying and short responses
  - Language: ✗ Marketing-speak ("incredibly," "amazing," "perfect balance")
  - Name usage: ✗ Mentions photo but fails to include her name
  - Ease of response: ✗ Creates pressure to live up to exaggerated image
  - Genuine interest: ✗ Focuses on admiration, not understanding
  - Overall: 2/10 - Creates uncomfortable conversation through excessive enthusiasm

- Prompt: "Typical Saturday" 
  Response: "You can find me at a farmer's market"
  Photo element: Woman dressed up for a night out
  Story explanation: "Do you find farmers markets good for meeting different people? Your elegant photo shows you appreciate connecting at various venues, Natalie."
  Simulated conversation:
  * Her: "Typical Saturday: You can find me at a farmer's market"
  * You: "Do you find farmers markets good for meeting different people? Your elegant photo shows you appreciate connecting at various venues, Natalie."
  * Her: "I just like fresh vegetables and supporting local farms. I don't really go to meet people."
  * You: "That's great! Which farmer's market do you usually visit? I love the one downtown for their local honey."
  * Her: "I go to the Capitol Square market. The produce selection is amazing."
  Analysis:
  - Structure: ✓ Question first, context second
  - Word count: ✓ Question (9 words), context (12 words)
  - Question stands alone: ✗ Makes an assumption about her intention for going to farmers markets (meeting people)
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Natural flow: ✗ Makes assumptions that don't match her actual interests
  - Enjoyable: ✗ Creates a social butterfly image she doesn't identify with
  - Invites sharing: ✗ Leads to corrections rather than elaboration
  - Language: ✓ More natural than previous example
  - Name usage: ✓ Includes her name after photo reference
  - Ease of response: ✗ Forces her to correct incorrect assumptions about her intentions
  - Genuine interest: ✗ Focuses on assumed social motivations, not her actual values
  - Overall: 4/10 - Creates conversation based on mismatched assumptions about her motivations"""

SELECT_BEST_STORY_EXPLANATION = """STEP 9: SELECT BEST OVERALL STORY EXPLANATION
1. Take the three best story explanations (one for each prompt/response pair) from Step 8
2. Compare them based on these additional criteria:
   - Which has a question/statement that STANDS COMPLETELY ON ITS OWN without requiring context?
   - Which follows the correct structure (question/statement first, context second)?
   - Which has the correct word count (8-10 words for question/statement, 10-15 words for context)?
   - Which avoids using information from other prompts?
   - Which creates the most natural conversation flow?
   - Which feels most like a genuine response to her prompt?
   - Which creates the best opportunity for her to share a real story?
   - Which has the most balanced, conversational tone?
   - Which makes the most authentic connection between her prompt and photos?
   - Which avoids assumptions while still being perceptive?
   - Which would make her feel most understood?
   - Which uses her name effectively after referencing her photo?
   - Which presents genuine contrast when discussing balance (indoor/outdoor, work/play)?
3. Select the single best story explanation from the three finalists
4. This will be your final conversation starter

Example of good final selection analysis:

FINAL SELECTION ANALYSIS:
- Finalist 1: "Do you enjoy different drinks for different occasions? I see you enjoying a beer in your bar photo, Emma."
  - Structure: ✓ Question first (8 words), context second (10 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - Question stands alone: ✓ Question is completely understandable without context
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Strengths: Natural question, concrete observation, easy to respond to, includes name after photo reference
  - Weaknesses: Slightly more surface-level than other options
  - Conversation quality: 9/10

- Finalist 2: "Is balance the key to a perfect weekend? Your photos show both exercise and relaxation time, Taylor."
  - Structure: ✓ Question first (8 words), context second (10 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Strengths: Thoughtful observation, shows understanding of her values, includes name after photo reference
  - Weaknesses: Slightly more generic than other options
  - Conversation quality: 8/10

- Finalist 3: "Do hikers value kindness and new experiences? You're hiking on trails in your mountain photos, Madison."
  - Structure: ✓ Question first (7 words), context second (12 words)
  - Word count: ✓ Question within 8-10 word limit, context within 10-15 word limit
  - No cross-prompt info: ✓ Only uses information from this prompt
  - Strengths: Direct observation, connects photo to stated values, authentic tone, includes name after photo reference
  - Weaknesses: None significant
  - Conversation quality: 9/10

Selected finalist: #3 - This story explanation creates an authentic connection between her values and visible activities. It's straightforward but insightful, inviting her to share more about her hiking experiences and how they connect to her preference for kind and adventurous people. The natural inclusion of her name after mentioning her photo creates a personal touch that will increase her engagement. It also perfectly follows the required structure and word count."""

IDENTIFY_SCREENSHOT = """STEP 10: IDENTIFY SCREENSHOT
1. Note which screenshot contains the prompt/response pair text of the woman's profile that matches the chosen story explanation
2. If prompt/response pair is cut off or spans multiple screenshots, note which screenshot contains the majority of the prompt/response pair text"""

FINAL_VALIDATION_CHECK = """STEP 11: FINAL VALIDATION CHECK
Before returning your result, perform one last critical check:
1. Review your story explanation one more time
2. Ensure it:
   - Is based on a concrete, visible element in her photo
   - Makes a logical connection to her prompt/response
   - Uses natural, conversational language
   - Invites her to share more about her experiences or thoughts
   - Includes a specific reference to the photo element
   - Creates a flattering but realistic image of her life
   - Avoids making assumptions not supported by her photos or prompt
   - Feels like a genuine response to her prompt
3. This is your LAST chance to ensure your story explanation is authentic and engaging"""

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
    You will analyze 3 screenshots of a Bumble profile. Each profile may contain the following elements (following the exact order described in bumbleFormat.txt):
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

    {STORY_BASED_EXPLANATIONS}

    {EVALUATE_STORY_EXPLANATIONS}

    {SELECT_BEST_STORY_EXPLANATION}

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

    {STORY_BASED_EXPLANATIONS}

    {EVALUATE_STORY_EXPLANATIONS}

    {SELECT_BEST_STORY_EXPLANATION}

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
