from ppadb.client import Client as AdbClient
import time
from PIL import Image
import pytesseract
import anthropic
from dotenv import load_dotenv
import os
import json
from config import ANTHROPIC_API_KEY

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


def generate_joke_from_screenshots(screenshots, format_txt_path, prompts_txt_path, captions_txt_path):
    """
    Generate a contextually appropriate joke response for a Hinge profile based on screenshots.

    Args:
        screenshots: List of paths to screenshot images in order
        format_txt_path: Path to format.txt describing profile structure
        prompts_txt_path: Path to prompts.txt containing available prompts
        captions_txt_path: Path to captions.txt containing possible captions

    Returns:
        dict: Contains selected prompt and generated joke response
    """
    # Read the content of our context files
    try:
        with open(format_txt_path, 'r') as f:
            format_content = f.read()
        with open(prompts_txt_path, 'r') as f:
            prompts_content = f.read()
        with open(captions_txt_path, 'r') as f:
            captions_content = f.read()
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
    system_prompt = f"""You are a witty and observant dating app assistant. You analyze Hinge dating profiles and generate clever, respectful jokes in response to profile prompts.

PROFILE STRUCTURE INFORMATION:
{format_content}

AVAILABLE PROMPTS:
{prompts_content}

AVAILABLE PHOTO CAPTIONS:
{captions_content}

Your role is to:
1. Understand profile content using the provided format structure in PROFILE STRUCTURE INFORMATION
2. Recognize prompts and captions from the provided lists in AVAILABLE PROMPTS and AVAILABLE PHOTO CAPTIONS
3. Generate appropriate jokes that are:
   - Contextually relevant
   - Respectful and appropriate
   - Clever and original
   - Concise (1-2 sentences)

Always return your response in JSON format:
{{
    "selected_prompt": "The exact prompt you're responding to",
    "joke_response": "Your generated joke response"
}}"""

    # User message focusing only on the analysis and joke generation steps
    user_message = """Analyze these profile screenshots and generate an appropriate joke response by:

1. Extract profile information:
   - Name, photos, prompts, responses, and profile details
   - Remember to check for content split across screenshots

2. Analyze personality through three lenses:
   a) Activities and Time Investment
      - What activities does she engage in?
      - How does she spend her time?
   
   b) Intellectual Framework
      - What are her thought patterns?
      - How does she make decisions?
      - What opinions does she express?
   
   c) Values and Priorities
      - What matters to her?
      - What brings her joy?
      - What is she seeking?

3. Generate and evaluate jokes for each prompt, considering:
   - Personality match
   - Prompt relevance
   - Humor quality
   - Originality
   - Appropriateness

4. Select and return the best joke."""

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
                "selected_prompt": result.get("selected_prompt"),
                "joke_response": result.get("joke_response")
            }
        except json.JSONDecodeError:
            print("Error: Response was not in expected JSON format")
            print("Raw response:", response.content[0].text)
            return None

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None
