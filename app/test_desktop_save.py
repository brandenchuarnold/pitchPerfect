import os
import json
from PIL import Image
import shutil
from datetime import datetime
from helper_functions import save_profile_results


def create_test_screenshot():
    """Create a test screenshot image"""
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    screenshot_path = "images/test_screenshot.png"
    img.save(screenshot_path)
    return screenshot_path


def test_save_to_desktop():
    """Test saving files to both container and desktop locations"""
    print("Creating test screenshot...")
    screenshot_path = create_test_screenshot()

    # Create a sample AI response
    ai_response = {
        "prompt": "Test prompt",
        "response": "Test response",
        "conversation_starter": "Test conversation starter",
        "screenshot_index": 0
    }

    print("Saving test profile results...")
    # Use save_profile_results function to save to both locations
    profile_dir = save_profile_results(999, [screenshot_path], ai_response)

    print(f"Test complete. Check your desktop for PitchPerfect_Results/profile_999 folder")
    print("Container path:", profile_dir)
    print("Desktop path: /app/desktop/PitchPerfect_Results/profile_999")

    return True


if __name__ == "__main__":
    test_save_to_desktop()
