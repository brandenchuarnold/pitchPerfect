import os
import json
from datetime import datetime


def test_volume_mount():
    """Test if the volume mount to desktop is working properly"""

    # Create desktop directory path if it doesn't exist
    desktop_dir = "/app/desktop"
    if not os.path.exists(desktop_dir):
        print(f"ERROR: Desktop directory {desktop_dir} doesn't exist.")
        print("Volume mount might not be working correctly.")
        return False

    # Create a test file in the desktop directory
    test_file_path = os.path.join(desktop_dir, "test_volume_mount.txt")

    # Write test data to the file
    test_data = {
        "test_id": "volume_mount_test",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "If you can see this file on your Mac desktop, volume mounting is working!"
    }

    try:
        with open(test_file_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test file successfully created at {test_file_path}")
        print(
            "If volume mounting is working, you should see this file on your Mac desktop.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create test file: {e}")
        return False


if __name__ == "__main__":
    test_volume_mount()
