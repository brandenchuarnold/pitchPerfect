# Pitch Perfect

This project also demonstrates how to automate interactions with Hinge (a dating app) using a combination of the following tools and techniques:

- **ADB (Android Debug Bridge)**: Automate device actions such as taps, swipes, and text input.
- **Computer Vision (OpenCV)**: Detect and locate UI elements on the screen using feature-based and template matching methods.
- **OCR (Tesseract via pytesseract)**: Extract text from screenshots to analyze profile descriptions or other textual content.
- **LLM (OpenAI GPT-4)**: Generate personalized, human-like comments based on extracted text content.

By integrating these components, the script can make automated decisions (like or dislike profiles) and even respond with a custom-generated pickup line or comment.

## Demo

[![PitchPerfect Demo: ](https://img.youtube.com/vi/VgES1_QHrR8/maxresdefault.jpg)](https://youtube.com/shorts/?feature=share)

## Features

- **Connect to Android Device**: Establish a connection to your Android device over ADB and retrieve screen resolution.
- **Capture Screenshots**: Save current device screen state to an image file.
- **UI Element Detection**: Locate buttons or icons using ORB feature matching and fallback template matching.
- **Text Extraction**: Use Tesseract OCR to read text content from on-screen images.
- **Comment Generation**: Use GPT-4 to create personalized, one-line comments based on the extracted profile text.
- **Automated Actions**: Simulate user input (taps, swipes, text entry) to interact with the app, such as liking or disliking profiles and inputting custom messages.

## Requirements

- **Python 3.x**
- **ADB**:  
  Install the [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools) and ensure `adb` is accessible from your PATH.
- **Device Setup**:

  - Enable Developer Options and USB Debugging on your Android device.
  - Authorize your computer for USB debugging when prompted.

- **Python Libraries**:

  - [pure-python-adb (ppadb)](https://pypi.org/project/pure-python-adb/) for ADB interactions:
    ```bash
    pip install pure-python-adb
    ```
  - [OpenCV](https://pypi.org/project/opencv-python/) for computer vision:
    ```bash
    pip install opencv-python
    ```
  - [Pillow](https://pypi.org/project/Pillow/) for image handling:
    ```bash
    pip install pillow
    ```
  - [pytesseract](https://pypi.org/project/pytesseract/) for OCR (requires Tesseract OCR engine installed on your system):
    ```bash
    pip install pytesseract
    ```
  - [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variables:
    ```bash
    pip install python-dotenv
    ```
  - [openai](https://pypi.org/project/openai/) for GPT-4 integration:
    ```bash
    pip install openai
    ```

- **Tesseract OCR Engine**:
  - **Windows**: [Download the installer here](https://github.com/UB-Mannheim/tesseract/wiki).
  - **macOS/Linux**: Install via Homebrew (`brew install tesseract`) or your package manager.

## Setup

1. **Add your OpenAI API Key**:
   Create a `.env` file in the project directory to add your OpenAI key and phone's IP address:
   ```env
   OPENAI_API_KEY=your-api-key
   ```
2. **Build the docker container**:

   ```
   docker build -t my-ocr-bot -f .\docker\Dockerfile .
   ```

3. **Run the docker container**:
   ```
   docker run my-ocr-bot
   ```

**Note: To debug**:
In case of weird behaviour, open the container and check what's up.

## Running with Docker

### Build the Docker Image

```
docker build -t my-ocr-bot -f docker/Dockerfile .
```

### Run the Docker Container

To properly run the container with volume mapping (so files can be saved to your desktop):

```
docker run -d --name pitchperfect -v ~/Desktop:/app/desktop -v $(pwd)/results:/app/results my-ocr-bot
```

This command:
- Maps your Mac desktop to `/app/desktop` inside the container
- Maps the local `results` folder to `/app/results` inside the container
- Ensures files saved in the container can be accessed from your host system

### Troubleshooting File Saving Issues

If files aren't being saved to your desktop:

1. Make sure you've started the container with the volume mounts as shown above
2. Check if the container has proper permissions to write to these directories
3. Verify the container is running with: `docker ps`
4. For debugging, run the test script: `docker exec -it pitchperfect python /app/test_volume_mount.py`

### Delete Existing Container

To delete an existing container with the same name before creating a new one:

```
docker rm -f pitchperfect 2>/dev/null || true
```

This command:
- Attempts to force remove (-f) a container named "pitchperfect"
- Suppresses any error messages if the container doesn't exist
- The `|| true` ensures the command succeeds even if no container is found

### Combined Build and Run Command

To build the image and run the container in one command:

```
docker build -t my-ocr-bot -f docker/Dockerfile . && docker run -d --name pitchperfect -v ~/Desktop:/app/desktop -v $(pwd)/results:/app/results my-ocr-bot
```

### All-in-One Command (Delete, Build, Run)

To delete any existing container, build the image, and run a new container:

#### For Bumble:
```
docker rm -f pitchperfect 2>/dev/null || true && docker build -t my-ocr-bot -f docker/Dockerfile . && docker run -d --name pitchperfect -e DATING_APP=bumble -v ~/Desktop:/app/desktop -v $(pwd)/results:/app/results my-ocr-bot
```

#### For Hinge:
```
docker rm -f pitchperfect 2>/dev/null || true && docker build -t my-ocr-bot -f docker/Dockerfile . && docker run -d --name pitchperfect -e DATING_APP=hinge -v ~/Desktop:/app/desktop -v $(pwd)/results:/app/results my-ocr-bot
```

### Shell Access

For debugging or manual operations, access the container shell:

```
docker exec -it pitchperfect /bin/bash
```

### Connecting Wirelessly

To connect wirelessly from shell:

```
adb tcpip 5555

adb connect 192.168.X.Y:5555
```

### Taking Screenshots with ADB

When working with a single device:
```
adb shell screencap -p /sdcard/screen.png && adb pull /sdcard/screen.png ~/Desktop/device_screenshot_$(date +%Y%m%d_%H%M%S).png && adb shell rm /sdcard/screen.png
```

When multiple devices are connected, specify the device ID:
```
adb -s DEVICE_ID shell screencap -p /sdcard/screen.png && adb -s DEVICE_ID pull /sdcard/screen.png ~/Desktop/device_screenshot_$(date +%Y%m%d_%H%M%S).png && adb -s DEVICE_ID shell rm /sdcard/screen.png
```

For this specific device (2B251JEGR10752):
```
adb -s 2B251JEGR10752 shell screencap -p /sdcard/screen.png && adb -s 2B251JEGR10752 pull /sdcard/screen.png ~/Desktop/device_screenshot_$(date +%Y%m%d_%H%M%S).png && adb -s 2B251JEGR10752 shell rm /sdcard/screen.png
```

To list available device IDs:
```
adb devices
```
