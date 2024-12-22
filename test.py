from helper_functions import (
    connect_device,
    get_screen_resolution,
    open_hinge,
    swipe,
    capture_screenshot,
    extract_text_from_image,
    do_comparision,
    find_icon,
    generate_comment,
    tap,
    input_text,
)


def main():
    device = connect_device()

    if not device:
        return

    width, height = get_screen_resolution(device)

    # Approximate coordinates based on experimentation
    x_comment_box = int(width * 0.9)
    y_comment_box = int(height * 0.95)

    x_send = int(width * 0.95)
    y_send = int(width * 0.99)

    x_send_button = int(width * 0.90)
    y_send_button = int(height * 0.89)

    x1_swipe = int(width * 0.15)
    x2_swipe = x1_swipe

    y1_swipe = int(height * 0.5)
    y2_swipe = int(y1_swipe * 0.75)

    text = "Hi, how are you?"

    # capture_screenshot(device, "test")

    input_text(device, text)

    tap(device, x_comment_box, y_comment_box)
    # swipe(device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)

    # works for sending info in whatsapp
    swipe(device, width * 0.90, y_send_button, width * 0.95, y_send_button)
    # tap(device, x_send_button, y_send_button)

    for x in range(10):
        tap(device, int(height * 0.99), int(width * (0.9 + 0.01 * x)))
    # for x in range(100):
    # for y in range(100):
    # tap(device, 0.01 * i, 0.99)
    # tap(device, int(width * (0.9 + 0.01 * x)), int(height * (0.99 + 0.01 * y)))


main()
