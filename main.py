import cv2
import numpy as np
import os
import sys
import time
import argparse
from PIL import Image

chars = np.asarray(list("ツニヌネノハヒフホモラリルレロワヲン"))
#chars = np.asarray(list("⬛"))
#chars = np.asarray(list("@#S%?*+"))
#chars = np.asarray(list("░▒▓█"))
#chars = np.asarray(list("┌─┬┐│├┼┤└┴┘"))
#chars = np.asarray(list("⠀⠁⠃⠇⡇⣇⣿"))
#chars = np.asarray(list("ᚠᚢᚦᚨᚩᚱᚷᚺᚻᚾ"))
#chars = np.asarray(list("▲△▼▽◉◎◆◇"))
#chars = np.asarray(list("♔♕♖♗♘♙"))

def toASCII(frame, scale = 0.25):
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pixels = np.array(frame)

    brightness = np.sum(pixels, axis=-1) // 3
    lookup = np.linspace(0, 255, len(chars), dtype=int)
    index = np.digitize(brightness, lookup) - 1

    buffer = []
    for row, indicies in zip(pixels, index):
        ascii_row = "".join(f"\033[38;2;{r};{g};{b}m{chars[index]}\033[0m" for (r, g, b), index in zip(row, indicies))
        buffer.append(ascii_row)
    return "\n".join(buffer)

def cropToSquare(image):
    h, w, _ = image.shape
    size = min(h, w)
    x1 = max(0, (w - size) // 2)
    x2 = min(w, x1 + size)
    cropped_image = image[:, x1:x2]
    return cropped_image

def captureFrames(path, fps = 240, scale = 0.25):
    sys.stdout.write("\033[?25l")  # Hide cursor
    vid = cv2.VideoCapture(path)

    frame_time = 1.0 / fps
    prev_time = time.time()

    while vid.isOpened():
        success, image = vid.read()
        if not success:
            break

        image = cropToSquare(image)
        ascii_image = toASCII(image, scale)

        sys.stdout.write("\033[H\033[J")
        sys.stdout.write(ascii_image + "\n")
        sys.stdout.flush()

        elpased_time = time.time() - prev_time
        sleep_time = max(0, frame_time - elpased_time)
        time.sleep(sleep_time)
        prev_time = time.time()
    vid.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the video file")
    parser.add_argument("--scale", type=float, default=0.25, help="Scale for ASCII conversion")
    parser.add_argument("--fps", type=int, default=240, help="Frames per second for video playback")
    args = parser.parse_args()

    if not args.file:
        print("Please provide a video file path using --file argument.")
        exit()    
    captureFrames(args.file, args.fps, args.scale)
    sys.stdout.write("\033[?25h")