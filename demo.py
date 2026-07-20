"""
The real demo. Runs ON THE PI. Uses the Pi Camera + servo.
Setup on the Pi (Bookworm 64-bit):
    sudo apt update && sudo apt install -y python3-picamera2
    python3 -m venv --system-site-packages ~/trashdemo
    source ~/trashdemo/bin/activate
    pip install torch transformers pillow gpiozero
Then:  python demo.py
"""
from picamera2 import Picamera2
from gpiozero import AngularServo
from time import sleep
from classify import classify

SERVO_PIN = 17  # signal wire on GPIO17

HOME_ANGLE = 0
RECYCLE_ANGLE = 35
TRASH_ANGLE = -35

servo = AngularServo(
    SERVO_PIN,
    min_angle=-45,
    max_angle=45,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025,
)
cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()
sleep(2)  # camera warm-up


def sort_once():
    cam.capture_file("shot.jpg")
    label, score, is_recycle = classify("shot.jpg")
    print(f"Saw: {label} ({score:.0%})  ->  {'RECYCLE' if is_recycle else 'TRASH'}")

    servo.angle = RECYCLE_ANGLE if is_recycle else TRASH_ANGLE
    sleep(0.8)
    servo.angle = HOME_ANGLE


if __name__ == "__main__":
    print("Ready. Hold an object in front of the camera.")
    try:
        while True:
            input("Press Enter to sort (Ctrl+C to quit)...")
            sort_once()
    except KeyboardInterrupt:
        print("\nDone.")
