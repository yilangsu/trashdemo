"""Drive the GPIO18 servo with approximate PWM pulse widths.

Usage on the Pi:
    sudo pigpiod
    python3 test_servos.py

Wiring assumed:
    GPIO18 / physical pin 12
"""

from time import sleep

try:
    import pigpio
except ImportError as exc:  # pragma: no cover - depends on local Pi setup
    raise SystemExit(
        "pigpio is not installed or not on PYTHONPATH. Install it or use the pigpio package in this repo."
    ) from exc


CENTER_US = 1500
LEFT_US = 1150
RIGHT_US = 1850
HOLD_SECONDS = 2.0

SERVO_NAME = "GPIO18 / physical pin 12"
SERVO_PIN = 18


def set_servo(pi, pulsewidth, label):
    print(f"  {label} -> {pulsewidth} us")
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)


def main():
    pi = pigpio.pi()
    if not pi.connected:
        raise SystemExit("Could not connect to pigpiod. Start it with: sudo pigpiod")

    print("Servo PWM test starting.")
    print("GPIO18 will sweep around center and hold each position for 2 seconds.")

    try:
        set_servo(pi, CENTER_US, f"{SERVO_NAME} center")
        sleep(HOLD_SECONDS)

        set_servo(pi, RIGHT_US, f"{SERVO_NAME} right")
        sleep(HOLD_SECONDS)

        set_servo(pi, LEFT_US, f"{SERVO_NAME} left")
        sleep(HOLD_SECONDS)

        set_servo(pi, CENTER_US, f"{SERVO_NAME} center")
        print("\nHolding at center. Press Ctrl+C to stop.")
        while True:
            sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)
        pi.stop()


if __name__ == "__main__":
    main()