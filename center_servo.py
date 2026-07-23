"""Safely jog the GPIO18 servo by hand and re-center it.

A hobby servo has NO position feedback, so we can't read its current angle.
The FIRST pulse you send makes it snap to that spot at full speed -- that's the
flip risk. This tool lets you creep the servo in small steps so you can find
where it already is and gently walk it back to center without a hard swing.

Run ON THE PI:
    sudo pigpiod          # if not already running
    python3 center_servo.py

Keys:
    j / k   jog down / up by STEP microseconds (small = gentle)
    J / K   jog down / up by a bigger step
    h       go to HOME (center, 1500us)  -- only after you've crept close!
    r       set current spot as the "hold" and keep holding it
    0       release the servo (stop sending PWM -- it goes limp)
    q       release and quit

SAFETY: if anything heavy is attached, support/hold the load by hand the first
time you send a pulse, and start jogging from a value you THINK is near where it
sits. Do NOT jump straight to 'h' if you have no idea where it is.
"""

import sys
import termios
import tty

try:
    import pigpio
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pigpio not installed. Run this on the Pi with pigpiod running.") from exc

SERVO_PIN = 18
HOME_US = 1500      # center, matches live.py
MIN_US = 800        # matches TRASH_US -- don't drive past your mechanical limits
MAX_US = 2200       # matches RECYCLE_US
STEP = 25           # gentle jog, microseconds
BIG_STEP = 100


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clamp(us):
    return max(MIN_US, min(MAX_US, us))


def main():
    pi = pigpio.pi()
    if not pi.connected:
        raise SystemExit("Could not connect to pigpiod. Start it with: sudo pigpiod")

    # Start "released" -- we do NOT send a pulse until you press a key, so the
    # servo won't move the instant you launch this.
    current = HOME_US
    holding = False

    print(__doc__)
    print(f"Not sending any pulse yet. Servo range clamped to {MIN_US}-{MAX_US}us.")
    print("Press j/k to START jogging from 1500us (center). Ctrl+C or q to quit.\n")

    try:
        while True:
            c = getch()
            if c == "q":
                break
            elif c == "0":
                pi.set_servo_pulsewidth(SERVO_PIN, 0)  # release -> limp
                holding = False
                print("Released (limp, no PWM).")
                continue
            elif c == "h":
                current = HOME_US
            elif c == "r":
                pass  # just re-assert current below
            elif c == "j":
                current = clamp(current - STEP)
            elif c == "k":
                current = clamp(current + STEP)
            elif c == "J":
                current = clamp(current - BIG_STEP)
            elif c == "K":
                current = clamp(current + BIG_STEP)
            else:
                continue

            pi.set_servo_pulsewidth(SERVO_PIN, current)
            holding = True
            delta = current - HOME_US
            print(f"  -> {current} us   ({'HOME' if current == HOME_US else f'{delta:+d} from home'})")
    except KeyboardInterrupt:
        pass
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # release on exit
        pi.stop()
        print("\nReleased and disconnected.")


if __name__ == "__main__":
    main()
