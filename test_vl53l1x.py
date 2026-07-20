"""Quick VL53L1X sanity test.

Usage:
    python3 test_vl53l1x.py

It prints distance readings once the sensor is detected on I2C address 0x29.
"""

import time

import board
import busio
import adafruit_vl53l1x


def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_vl53l1x.VL53L1X(i2c)

    # A longer timing budget is more stable for a quick bench test.
    sensor.distance_mode = 2  # long range
    sensor.timing_budget = 100
    sensor.start_ranging()

    print("VL53L1X started. Press Ctrl+C to stop.")
    try:
        while True:
            if sensor.data_ready:
                distance_mm = sensor.distance
                print(f"Distance: {distance_mm} mm")
                sensor.clear_interrupt()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        sensor.stop_ranging()


if __name__ == "__main__":
    main()
