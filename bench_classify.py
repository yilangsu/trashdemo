"""Benchmark the image classifier on a single image.

Usage:
    python3 bench_classify.py test.jpg --runs 20
"""

import argparse
import statistics
import time

from classify import classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a test image")
    parser.add_argument("--runs", type=int, default=20, help="number of timed runs")
    args = parser.parse_args()

    classify(args.image)

    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        classify(args.image)
        timings.append(time.perf_counter() - start)

    average = statistics.mean(timings)
    print(f"runs: {args.runs}")
    print(f"avg: {average * 1000:.1f} ms")
    print(f"min: {min(timings) * 1000:.1f} ms")
    print(f"max: {max(timings) * 1000:.1f} ms")


if __name__ == "__main__":
    main()