#!/usr/bin/env python3
"""
Raw sanity check for a Hiwonder bus servo on a BusLinker.
Sends one broadcast "read ID" frame and prints every byte that comes back,
at several baud rates, so you can tell WHERE the link breaks.

Use with exactly ONE servo connected.
    python3 bus_raw_test.py --port /dev/ttyACM0
"""
from __future__ import annotations

import argparse
import time

import serial

# 0x55 0x55 | ID=254 (broadcast) | LEN=3 | CMD=14 (ID_READ) | checksum
BODY = bytes([254, 3, 14])
FRAME = b"\x55\x55" + BODY + bytes([(~sum(BODY)) & 0xFF])


def try_baud(port: str, baud: int) -> bytes:
    with serial.Serial(port, baud, timeout=0.05) as ser:
        ser.reset_input_buffer()
        ser.write(FRAME)
        ser.flush()
        end = time.monotonic() + 0.3
        rx = bytearray()
        while time.monotonic() < end:
            rx += ser.read(64)
        return bytes(rx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    a = ap.parse_args()
    print("sent:", FRAME.hex(" "))
    for baud in (115200, 1000000, 500000, 250000, 57600, 38400, 19200, 9600):
        rx = try_baud(a.port, baud)
        note = ""
        if not rx:
            note = "nothing back"
        elif rx == FRAME:
            note = "only an echo of what we sent (servo silent)"
        elif FRAME in rx:
            note = "echo + extra bytes (extra bytes = servo reply?)"
        else:
            note = "bytes back"
        print(f"{baud:>8} baud: {rx.hex(' ') or '-':<40} {note}")


if __name__ == "__main__":
    main()