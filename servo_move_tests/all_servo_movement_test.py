#!/usr/bin/env python3
"""
Minimal driver for Hiwonder bus servos (HX-35HM etc.) through a BusLinker
V3.0 in USB mode. No vendor software needed: the board's CH341 USB-serial
chip uses the in-kernel `ch341` driver on Linux, so it shows up as /dev/ttyACM*.

Protocol (Hiwonder / LewanSoul Bus Servo Communication Protocol):
    0x55 0x55 | ID | LEN | CMD | PARAMS... | CHECKSUM
    LEN      = len(PARAMS) + 3
    CHECKSUM = (~(ID + LEN + CMD + sum(PARAMS))) & 0xFF
    ID 254 (0xFE) = broadcast (use it only with ONE servo attached when reading the ID)

Usage:
    pip install pyserial
    python3 hiwonder_bus.py --port /dev/ttyACM0 scan
    python3 hiwonder_bus.py --port /dev/ttyACM0 info 1
    python3 hiwonder_bus.py --port /dev/ttyACM0 move 1 750 1000   # pos 750, 1000 ms
"""
import argparse
import time

import serial

# Command numbers from the protocol document
MOVE_TIME_WRITE = 1
ID_WRITE = 13
ID_READ = 14
TEMP_READ = 26
VIN_READ = 27
POS_READ = 28
LOAD_OR_UNLOAD_WRITE = 31

BROADCAST_ID = 254

serialHandle = serial.Serial('/dev/ttyACM0', 1000000) #Sets baud rate


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


class ServoBus:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self):
        self.ser.close()

    # ---- framing -----------------------------------------------------------
    def _send(self, sid: int, cmd: int, params: bytes = b""):
        body = bytes([sid, len(params) + 3, cmd]) + params
        self.ser.reset_input_buffer()
        self.ser.write(b"\x55\x55" + body + bytes([checksum(body)]))
        self.ser.flush()

    def _recv(self, cmd: int, deadline_s: float = 0.1) -> bytes | None:
        """Return params of the first valid reply to `cmd`.

        Half-duplex bus: depending on the adapter you may also read back your
        own request (an "echo"). Requests to read commands have LEN == 3, real
        replies have LEN > 3, so echoes are skipped by that test.
        """
        end = time.monotonic() + deadline_s
        buf = bytearray()
        while time.monotonic() < end:
            buf += self.ser.read(self.ser.in_waiting or 1)
            while len(buf) >= 6:
                if buf[0] != 0x55 or buf[1] != 0x55:
                    del buf[0]
                    continue
                length = buf[3]
                frame_len = length + 3  # 2 header bytes + ID + LEN..CHECKSUM
                if len(buf) < frame_len:
                    break
                frame = bytes(buf[:frame_len])
                del buf[:frame_len]
                body, cs = frame[2:-1], frame[-1]
                if checksum(body) != cs:
                    continue  # corrupted frame
                if body[2] == cmd and length > 3:
                    return body[3:]
        return None

    def _query(self, sid: int, cmd: int) -> bytes | None:
        self._send(sid, cmd)
        return self._recv(cmd)

    # ---- commands ----------------------------------------------------------
    def move(self, sid: int, pos: int, ms: int):
        """HX-35HM: pos 0..1500 maps to 0..360 deg (HX-35H: 0..1000 -> 0..240)."""
        p = bytes([pos & 0xFF, (pos >> 8) & 0xFF, ms & 0xFF, (ms >> 8) & 0xFF])
        self._send(sid, MOVE_TIME_WRITE, p)

    def torque(self, sid: int, on: bool):
        self._send(sid, LOAD_OR_UNLOAD_WRITE, bytes([1 if on else 0]))

    def read_pos(self, sid: int) -> int | None:
        r = self._query(sid, POS_READ)
        return None if r is None else int.from_bytes(r[:2], "little", signed=True)

    def read_vin_mv(self, sid: int) -> int | None:
        r = self._query(sid, VIN_READ)
        return None if r is None else int.from_bytes(r[:2], "little")

    def read_temp_c(self, sid: int) -> int | None:
        r = self._query(sid, TEMP_READ)
        return None if r is None else r[0]

    def read_id(self, sid: int = BROADCAST_ID) -> int | None:
        r = self._query(sid, ID_READ)
        return None if r is None else r[0]

    def set_id(self, old: int, new: int):
        self._send(old, ID_WRITE, bytes([new]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1000000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    s = sub.add_parser("info");
    s.add_argument("id", type=int)
    s = sub.add_parser("move");
    s.add_argument("id", type=int)
    s.add_argument("pos", type=int);
    s.add_argument("ms", type=int, nargs="?", default=1000)
    s = sub.add_parser("setid");
    s.add_argument("old", type=int);
    s.add_argument("new", type=int)
    s = sub.add_parser("off");
    s.add_argument("id", type=int)

    a = ap.parse_args()

    bus = ServoBus(a.port, a.baud)

    left_leg_ids = [11,12,13,14,15]
    right_leg_ids = [21,22,23,24,25]
    neck_ids = [1,2,3,4]

    try:
        if a.cmd == "scan":
            found = [i for i in range(0, 254) if bus.read_id(i) is not None]
            print("servos found:", found or "none")
        elif a.cmd == "info":
            print(f"id={a.id} pos={bus.read_pos(a.id)} "
                  f"vin={bus.read_vin_mv(a.id)} mV temp={bus.read_temp_c(a.id)} C")
        elif a.cmd == "move":
            bus.move(a.id, a.pos, a.ms)
        elif a.cmd == "setid":
            bus.set_id(a.old, a.new)
            time.sleep(0.05)
            print("now reads as:", bus.read_id(a.new))
        elif a.cmd == "off":
            bus.torque(a.id, False)
    finally:
        bus.close()


if __name__ == "__main__":
    main()