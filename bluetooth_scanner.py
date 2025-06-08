import asyncio
from bleak import BleakScanner
import bluetooth_scanner as scanner

async def scan_and_mark():
    devices = await BleakScanner.discover()
    student_names = get_students()
    for d in devices:
        if d.name in student_names:
            mark_attendance(d.name)
            print(f"Marked attendance for: {d.name}")
