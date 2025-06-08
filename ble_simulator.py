# This module simulates BLE scan & advertising.

registered_devices = {}  # device_id -> student info

def advertise_device(device_id, student_info):
    registered_devices[device_id] = student_info

def scan_for_devices():
    # Simulate scanning by returning all registered devices nearby
    # In real use, BLE scanning would detect nearby devices
    return registered_devices
