import serial
import time
import requests

# ------------------ CONFIGURATION ------------------
SERIAL_PORT = "COM11"    # Change if your Arduino uses another port
BAUD_RATE = 9600
API_URL = "http://localhost:5000/api/ingest"
RECONNECT_DELAY = 5      # seconds to wait before reconnecting
POST_TIMEOUT = 5         # seconds to wait for server response
# ---------------------------------------------------

def parse_line(line: str) -> dict:
    """
    Parse incoming serial line like:
    'moisture:45,temperature:24.50'
    Returns a dictionary: {'moisture': 45.0, 'temperature': 24.5}
    """
    result = {}
    for part in line.strip().split(','):
        if ':' in part:
            key, value = part.split(':', 1)
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                result[key.strip()] = None
    return result


def open_serial():
    """Try to open the serial port safely."""
    while True:
        try:
            print(f"🔌 Trying to connect to {SERIAL_PORT} at {BAUD_RATE} baud...")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            time.sleep(2)  # allow Arduino to reset
            print("✅ Connected to Arduino!")
            return ser
        except serial.SerialException as e:
            print(f"❌ Could not open {SERIAL_PORT}: {e}")
            print(f"⏳ Retrying in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)


def main():
    ser = open_serial()
    while True:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw:
                continue

            print(f"📥 RAW > {raw}")
            data = parse_line(raw)

            payload = {
                'moisture': data.get('moisture'),
                'temperature': data.get('temperature'),
                'ph': 7.0,   # You can update this if you add a pH sensor
                'npk': 120   # Same for NPK sensor
            }

            try:
                r = requests.post(API_URL, json=payload, timeout=POST_TIMEOUT)
                print(f"🌐 Sent -> {payload} | Server Response: {r.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error sending data to backend: {e}")

        except serial.SerialException:
            print("⚠️ Serial connection lost. Attempting to reconnect...")
            ser.close()
            ser = open_serial()
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            time.sleep(2)  # small delay before retry


if __name__ == '__main__':
    main()
