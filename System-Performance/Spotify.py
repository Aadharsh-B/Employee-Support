import psutil
import serial
import time

# Check if Spotify is running
def is_spotify_running():
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == 'spotify.exe':
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# Check if WhatsApp is running
def is_whatsapp_running():
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == 'whatsapp.exe':
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# Open serial port (change 'COMx' to the correct port)
def get_serial_connection():
    return serial.Serial('COM3', 9600, timeout=1)

def main():
    ser = get_serial_connection()

    while True:
        spotify_running = is_spotify_running()
        whatsapp_running = is_whatsapp_running()

        if spotify_running and whatsapp_running:
            ser.write(b'3')  # Send '3' if both Spotify and WhatsApp are running
        elif spotify_running:
            ser.write(b'1')  # Send '1' if only Spotify is running
        elif whatsapp_running:
            ser.write(b'2')  # Send '2' if only WhatsApp is running
        else:
            ser.write(b'0')  # Send '0' if neither are running

        time.sleep(1)

if __name__ == "__main__":
    main()
