#!/usr/bin/env python3
import sys
import os
import time
import csv
import argparse
from datetime import datetime

try:
    import serial
except ImportError:
    print("Error: 'pyserial' package is not installed. Run: pip install pyserial")
    sys.exit(1)


def record_ecg_data(port: str = "/dev/ttyACM0", baudrate: int = 115200, duration_sec: int = 60, output_path: str = None):
    """
    Connects to ESP32 serial stream, triggers continuous streaming,
    records data for `duration_sec` seconds (default 60s), and saves to CSV.
    """
    if output_path is None:
        os.makedirs("recordings", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.abspath(f"recordings/ecg_record_{timestamp_str}.csv")
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 65)
    print("      NEXVIORA 60-SECOND ECG & TELEMETRY RECORDER")
    print("=" * 65)
    print(f"Target Serial Port: {port}")
    print(f"Baud Rate:          {baudrate}")
    print(f"Recording Duration: {duration_sec} seconds")
    print(f"Output CSV File:    {output_path}")
    print("=" * 65)

    headers = ["timestamp_ms", "ecg_filtered", "ppg_dc_ir", "ecg_bpm", "spo2_pct"]

    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
        time.sleep(1.5)  # Allow serial connection to settle
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"\n[ERROR] Failed to open serial port '{port}': {e}")
        print("\nPossible solutions:")
        print("1. Ensure ESP32 is plugged in via USB cable.")
        print("2. Make sure no other process (like Arduino IDE Serial Monitor or FastAPI backend) is using the port.")
        print("3. Check serial port permission: sudo chmod a+rw /dev/ttyACM0")
        sys.exit(1)

    print("\nSending #REC_START command to ESP32...")
    ser.write(b"#REC_START\n")
    ser.flush()

    recorded_rows = 0
    start_time = time.time()
    last_print_time = start_time

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        print(f"\nRecording started... (will stop automatically after {duration_sec}s)\n")

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration_sec:
                    break

                line_bytes = ser.readline()
                if not line_bytes:
                    continue

                try:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                # Parse lines starting with "REC,"
                if line.startswith("REC,"):
                    parts = line.split(",")
                    if len(parts) >= 6:
                        try:
                            timestamp_ms = int(parts[1])
                            ecg_filtered = float(parts[2])
                            ppg_dc_ir = float(parts[3])
                            ecg_bpm = float(parts[4])
                            spo2_pct = float(parts[5])

                            writer.writerow([timestamp_ms, ecg_filtered, ppg_dc_ir, ecg_bpm, spo2_pct])
                            recorded_rows += 1

                        except ValueError:
                            continue

                # Progress readout every 1 second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    remaining = max(0, int(duration_sec - elapsed))
                    samples_per_sec = int(recorded_rows / max(elapsed, 0.1))
                    sys.stdout.write(f"\rProgress: [{int(elapsed)}s / {duration_sec}s] | Samples Saved: {recorded_rows} ({samples_per_sec} Hz)  ")
                    sys.stdout.flush()
                    last_print_time = current_time

        except KeyboardInterrupt:
            print("\n\n[USER INTERRUPT] Recording stopped early by user.")

        finally:
            print("\nSending #REC_STOP command to ESP32...")
            try:
                ser.write(b"#REC_STOP\n")
                ser.flush()
                ser.close()
            except Exception:
                pass

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("               RECORDING COMPLETE")
    print("=" * 65)
    print(f"Total Duration:     {total_elapsed:.2f} seconds")
    print(f"Total Samples:      {recorded_rows} samples")
    print(f"Columns Recorded:   {', '.join(headers)}")
    print(f"Saved CSV Location: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record 60s of ECG & PPG data from ESP32 to CSV")
    parser.add_argument("-p", "--port", type=str, default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Recording duration in seconds (default: 60)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path (optional)")

    args = parser.parse_args()
    record_ecg_data(port=args.port, baudrate=args.baud, duration_sec=args.duration, output_path=args.output)
