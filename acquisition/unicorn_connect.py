import sys
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError


def discover_unicorn_devices():
    """Discover available Unicorn Hybrid Black devices."""
    try:
        devices = BoardShim.get_device_name(BoardIds.UNICORN_BOARD.value)
        return devices
    except BrainFlowError as e:
        print(f"Error discovering devices: {e}")
        return []


def connect_to_unicorn(serial_port=None):
    """Connect to the Unicorn Hybrid Black EEG headset."""
    params = BrainFlowInputParams()
    if serial_port:
        params.serial_port = serial_port
    try:
        board = BoardShim(BoardIds.UNICORN_BOARD.value, params)
        board.prepare_session()
        print("Connection successful.")
        return board
    except BrainFlowError as e:
        print(f"Failed to connect to Unicorn device: {e}")
        return None


def start_streaming(board):
    try:
        board.start_stream()
        print("Data streaming started.")
    except BrainFlowError as e:
        print(f"Failed to start streaming: {e}")


def stop_streaming(board):
    try:
        board.stop_stream()
        print("Data streaming stopped.")
    except BrainFlowError as e:
        print(f"Failed to stop streaming: {e}")


def release_resources(board):
    try:
        board.release_session()
        print("Resources released.")
    except BrainFlowError as e:
        print(f"Failed to release resources: {e}")


def main():
    devices = discover_unicorn_devices()
    if not devices:
        print("No Unicorn Hybrid Black devices found.")
        sys.exit(1)
    print(f"Discovered devices: {devices}")
    # For simplicity, connect to the first discovered device
    board = connect_to_unicorn()
    if not board:
        sys.exit(1)
    try:
        start_streaming(board)
        print("Starting real-time EEG data acquisition. Press Ctrl+C to stop.")
        buffer = []
        csv_filename = 'unicorn_stream_data.csv'
        header_written = False
        while True:
            try:
                # Get all available data since last call
                data = board.get_board_data()
                if data.shape[1] > 0:
                    buffer.append(data)
                    print(f"Buffered {data.shape[1]} new samples. Total buffered: {sum(b.shape[1] for b in buffer)}")
                    # Print header information once
                    if len(buffer) == 1:
                        print("\n----- DATA HEADER INFORMATION -----")
                        sampling_rate = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
                        eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
                        accel_channels = BoardShim.get_accel_channels(BoardIds.UNICORN_BOARD.value)
                        timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.UNICORN_BOARD.value)
                        print(f"Sampling Rate: {sampling_rate} Hz")
                        print(f"Data shape: {data.shape} - Rows are channels, Columns are time samples")
                        print(f"EEG Channels (indices): {eeg_channels}")
                        print(f"Accelerometer Channels (indices): {accel_channels}")
                        print(f"Timestamp Channel (index): {timestamp_channel}")
                        print("---------------------------------\n")
                    print(f"Latest data: {data[:, -1]}")
                    # --- Save to CSV immediately as data arrives ---
                    import os
                    import numpy as np
                    # Write header if needed
                    if not header_written:
                        ch_labels = [f'ch{ch+1}' for ch in range(data.shape[0]-1)] + ['timestamp']
                        header = ','.join(ch_labels)
                        write_header = not os.path.exists(csv_filename)
                        with open(csv_filename, 'a') as f:
                            if write_header:
                                f.write(header + '\n')
                        header_written = True
                    # Write each new sample (column) to CSV as soon as it is received
                    with open(csv_filename, 'a') as f:
                        for i in range(data.shape[1]):
                            row = ','.join(f'{data[ch, i]:.5f}' for ch in range(data.shape[0]))
                            f.write(row + '\n')
                time.sleep(0.1)  # Adjust polling interval as needed
            except BrainFlowError as e:
                print(f"Data acquisition error: {e}")
                break
            except KeyboardInterrupt:
                print("\nAcquisition stopped by user.")
                break
        stop_streaming(board)
    finally:
        release_resources(board)


if __name__ == "__main__":
    main()
