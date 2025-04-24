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
        while True:
            try:
                # Get all available data since last call
                data = board.get_board_data()
                if data.shape[1] > 0:
                    buffer.append(data)
                    print(f"Buffered {data.shape[1]} new samples. Total buffered: {sum(b.shape[1] for b in buffer)}")
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
