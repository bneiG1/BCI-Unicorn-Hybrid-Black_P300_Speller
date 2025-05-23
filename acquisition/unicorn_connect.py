import sys
import time
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)
from config.config_loader import config

# Electrode mode: 'dry' (default) or 'gel'. Loaded from config, can be changed at runtime.
_electrode_mode = config.get("electrode_mode", "dry")

def get_electrode_mode():
    """Get current electrode mode ('dry' or 'gel')."""
    return _electrode_mode

def set_electrode_mode(mode: str):
    """Set electrode mode ('dry' or 'gel')."""
    global _electrode_mode
    if mode not in ("dry", "gel"):
        raise ValueError("Electrode mode must be 'dry' or 'gel'.")
    _electrode_mode = mode
    print(f"Electrode mode set to: {_electrode_mode}")


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


def discover_and_connect():
    devices = discover_unicorn_devices()
    if not devices:
        print("No Unicorn Hybrid Black devices found.")
        return None
    print(f"Discovered devices: {devices}")
    board = connect_to_unicorn()
    if not board:
        print("Failed to connect to Unicorn device.")
        return None
    return board


def start_eeg_streaming(board):
    try:
        start_streaming(board)
        print("Starting real-time EEG data acquisition. Press Ctrl+C to stop.")
        return True
    except Exception as e:
        print(f"Failed to start streaming: {e}")
        return False


def get_board_data(board):
    ch_labels = []
    data = board.get_board_data()
    print("\n----- DATA HEADER INFORMATION -----")
    board_id = BoardIds.UNICORN_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    for name, indices in board_descr.items():
        print(f"{name}: {indices}")
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    accel_channels = BoardShim.get_accel_channels(BoardIds.UNICORN_BOARD.value)
    gyroscope_channels = BoardShim.get_gyro_channels(BoardIds.UNICORN_BOARD.value)
    battery_channel = BoardShim.get_battery_channel(BoardIds.UNICORN_BOARD.value)
    counter_channel = BoardShim.get_package_num_channel(BoardIds.UNICORN_BOARD.value)
    validation_channel = BoardShim.get_other_channels(BoardIds.UNICORN_BOARD.value)
    timestamp_channel = BoardShim.get_timestamp_channel(BoardIds.UNICORN_BOARD.value)
    markers_channel = BoardShim.get_marker_channel(BoardIds.UNICORN_BOARD.value)
    # Build header and print mapping

    eeg_names = ["Fp1", "Fp2", "C3", "C4", "Pz", "O1", "O2", "Fz"]
    for i, ch in enumerate(eeg_channels):
        name = eeg_names[i] if i < len(eeg_names) else f"EEG {i+1}"
        ch_labels.append(name)
    for i, ch in enumerate(accel_channels):
        name = (
            ["Accelerometer X", "Accelerometer Y", "Accelerometer Z"][i]
            if i < 3
            else f"Accelerometer {i+1}"
        )
        ch_labels.append(name)
    for i, ch in enumerate(gyroscope_channels):
        name = (
            ["Gyroscope X", "Gyroscope Y", "Gyroscope Z"][i]
            if i < 3
            else f"Gyroscope {i+1}"
        )
        ch_labels.append(name)
    if battery_channel is not None:
        ch_labels.append("Battery Level")
    if counter_channel is not None:
        ch_labels.append("Counter")
    # Validation indicator may be a list
    if validation_channel is not None:
        if isinstance(validation_channel, list) and len(validation_channel) > 0:
            ch_labels.append("Validation Indicator")
        elif isinstance(validation_channel, int):
            ch_labels.append("Validation Indicator")
    if timestamp_channel is not None:
        ch_labels.append("Timestamp")
    if markers_channel is not None:
        ch_labels.append("Markers")
    print(f"Data shape: {data.shape} - Rows are channels, Columns are time samples")
    print(f'CSV header: {", ".join(ch_labels)}')
    print("\n")
    return ch_labels


def save_data_to_csv(eeg_data, csv_filename, header=None):
    import os

    # eeg_data: np.ndarray (channels x samples)
    # header: list of channel names or None
    if not os.path.exists("data"):
        os.makedirs("data")
    write_header = header is not None and not os.path.exists(csv_filename)
    if write_header and header:
        with open(csv_filename, "a") as f:
            f.write(",".join(header) + "\n")
    with open(csv_filename, "a") as f:
        for i in range(eeg_data.shape[1]):
            row = ",".join(f"{eeg_data[ch, i]:.5f}" for ch in range(eeg_data.shape[0]))
            f.write(row + "\n")
            # print(f"Latest data: {row}")


def run_acquisition():
    import datetime

    board = discover_and_connect()
    if not board:
        sys.exit(1)
    try:
        if not start_eeg_streaming(board):
            release_resources(board)
            sys.exit(1)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"data/unicorn_stream_data_{timestamp_str}.csv"
        save_data_to_csv(board, csv_filename)
        stop_streaming(board)
    finally:
        release_resources(board)


def main():
    run_acquisition()


if __name__ == "__main__":
    main()
