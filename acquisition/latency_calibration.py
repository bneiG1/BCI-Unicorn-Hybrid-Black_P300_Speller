import time
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

def measure_bluetooth_latency(board, n_trials=10):
    """
    Measure Bluetooth transmission latency by sending a marker and timing round-trip delay.
    Args:
        board: BrainFlow BoardShim instance
        n_trials: Number of latency measurements
    Returns:
        List of measured latencies (seconds)
    """
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        # Send marker (if supported)
        try:
            board.insert_marker(99)
        except Exception:
            pass  # Not all boards support marker insertion
        # Wait for marker to appear in data
        found = False
        for _ in range(100):
            data = board.get_board_data()
            marker_channel = BoardShim.get_marker_channel(BoardIds.UNICORN_BOARD.value)
            if marker_channel < data.shape[0] and np.any(data[marker_channel] == 99):
                found = True
                break
            time.sleep(0.01)
        t1 = time.perf_counter()
        if found:
            latencies.append(t1 - t0)
        else:
            latencies.append(np.nan)
    return latencies

if __name__ == "__main__":
    from acquisition.unicorn_connect import connect_to_unicorn, start_streaming, stop_streaming
    board = connect_to_unicorn()
    if board:
        print("Measuring Bluetooth latency...")
        try:
            start_streaming(board)
            lats = measure_bluetooth_latency(board)
            print(f"Latencies (s): {lats}")
            print(f"Mean latency: {np.nanmean(lats):.4f} s, Std: {np.nanstd(lats):.4f} s")
            stop_streaming(board)
        except Exception as e:
            print(f"Error during latency measurement: {e}")
        finally:
            board.release_session()
    else:
        print("Could not connect to Unicorn device.")
