"""
P300 Speller BCI Application using Unicorn Hybrid Black and Brainflow
====================================================================
This script sets up a connection to the Unicorn Hybrid Black device via Brainflow
and implements the core functionality for a P300 speller application.

Requirements:
- brainflow
- numpy
"""

import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# Import Brainflow modules
import brainflow
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnicornP300Speller")


class UnicornP300Speller:
    """
    Class to handle Unicorn Hybrid Black device connection and data acquisition for a P300 speller.
    """

    # Unicorn Hybrid Black has 8 EEG channels
    CHANNEL_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    SAMPLING_RATE = 250  # Hz

    def __init__(self, serial_port: Optional[str] = None):
        """
        Initialize the P300 speller with Unicorn Hybrid Black device.

        Args:
            serial_port: Optional serial port for the device. If None, will attempt
                         to autodetect.
        """
        self.board = None
        self.serial_port = serial_port
        self.data_buffer = None
        self.sampling_rate = (
            self.SAMPLING_RATE
        )  # Make sampling rate accessible as instance attribute
        self.buffer_size = self.sampling_rate  # 1-second epochs
        self.event_markers = {}
        self.board_id = BoardIds.UNICORN_BOARD
        self.is_connected = False

        # Get the channel indices for EEG from Brainflow
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        logger.info(
            f"Initialized UnicornP300Speller with {len(self.CHANNEL_NAMES)} channels"
        )

    def connect(self) -> bool:
        """
        Connect to the Unicorn Hybrid Black device.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            params = BrainFlowInputParams()
            if self.serial_port:
                params.serial_port = self.serial_port

            logger.info("Attempting to connect to Unicorn Hybrid Black device...")

            # Enable Brainflow logger for debugging
            BoardShim.enable_dev_board_logger()

            # Initialize the board
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()

            # Start streaming with our specified sampling rate
            self.board.start_stream(45000, "")  # Buffer size set to 45000

            # Initialize the data buffer for 1-second epochs (all channels)
            num_channels = len(self.eeg_channels)
            self.data_buffer = np.zeros((num_channels, self.buffer_size))

            self.is_connected = True
            logger.info("Successfully connected to Unicorn Hybrid Black device")
            return True

        except BrainFlowError as e:
            logger.error(f"Error connecting to Unicorn Hybrid Black device: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            return False

    def disconnect(self) -> None:
        """Safely disconnect from the device."""
        if self.board and self.is_connected:
            try:
                logger.info("Stopping data stream...")
                self.board.stop_stream()
                logger.info("Releasing session...")
                self.board.release_session()
                self.is_connected = False
                logger.info("Successfully disconnected from device")
            except BrainFlowError as e:
                logger.error(f"Error during disconnection: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during disconnection: {e}")

    def update_buffer(self) -> bool:
        """
        Update the data buffer with the latest 1-second of data.

        Returns:
            bool: True if buffer updated successfully, False otherwise.
        """
        if not self.is_connected:
            logger.error("Cannot update buffer: Device not connected")
            return False

        try:
            # Get the latest data from the board
            data = self.board.get_current_board_data(self.buffer_size)

            if data.size == 0:
                logger.debug("No new data available")
                return False

            # Extract only EEG channels
            eeg_data = data[self.eeg_channels, :]

            if eeg_data.shape[1] < self.buffer_size:
                logger.debug(
                    f"Not enough data points yet: {eeg_data.shape[1]}/{self.buffer_size}"
                )
                return False

            # Update our buffer
            self.data_buffer = eeg_data
            return True

        except Exception as e:
            logger.error(f"Error getting data from device: {e}")
            return False

    def mark_event(self, event_type: str, event_data: Dict = None) -> int:
        """
        Mark an event in the EEG stream with a timestamp.

        Args:
            event_type: Type of event (e.g., 'stimulus', 'response')
            event_data: Additional data related to the event

        Returns:
            int: Timestamp of the event
        """
        if not self.is_connected:
            logger.error("Cannot mark event: Device not connected")
            return -1

        try:
            # Get current timestamp
            timestamp = int(time.time() * 1000)  # milliseconds

            # Get the current sample index from the board
            current_sample = self.board.get_board_data_count()

            # Store the event information
            event_info = {
                "timestamp": timestamp,
                "sample_index": current_sample,
                "type": event_type,
                "data": event_data,
            }

            # Store the event in our events dictionary
            self.event_markers[timestamp] = event_info

            logger.info(f"Event marked: {event_type} at {timestamp}")

            return timestamp

        except BrainFlowError as e:
            logger.error(f"Error marking event: {e}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error marking event: {e}")
            return -1

    def get_eeg_data(self, duration_seconds: float = 1.0) -> np.ndarray:
        """
        Get a specific duration of EEG data.

        Args:
            duration_seconds: Duration of data to get in seconds

        Returns:
            np.ndarray: EEG data array of shape (channels, samples)
        """
        if not self.is_connected:
            logger.error("Cannot get EEG data: Device not connected")
            return np.array([])

        try:
            num_samples = int(duration_seconds * self.SAMPLING_RATE)

            # Get data with retries
            max_retries = 3
            retry_delay = 0.01  # 10ms

            for attempt in range(max_retries):
                data = self.board.get_current_board_data(num_samples)

                if data is not None and data.size > 0:
                    eeg_data = data[self.eeg_channels, :]

                    # Validate data shape and content
                    if eeg_data.shape[0] == len(self.eeg_channels):
                        # Check for invalid values
                        if np.all(np.isfinite(eeg_data)):
                            return eeg_data
                        else:
                            logger.warning("Invalid values in EEG data, cleaning...")
                            # Replace invalid values with zeros
                            eeg_data = np.nan_to_num(
                                eeg_data, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            return eeg_data

                if attempt < max_retries - 1:
                    logger.debug(f"Retry {attempt + 1}/{max_retries} getting EEG data")
                    time.sleep(retry_delay)

            logger.error("Failed to get valid EEG data after retries")
            return np.array([])

        except BrainFlowError as e:
            logger.error(f"Error getting EEG data: {e}")
            return np.array([])
        except Exception as e:
            logger.error(f"Unexpected error getting EEG data: {e}")
            return np.array([])


# Example usage
if __name__ == "__main__":
    try:
        # Create the P300 speller
        p300_speller = UnicornP300Speller()

        # Connect to the device
        if p300_speller.connect():
            logger.info("Connected to Unicorn Hybrid Black device")

            # Acquire some data and mark events for testing
            for i in range(5):
                time.sleep(1)  # Wait 1 second

                # Update the buffer with new data
                p300_speller.update_buffer()

                # Mark a test event
                p300_speller.mark_event("test_stimulus", {"stimulus_id": i})

                # Print some stats about the data
                data = p300_speller.get_eeg_data()
                if data.size > 0:
                    logger.info(f"Data shape: {data.shape}")
                    logger.info(f"Mean values: {np.mean(data, axis=1)}")

            # Disconnect from the device
            p300_speller.disconnect()
            logger.info("Disconnected from Unicorn Hybrid Black device")
        else:
            logger.error("Failed to connect to the device")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        if p300_speller.is_connected:
            p300_speller.disconnect()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if "p300_speller" in locals() and p300_speller.is_connected:
            p300_speller.disconnect()
