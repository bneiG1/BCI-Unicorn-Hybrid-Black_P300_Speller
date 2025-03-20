import logging
import threading
import UnicornPy
import numpy as np
# Import acquisition settings from config
from config.config import source, udp_acquisition_port, udp_acquisition_ip,device_id, testsignale_enabled, frame_length
from network.udp_listener import listen_for_character


# Global variables to manage the acquisition process
acquisition_thread = None
acquisition_running = False
acquisition_lock = threading.Lock()  # Add a lock for thread safety


def acquire_data(callback=None):  # Add a callback parameter
    try:
        # Get available devices.
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception(
                "No device available. Please pair with a Unicorn first.")

        print("Available devices:")
        for i, device in enumerate(deviceList):
            print(f"#{i} {device}")

        if device_id < 0 or device_id >= len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        print()
        print(f"Trying to connect to '{deviceList[device_id]}'.")
        device = UnicornPy.Unicorn(deviceList[device_id])
        print(f"Connected to '{deviceList[device_id]}'.")
        print()

        # Initialize acquisition members.
        numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

            # Get indices of EEG channels
        # eeg_channel_indices = [
        #     device.GetChannelIndex(f"EEG {i+1}") for i in range(UnicornPy.EEGChannelsCount)
        # ]

        print("Acquisition Configuration:")
        print(f"Sampling Rate: {UnicornPy.SamplingRate} Hz")
        print(f"Frame Length: {frame_length}")
        print(f"Number Of Acquired Channels: {numberOfAcquiredChannels}")
        # print(f"EEG Channel Indices: {eeg_channel_indices}")
        print()

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = frame_length * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        try:
            # Start data acquisition.
            device.StartAcquisition(testsignale_enabled)
            print("Data acquisition started.")

            # Continuous acquisition loop.
            while acquisition_running:
                device.GetData(frame_length, receiveBuffer,
                               receiveBufferBufferLength)

                # Convert receive buffer to numpy float array.
                data = np.frombuffer(
                    receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * frame_length)
                data = np.reshape(
                    data, (frame_length, numberOfAcquiredChannels))
                
                                # Extract only EEG channel data
                # eeg_data = data[:, eeg_channel_indices]

                if callback:
                    callback(data)  # Pass data to the callback function

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print(f"An unknown error occurred. {e}")
        finally:
            del receiveBuffer
            device.StopAcquisition()
            del device
            print("Disconnected from Unicorn")

    except UnicornPy.DeviceException as e:
        print(e)


# Function to start the acquisition process
# This function starts a background thread to acquire data from the Unicorn device.
def start_acquisition():
    if source == "internal":
        global acquisition_thread, acquisition_running
        with acquisition_lock:  # Use the lock to ensure thread safety
            if not acquisition_running:
                logging.info("Starting acquisition process")
                acquisition_running = True

                def acquisition_task():
                    acquire_data(callback=process_data)  # Pass a callback function

                acquisition_thread = threading.Thread(
                    target=acquisition_task, daemon=True)
                acquisition_thread.start()
    elif source == "external":
        listener_thread = threading.Thread(
            target=listen_for_character, args=(
                udp_acquisition_ip, udp_acquisition_port), daemon=True
        )
        listener_thread.start()  # Use external UDP listener
    else:
        logging.error(
            "Invalid source configuration. Must be 'internal' or 'external'.")


def process_data(data):
    # Placeholder for processing acquired data
    logging.info(f"Processing data: {data}")


# Function to stop the acquisition process
# This function stops the background thread that acquires data from the Unicorn device.
def stop_acquisition():
    global acquisition_running
    if acquisition_running:
        logging.info("Stopping acquisition process")
        acquisition_running = False
        if acquisition_thread:
            acquisition_thread.join()
