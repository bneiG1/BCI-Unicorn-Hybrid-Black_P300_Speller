import UnicornPy
import numpy as np
import socket

def main():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    TestsignaleEnabled = False
    FrameLength = 1
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1001

    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        # Get available devices.
        #-------------------------------------------------------------------------------------
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available. Please pair with a Unicorn first.")

        print("Available devices:")
        for i, device in enumerate(deviceList):
            print(f"#{i} {device}")

        print()
        deviceID = int(input("Select device by ID #"))
        if deviceID < 0 or deviceID >= len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        print()
        print(f"Trying to connect to '{deviceList[deviceID]}'.")
        device = UnicornPy.Unicorn(deviceList[deviceID])
        print(f"Connected to '{deviceList[deviceID]}'.")
        print()

        # Initialize acquisition members.
        #-------------------------------------------------------------------------------------
        numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

        print("Acquisition Configuration:")
        print(f"Sampling Rate: {UnicornPy.SamplingRate} Hz")
        print(f"Frame Length: {FrameLength}")
        print(f"Number Of Acquired Channels: {numberOfAcquiredChannels}")
        print()

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        # Set up UDP socket.
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            # Start data acquisition.
            #-------------------------------------------------------------------------------------
            device.StartAcquisition(TestsignaleEnabled)
            print("Data acquisition started. Press Ctrl+C to stop.")

            consoleUpdateRate = max(1, int((UnicornPy.SamplingRate / FrameLength) / 25.0))

            # Continuous acquisition loop.
            #-------------------------------------------------------------------------------------
            while True:
                device.GetData(FrameLength, receiveBuffer, receiveBufferBufferLength)

                # Convert receive buffer to numpy float array.
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))

                # Send data via UDP.
                udp_socket.sendto(data.tobytes(), (UDP_IP, UDP_PORT))

                # Update console to indicate that the data acquisition is running.
                print('.', end='', flush=True)

        except KeyboardInterrupt:
            print("\nData acquisition stopped by user.")
        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print(f"An unknown error occurred. {e}")
        finally:
            del receiveBuffer
            udp_socket.close()
            device.StopAcquisition()
            del device
            print("Disconnected from Unicorn")

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print(f"An unknown error occurred. {e}")

    input("\n\nPress ENTER key to exit")

# Execute main
main()
