import socket
import logging


def listen_for_character(ip, port):  # Use acquisition-specific IP and port
    acquisition_ip = ip
    acquisition_port = port

    logging.info(f"Listening for messages on {acquisition_ip}:{acquisition_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((acquisition_ip, acquisition_port))

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        character = data.decode("utf-8")
        logging.info(f"Received character '{character}' from {addr}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listen_for_character()
