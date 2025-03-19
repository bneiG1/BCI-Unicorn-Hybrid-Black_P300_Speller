import socket
import logging


def listen_for_character(ip, port):  # Add ip and port parameters
    udp_ip = ip  # Use the ip parameter
    udp_port = port  # Use the port parameter

    logging.info(f"Listening for messages on {udp_ip}:{udp_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        character = data.decode("utf-8")
        logging.info(f"Received character '{character}' from {addr}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listen_for_character()
