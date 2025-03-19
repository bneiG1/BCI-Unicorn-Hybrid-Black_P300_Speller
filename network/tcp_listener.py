import socket
import logging


def listen_for_character(ip, port):  # Add ip and port parameters
    tcp_ip = ip  # Use the ip parameter
    tcp_port = port  # Use the port parameter

    logging.info(f"Listening for messages on {tcp_ip}:{tcp_port} via TCP")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((tcp_ip, tcp_port))
    sock.listen(1)

    while True:
        conn, addr = sock.accept()
        logging.info(f"Connection from {addr}")
        data = conn.recv(1024)  # Buffer size is 1024 bytes
        if data:
            character = data.decode("utf-8")
            logging.info(f"Received character '{character}'")
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listen_for_character()
