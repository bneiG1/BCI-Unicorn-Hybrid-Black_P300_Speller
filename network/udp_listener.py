import socket
import logging


def listen_for_character(ip="0.0.0.0", port=1001):  # Default IP and port
    logging.info(f"Listening for messages on {ip}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
    try:
        sock.bind((ip, port))
    except OSError as e:
        logging.error(f"Failed to bind to {ip}:{port} - {e}")
        return

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
                character = data.decode("utf-8")
                logging.info(f"Received character '{character}' from {addr}")
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                break
    except KeyboardInterrupt:
        logging.info("Listener stopped by user.")
    finally:
        sock.close()
        logging.info("Socket closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listen_for_character()
