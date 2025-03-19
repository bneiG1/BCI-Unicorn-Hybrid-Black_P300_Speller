import socket
import logging


def send_character(character, ip, port):  # Add ip and port parameters
    tcp_ip = ip  # Use the ip parameter
    tcp_port = port  # Use the port parameter
    message = character.encode("utf-8")

    logging.info(f"Sending character '{character}' to {tcp_ip}:{tcp_port} via TCP")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((tcp_ip, tcp_port))
        sock.sendall(message)
        logging.info("Message sent successfully")
    except Exception as e:
        logging.error(f"Failed to send message: {e}")
    finally:
        sock.close()
