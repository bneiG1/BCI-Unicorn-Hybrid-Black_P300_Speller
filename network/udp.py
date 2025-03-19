import socket
import logging  # Add this import


def send_character(character, ip, port):  # Add ip and port parameters
    udp_ip = ip  # Use the ip parameter
    udp_port = port  # Use the port parameter
    message = character.encode("utf-8")

    logging.info(f"Sending character '{character}' to {udp_ip}:{udp_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message, (udp_ip, udp_port))
        logging.info("Message sent successfully")
    except Exception as e:
        logging.error(f"Failed to send message: {e}")
    finally:
        sock.close()
