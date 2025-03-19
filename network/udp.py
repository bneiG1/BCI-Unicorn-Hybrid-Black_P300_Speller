import socket
import logging  # Add this import


def send_character(character, ip, port):  # Use sender-specific IP and port
    udp_ip = ip  # Use sender_ip
    udp_port = port  # Use sender_port
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
