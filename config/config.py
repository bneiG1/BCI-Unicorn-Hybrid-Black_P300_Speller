import json
import os

# Load configuration from the JSON file
config_path = os.path.join(os.path.dirname(__file__), 'config.json')

with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Access configurations as needed
timings = config['timings']
cooldown = timings['cooldown']
times_flashing = timings['times_flashing']

network = config['network']
udp = network['udp']
udp_sender_ip = udp['sender_ip']
udp_sender_port = udp['sender_port']
udp_acquisition_ip = udp['acquisition_ip']
udp_acquisition_port = udp['acquisition_port']

tcp = network['tcp']
tcp_ip = tcp['ip']
tcp_port = tcp['port']

boards = config['boards']
qwerty_6x6 = boards[0]['qwerty_6x6']
alphabetical_6x6 = boards[1]['alphabetical_6x6']
qwerty = boards[2]['qwerty']
alphabetical = boards[3]['alphabetical']

acquisition = config['acquisition']
source = acquisition['source']
testsignale_enabled = acquisition['testsignale_enabled']
frame_length = acquisition['frame_length']
device_id = acquisition['device_id']
