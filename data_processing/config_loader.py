import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/config.json')

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return json.load(f)

# Singleton config instance for convenience
config = load_config()
