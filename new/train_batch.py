import json
import train
from config import Config

config_file = open('config.json', 'r')
configs = json.load(config_file)

for config in configs:
    train.train(Config(**config))
