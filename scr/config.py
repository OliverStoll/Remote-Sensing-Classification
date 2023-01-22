import yaml


# import config.yaml
with open("fileS/config.yaml", 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
