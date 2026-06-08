import yaml

class Config:

    def __init__(self, d):

        for k, v in d.items():

            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def load_config(path):

    with open(path, "r") as f:
        return Config(
            yaml.safe_load(f)
        )