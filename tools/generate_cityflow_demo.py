import cityflow
import json
import os
import time

import click
import numpy as np


def save_config_file(roadnetFile, flowFile):
    config_dict = {
        "interval": 1,
        "seed": 0,
        "dir": "./",
        "roadnetFile": roadnetFile,
        "flowFile": flowFile,
        "rlTrafficLight": False,
        "saveReplay": True,
        "roadnetLogFile": "roadnet_.json",
        "replayLogFile": "replay_.txt",
    }
    config_name = str(time.time()) + str(np.random.random()) + ".tmp"
    with open(config_name, "w") as f:
        json.dump(config_dict, f)
    return config_name


@click.command()
@click.option("--roadnet_file", default="roadnet_p4a_lt.json", help="roadnetFile")
@click.option("--flow_file", default="flow.json", help="flowFile")
def main(roadnet_file, flow_file):
    if not os.path.isfile(roadnet_file):
        raise FileExistsError("file not exist! check it %s" % roadnet_file)
    if not os.path.isfile(flow_file):
        raise FileExistsError("file not exist! check it %s" % flow_file)

    config_file = save_config_file(roadnet_file, flow_file)
    eng = cityflow.Engine(config_file, 1)
    os.remove(config_file)
    print('start')
    eng.reset()
    for _ in range(360):
        eng.next_step()
        # eng.set_tl_phase('gneJ1', 1)
    print('end.')


if __name__ == '__main__':
    main()
