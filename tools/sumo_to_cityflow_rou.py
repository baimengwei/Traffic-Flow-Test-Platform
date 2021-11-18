import json

import click
import xml.etree.ElementTree as ET


def get_default_vehicle():
    dic_vehicle = {
        "vehicle": {
            "length": 5.0,
            "width": 2.0,
            "maxPosAcc": 2.0,
            "maxNegAcc": 4.5,
            "usualPosAcc": 2.0,
            "usualNegAcc": 4.5,
            "minGap": 2.5,
            "maxSpeed": 11.11,
            "headwayTime": 2.0
        },
        "route": [
        ],
        "interval": 5,
        "startTime": 0,
        "endTime": 0
    }
    return dic_vehicle


@click.command()
@click.option("--sumo_rou", default="cps_multi.rou.xml",
              help="source, sumo route")
@click.option("--cityflow_rou", default="default_flow.json",
              help="target, cityflow route")
def main(sumo_rou, cityflow_rou):
    tree = ET.parse(sumo_rou)
    root = tree.getroot()

    list_target = []
    if list(root)[0].attrib['id'].split('_')[0] == 'type':
        # format: vehicle
        for vehicle in root:
            dic_vehicle = get_default_vehicle()
            if vehicle.attrib['id'].split('_')[0] != 'vehicle':
                continue
            start_time = vehicle.attrib['depart']
            list_route = [vehicle.attrib["from"], vehicle.attrib["to"]]
            dic_vehicle["startTime"] = int(float(start_time))
            dic_vehicle["endTime"] = int(float(start_time))
            dic_vehicle["route"] = list_route
            list_target.append(dic_vehicle)
    else:
        # format: trip
        for vehicle in root:
            dic_vehicle = get_default_vehicle()
            start_time = vehicle.attrib['depart']
            list_route = vehicle[0].attrib["edges"].split(' ')
            dic_vehicle["startTime"] = int(float(start_time))
            dic_vehicle["endTime"] = int(float(start_time))
            dic_vehicle["route"] = list_route
            if len(list_route) <= 1:
                continue
            list_target.append(dic_vehicle)

    with open(cityflow_rou, 'w') as f:
        json.dump(list_target, f, indent=2)
    print('finished. output:%s' % cityflow_rou)


if __name__ == '__main__':
    print('start convert.')
    main()
    print('finished.')
