import argparse
import json
import os

from lxml import etree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=str, default=None)
    parser.add_argument("--roadnet", type=str, default=None)
    parser.add_argument("--sumonet", type=str, default='default.net.xml')
    parser.add_argument("--sumorou", type=str, default='default.rou.xml')
    parser.add_argument("--sumocfg", type=str, default='default.sumocfg')
    return parser.parse_args()


def create_file_node(args, roadnet):
    intersections = roadnet['intersections']
    root = etree.Element("nodes")
    for each_inter in intersections:
        inter_id = each_inter['id']
        point = each_inter['point']
        type_ = each_inter['virtual']
        dict_node = {
            'id': inter_id,
            'x': str(point['x']),
            'y': str(point['y']),
        }
        if type_:
            type_ = 'priority'
            dict_node.update({'type': type_})
        else:
            type_ = 'traffic_light'
            dict_node.update({'type': type_})
            dict_node.update({'tl': inter_id})
        etree.SubElement(root, 'node', dict_node)
    etree.ElementTree(root).write(str(args.sumonet).split('.')[0] + '.nod.xml',
                                  pretty_print=True)
    print('create_file_node ok.')


def create_file_edge(args, roadnet):
    roads = roadnet['roads']
    root = etree.Element('edges')
    for each_road in roads:
        road_id = each_road['id']
        points = each_road['points']
        lanes = each_road['lanes']
        start_intersection = each_road['startIntersection']
        end_intersection = each_road['endIntersection']

        dict_edge = {
            'id': road_id,
            'from': start_intersection,
            'to': end_intersection,
            'priority': "3",
            'numLanes': str(len(lanes)),
            'speed': str(lanes[0]['maxSpeed'])
        }
        etree.SubElement(root, 'edge', dict_edge)
    etree.ElementTree(root).write(str(args.sumonet).split('.')[0] + ".edg.xml",
                                  pretty_print=True)
    print('create_file_edge ok')
    pass


def create_file_connect(args, roadnet):
    inters = roadnet['intersections']
    root = etree.Element('connections')
    for each_inter in inters:
        if not each_inter['virtual']:
            list_links = each_inter['roadLinks']
            for each_link in list_links:
                start_road = each_link['startRoad']
                end_road = each_link['endRoad']
                for each_pair in each_link['laneLinks']:
                    start_idx = each_pair['startLaneIndex']
                    end_idx = each_pair['endLaneIndex']
                    start_idx = len(each_link['laneLinks']) - 1 - start_idx
                    end_idx = len(each_link['laneLinks']) - 1 - end_idx
                    if start_idx != end_idx:
                        continue
                    connect_item = {
                        "from": start_road,
                        "to": end_road,
                        "fromLane": str(start_idx),
                        "toLane": str(end_idx),
                    }
                    etree.SubElement(root, 'connection', connect_item)
    etree.ElementTree(root).write(str(args.sumonet).split('.')[0] + '.con.xml',
                                  pretty_print=True)
    print('create_file_connect ok.')


def create_file_trafficlight(args, roadnet):
    inters = roadnet['intersections']
    root = etree.Element('tlLogics')
    for each_inter in inters:
        inter_id = each_inter['id']
        if not each_inter['virtual']:
            list_links = each_inter['roadLinks']
            dict_tl = {'id': inter_id, 'type': "static",
                       'programID': "0", 'offset': "0", }
            tree_tl = etree.SubElement(root, 'tlLogic', dict_tl)
            list_phase = each_inter['trafficLight']['lightphases']

            for idx, each_phase in enumerate(list_phase):
                # ---------------------add tl---------------------------------
                time_ = each_phase["time"]
                available = each_phase["availableRoadLinks"]
                state = ''
                for i in range(len(list_phase)):
                    if i == idx and len(available) > 0:
                        state += 'G'
                    else:
                        state += 'r'
                tl_item = {"duration": str(time_), "state": state}
                etree.SubElement(tree_tl, 'phase', tl_item)
                # ---------------------add link------------------------------
                for each_available in available:
                    each_link = each_inter['roadLinks'][each_available]
                    start_road = each_link["startRoad"]
                    end_road = each_link["endRoad"]
                    for link_pair in each_link["laneLinks"]:
                        start_idx = link_pair['startLaneIndex']
                        end_idx = link_pair['endLaneIndex']
                        start_idx = len(each_link['laneLinks']) - 1 - start_idx
                        end_idx = len(each_link['laneLinks']) - 1 - end_idx
                        if start_idx != end_idx:
                            # because the cityflow config is a mess
                            # if not jump, the connection will be free more.
                            continue
                        connect_item = {
                            "from": start_road,
                            "to": end_road,
                            "fromLane": str(start_idx),
                            "toLane": str(end_idx),
                            "tl": inter_id,
                            "linkIndex": str(idx)
                        }
                        etree.SubElement(root, 'connection', connect_item)

    etree.ElementTree(root).write(str(args.sumonet).split('.')[0] + '.tll.xml',
                                  pretty_print=True)
    print('create_file_trafficlight ok.')


def create_file_net(args):
    cmd = "netconvert -n %s -e %s -x %s -i %s -o %s" % (
        str(args.sumonet).split('.')[0] + ".nod.xml",
        str(args.sumonet).split('.')[0] + ".edg.xml",
        str(args.sumonet).split('.')[0] + ".con.xml",
        str(args.sumonet).split('.')[0] + ".tll.xml",
        str(args.sumonet))
    os.system(cmd)


def create_file_flow(args, flow_info):
    root = etree.Element('routes')
    for idx, flow_each in enumerate(flow_info):
        type_item = {
            "id": "type_" + str(idx),
            "length": str(flow_each["vehicle"]["length"]),
            "width": str(flow_each["vehicle"]["width"]),
            "accel": str(flow_each["vehicle"]["usualPosAcc"]),
            "decel": str(flow_each["vehicle"]["usualNegAcc"]),
            "minGap": str(flow_each["vehicle"]["minGap"]),
            "maxSpeed": str(flow_each["vehicle"]["maxSpeed"]),
            "tau": str(flow_each["vehicle"]["headwayTime"]),
            "emergencyDecel": str(flow_each["vehicle"]["maxNegAcc"]),
        }
        etree.SubElement(root, "vType", type_item)
        trip_item = {
            "id": "vehicle_" + str(idx),
            "depart": str(flow_each["startTime"]),
            "from": flow_each["route"][0],
            "to": flow_each["route"][1],
            "color": "cyan",
            "type": "type_" + str(idx),
            "begin": str(flow_each["startTime"]),
            "end": str(flow_each["endTime"]),
            "period": str(flow_each["interval"])
        }
        etree.SubElement(root, "trip", trip_item)
    etree.ElementTree(root).write(args.sumorou, pretty_print=True)
    print('create_file_flow ok')


def create_file_sumocfg(args):
    root = etree.Element("configuration")
    input_ = etree.SubElement(root, "input")
    etree.SubElement(input_, "net-file", {"value": args.sumonet})
    etree.SubElement(input_, "route-files", {"value": args.sumorou})
    time_ = etree.SubElement(root, "time")
    etree.SubElement(time_, "begin", {"value": str(0)})

    etree.ElementTree(root).write(args.sumocfg, pretty_print=True)
    print('create_file_sumocfg ok')


def main(args):
    print("from cityflow to json")
    if args.roadnet is not None:
        with open(args.roadnet, 'r') as f:
            roadnet = json.load(f)
        create_file_node(args, roadnet)
        create_file_edge(args, roadnet)
        create_file_connect(args, roadnet)
        create_file_trafficlight(args, roadnet)
        create_file_net(args)
    if args.flow is not None:
        with open(args.flow, 'r') as f:
            flow_info = json.load(f)
        create_file_flow(args, flow_info)

    if args.flow is not None and args.roadnet is not None:
        create_file_sumocfg(args)


if __name__ == '__main__':
    print('because of using the converter.py which is a tool written by sumo')
    print('there may be some difference with the origin file(cord)')
    args = parse_args()
    main(args)
    print("finished.")
