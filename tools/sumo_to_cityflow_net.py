import os
import click
import sumolib
from collections import OrderedDict
import traci
import json


def get_laneLinks(each_conn_tls, each_conn_inter):
    laneLinks_config = []
    each_laneLink = OrderedDict()

    idx_start = int(each_conn_tls[0].split('_')[-1])
    idx_end = int(each_conn_tls[1].split('_')[-1])
    lanes_cnt_s = len(each_conn_inter.getFrom().getLanes())
    lanes_cnt_t = len(each_conn_inter.getTo().getLanes())
    each_laneLink['startLaneIndex'] = lanes_cnt_s - idx_start - 1
    each_laneLink['endLaneIndex'] = lanes_cnt_t - idx_end - 1
    # each_laneLink['startLaneIndex'] = idx_start
    # each_laneLink['endLaneIndex'] = idx_end

    points_s = each_conn_inter.getFrom().getLanes()[idx_start].getShape()[-1]
    points_e = each_conn_inter.getTo().getLanes()[idx_end].getShape()[0]
    points_list = []
    points_list.append({'x': points_s[0], 'y': points_s[1]})
    points_list.append({'x': points_e[0], 'y': points_e[1]})

    each_laneLink['points'] = points_list
    laneLinks_config.append(each_laneLink)
    return laneLinks_config


def get_roadLinks(inter, links_now):
    """
    https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#plain_connections
    Args:
        inter:
    Returns:
    """
    roadLinks_config = []
    dict_type = {'r': 'turn_right',
                 'l': 'turn_left',
                 's': 'go_straight',
                 't': 'turn'}

    for each_conn in links_now:
        each_links = OrderedDict()
        if len(each_conn) == 0:
            continue
        for conn in each_conn:
            road_start = '_'.join(conn[0].split('_')[:-1])
            road_end = '_'.join(conn[1].split('_')[:-1])

            for c in inter.getConnections():
                s = c.getFrom().getID()
                e = c.getTo().getID()
                d = c.getDirection()
                if s == road_start and e == road_end:
                    road_dir = dict_type[d]
                    break
            else:
                raise SyntaxError('traffic light not match.')
            each_links['type'] = road_dir
            each_links['startRoad'] = road_start
            each_links['endRoad'] = road_end
            #
            laneLinks = get_laneLinks(conn, c)
            each_links['laneLinks'] = laneLinks
        roadLinks_config.append(each_links)
    return roadLinks_config


def get_trafficLight(traci_tls, inter, inter_config):
    #
    for tls_id in traci_tls.getIDList():
        if tls_id == inter.getID():
            links_now = traci_tls.getControlledLinks(tls_id)
            #
            roadLinks_config = get_roadLinks(inter, links_now)
            inter_config['roadLinks'] = roadLinks_config
            inter_config['trafficLight']['roadLinkIndices'] = \
                [i for i in range(len(inter_config['roadLinks']))]

            phase = traci_tls.getCompleteRedYellowGreenDefinition(tls_id)[0]
            phase = phase.getPhases()
            list_lightphases = []
            each_lightphases = OrderedDict()
            each_lightphases['time'] = 5
            each_lightphases['availableRoadLinks'] = []
            list_lightphases.append(each_lightphases)
            for p in phase:
                each_lightphases = OrderedDict()
                each_lightphases['time'] = p.duration
                each_lightphases['availableRoadLinks'] = \
                    [idx for idx, s in enumerate(p.state) if s == 'G']
                list_lightphases.append(each_lightphases)

            inter_config['trafficLight']['lightphases'] = list_lightphases
            inter_config['width'] = 10
            inter_config['virtual'] = False
            break
    else:
        inter_config['roadLinks'] = []
        inter_config['virtual'] = True
    return inter_config


def read_intersection_node(inter):
    inter_config = OrderedDict()
    inter_config['id'] = inter.getID()
    point = inter.getCoord()
    inter_config['point'] = {'x': point[0], 'y': point[1]}
    #
    inter_config['width'] = 0
    roads = inter.getIncoming() + inter.getOutgoing()

    roads_id = [r.getID() for r in roads]
    inter_config['roads'] = roads_id

    inter_config['roadLinks'] = []
    inter_config['trafficLight'] = {}
    inter_config['virtual'] = None

    return inter_config


def read_road(road):
    road_config = OrderedDict()
    road_config['id'] = road.getID()

    points = [road.getFromNode().getCoord(), road.getToNode().getCoord()]
    road_config['points'] = [{'x': x, 'y': y} for x, y in points]

    lanes_cnt = len(road.getLanes())
    road_config['lanes'] = \
        [{'width': road.getLanes()[0].getWidth(),
          'maxSpeed': road.getLanes()[0].getSpeed()}
         for i in range(lanes_cnt)]

    road_config['startIntersection'] = road.getFromNode().getID()
    road_config['endIntersection'] = road.getToNode().getID()
    return road_config


@click.command()
@click.option('--sumo_net', default='cps_multi.net.xml', help='sumo_net')
@click.option('--sumocfg', default='cps_multi.sumocfg', help='sumocfg')
@click.option('--cityflow_net', default='default_net.json', help='cityflow_net')
def main(sumo_net, sumocfg, cityflow_net):
    sumo_net = sumolib.net.readNet(sumo_net)
    sumo_cmd = ["sumo", '-c', sumocfg, "--no-warnings", "--no-step-log",
                "--log", "None.tmp"]
    traci.start(sumo_cmd)
    traci_tls = traci.trafficlight
    # --
    list_intersection = []
    for inter in sumo_net.getNodes():
        inter_config = read_intersection_node(inter)
        # attach traffic light to a inter, update inter_config
        inter_config = get_trafficLight(traci_tls, inter, inter_config)
        list_intersection.append(inter_config)
    # --
    list_road = []
    #
    for road in sumo_net.getEdges():
        road_config = read_road(road)
        list_road.append(road_config)
    # --
    cityflow_config = OrderedDict()
    cityflow_config['intersections'] = list_intersection
    cityflow_config['roads'] = list_road
    with open(cityflow_net, mode='w') as f:
        json.dump(cityflow_config, f, indent=2)
    print('finished. output:%s' % cityflow_net)
    traci.close()
    os.remove("None.tmp")


if __name__ == "__main__":
    print('start.')
    main()
    print('finished.')
