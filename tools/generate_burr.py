import click
import sumolib.net
from scipy.stats import burr12
import math
import numpy as np


def gen_burr_demand(self):
    # https://docs.scipy.org/doc/scipy/reference/generated/
    # scipy.stats.burr12.html
    c = self.rain_dict['c']
    d = self.rain_dict['d']
    beta = self.rain_dict['scale']
    vlim_min = self.vehicle_params[self.method]['vlim_min']
    vlim_max = self.vehicle_params[self.method]['vlim_max']

    v_schedule = burr12.rvs(c, d, scale=beta,
                            size=self.sim_len) / 60 * self.scale
    v_schedule_temp = 0
    for idx, each in enumerate(v_schedule):
        if each < 1:
            v_schedule_temp += each
            v_schedule[idx] = 0
        else:
            v_schedule_temp += math.modf(each)[0]
            v_schedule[idx] = int(each)
        while v_schedule_temp >= 1:
            v_schedule[idx] += 1
            v_schedule_temp -= 1

    v_schedule = np.array(v_schedule, dtype=np.int32)
    v_schedule[v_schedule > 20] = vlim_max
    v_schedule[v_schedule < 0] = vlim_min

    if self.config.vehicle_params['debug_plot']:
        plt.plot(v_schedule)
        plt.xlabel('time')
        plt.xlabel('number of create vehicle')
        plt.show()

    v_schedule = [np.random.choice(self.origins,
                                   size=int(self.scale * n_veh),
                                   replace=True)
                  if n_veh > 0 else [] for n_veh in v_schedule]
    return v_schedule.__iter__()


@click.command()
@click.option('--file_roadnet', prompt='roadnet', help='sumo roadnet file')
@click.option('--file_flow', prompt='flowfile', help='sumo flow file')
def get_config_edge(file_roadnet, file_flow):
    roadnet = sumolib.net.readNet(file_roadnet)




if __name__ == '__main__':
    get_config_edge()
    print('1')
