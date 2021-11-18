文件说明：

| 文件名               | 功能                                 | 实现情况                                                     |
| -------------------- | ------------------------------------ | ------------------------------------------------------------ |
| sumo_to_cityflow_net | 和原始的CityFlow相同，               | 从CityFlow代码中摘取并修剪，只可转换sumo的net文件到cityflow  |
| sumo_to_cityflow_rou | 解析sumo文件并生成CityFlow的车辆路由 |                                                              |
| cityflow_to_sumo     | 手动实现的转换                       | 转换cityflow的net文件和flow文件到sumo对应的三个文件（net.xml， rou.xml，sumocfg文件），可能有bug。 |
| cityflow_web         | cityflow可视化最终的路径文件         | https://cityflow.readthedocs.io/en/latest/replay.html        |
| randomTrips          | 为sumo网络文件随机生成车辆           | 示例：python randomTrips.py -n T.net.xml -r T.rou.xml -p 2   |

