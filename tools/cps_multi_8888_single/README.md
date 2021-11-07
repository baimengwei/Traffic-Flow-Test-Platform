包含了一个丁字路口的实例，道路复用，文件说明如下表。

| 文件名            | 功能与使用                     | 说明                            |
| ----------------- | ------------------------------ | ------------------------------- |
| cps_multi.net.xml | sumo文件：路网                 | 手动原始文件                    |
| cps_multi.rou.xml | sumo文件：路由                 | 由randomTrips.py生成            |
| cps_multi.jpg.xml | sumo文件：背景                 | 手动原始文件                    |
| cps_multi.set.xml | sumo文件：sumocfg配置          | sumocfg文件配置导出             |
| cps_multi.sumocfg | sumo文件：主配置               | 手动原始文件，可用于仿真        |
| default_net.json  | cityflow文件：路网文件         | 由sumo_to_cityflow_net.py生成   |
| default_flow.json | cityflow文件：路由文件         | 由sumo_to_cityflow_rou.py生成   |
| roadnet_.json     | cityflow文件：仿真后的路网文件 | 由generate_cityflow_demo.py生成 |
| replay_.txt       | cityflow文件：仿真后的路由文件 | 由generate_cityflow_demo.py生成 |



其中，roadnet_.json文件和 replay_.txt文件可用于cityflow_web中的仿真。

从仿真结果中，整个路网拥塞，相位错乱。