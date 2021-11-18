#### How to run

check and modify the main.py in the root folder, then run it.



#### How to use

this is still an on going project, some algorithm might be not implement or occur some error. but the idea is simple, you can use it to test your idea, add new engine except for sumo and cityflow, add your specific map mode, develop new state, action, reward information, and so on. 

**this project is NOT implement fully, but can be used now**



#### folder describe

| folder  | describtion                                 |
| ------- | ------------------------------------------- |
| algs    | algorithm                                   |
| data    | traffic files and information               |
| configs | config class                                |
| misc    | tools maybe used by the project             |
| envs    | environment including sumo and cityflow     |
| records | the output on the running process           |
| tmp     | for debuging, tesing the gramma, and so on. |



#### file describe

| file name | information   |
| --------- | ------------- |
| main.py   | main entrance |
|           |               |



#### Structure of round learner

![round_learner](picture\round_learner.png)



#### the UML of the class

![uml_class](picture\uml_class.jpg)



#### 效果

sumo

![ecust_compus_sumo](picture\ecust_compus_sumo.png)

cityflow

![ecust_compus_cityflow](picture\ecust_compus_cityflow.png)

Curve example in dqn

<img src="picture\dqn_vehicle.png" alt="dqn_vehicle" style="zoom:30%;" />

<img src="picture\dqn_reward.png" alt="dqn_reward" style="zoom:30%;" />

#### Note

本代码框架部分思路来源于，感谢他们在算法和程序上做的贡献。

https://github.com/gjzheng93/frap-pub

https://github.com/zxsRambo/metalight

CityFlow

https://github.com/cityflow-project/CityFlow

Sumo

https://www.eclipse.org/sumo/



#### Contract

Mr. Bai：1872040489@qq.com

