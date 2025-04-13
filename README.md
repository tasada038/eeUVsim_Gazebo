# eeUVsim: Learning friendly ROS2 AUV simulator
**Now the related paper is under the review, official release is after the decision**
Fossen' physics model based multiple platform simulator including dynamics, reinforcement learning script, robot models, actuator models.

eeUVsim has task oriented RL training script with randomize and repeating function
![image](https://github.com/hama6767/pubdata/blob/main/Peek%202024-09-23%2013-52.gif?raw=true)

The training environment is impelemted based on ros2 pub-sub method as actual robot software basically has. Then it is easy to transfer to real robot from simulator.
![image](https://github.com/hama6767/pubdata/blob/main/Peek%202024-09-23%2015-22.gif?raw=true)



# Installation
### Tested environment
| Ubuntu version                  | ROS2 Version | Comment                                                  |
| -------------------------- | ------- | ------------------------------------------------------------ |
| 22.04                  | Humble   |  Recommnded                                                            |
| 20.04                  | Galactic   |                                                              |

### Depended libraries
| Package                                                      | Version      | Comment                                                      |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| Python                                                       | 3.10         | Basically should work with ubuntu default version            |
| ROS2                                                         | (Above)      |                                                              |
| Gazebo                                                       | 11 (Classic) | It is **only used for visualization without physics**        |
| Gym                                                          | 0.26.2       | It used for the reinforcement learing environment            |
| Stable-baselines3                                            | 2.3.2        | Used to train reinforcement learning agent                   |
| sb3_contrib                                                  | 2.3.0        | Used for LSTM-PPO training                                   |


```sh
cd ./src/eeUVsim_Gazebo
pip install -r requirements.txt
```

### Quick start
In UCAT case, you can launch fundamental scripts (motion controller, dynamics, robot model) with this command.

`ros2 launch eeuv_sim spawn_UCAT.launch.py`

When you want to train surfacing controller, you can run

`ros2 run eeuv_sim RL_UCAT.py -o`

Then iterative training will start.