# Autoencoder-PPO approach for Predator and Prey Learning in Simulation and Reality

This is the repositiory for the Bachelor's Thesis titled "Autoencoder-PPO approach for Predator and Prey Learning in Simulation and Reality" by Henri Grotzeck. The repo is structures seperated into two parts: the Simulation-Training and the Real World Experiments.

## Simulation Training

### ROS2 Simulation

Running the simulation with the robots requires ROS2 humble and gazebo 11 classic.

#### Build Simulation 

``` bash 
cd Simulation-Training/ros_ws

colcon build --symlink-install 
```

It may also require you to move all model files off the Turtlebot4 to the gazebo models folder.

#### Running the Simulation 

Running the simulation can be done by simply createing an instance of the BasicPredatorPreyEnv gym enviroment. This can be found in the file `Simulation-Training/python-simulation-training/gym_env/basic_predator_prey_gym.py`.

e.g.:
```python
BasicPredatorPreyEnv(PREDATOR_NUM,PREY_NUM,max_steps=1000,initial_state_fn=initial_state,env_num=1,model='turtlebot4')
```


### Image Collection 
To run the pipeline of collecting the images in the simulation one has to run:
```
cd Simulation-Training/python-simulation-training
python3 collector.py
```


### Training the autoencoder 
Training a autoencoder can be done by:

```
cd Simulation-Training/autoencoder-training
python3 training.py
```

### Training the policies
Training the policies is done by:

```
cd Simulation-Training/python-simulation-training
python3 ray-lin-ppo-train.py
```

##  Real-World-Experiments

This is the code used to run and evaluating the real world experiments.

### Running in the real world

Setup turtlebot conenction in current bash terminal:
e.g.:
```
cd Real-World-Experiments/turtlebot1
source setup.bash
```

Run the appliaction for example as predator on turtlebot1:
```
cd ..
python3 main.py --type predator --num 1
```

### Camera calibartion

When using the go-pro Hero 7 the distortion matrix can be used so there is no need to start the undistortion calibartion. But the calibration for reprojecting can be done with the file `Real-World-Experiments/camera-evaluation/calibration.py`. Here the real world position and video have to be replaced.

### Data collection 
To get the positions for each robot from the videos one can run:
```
cd Real-World-Experiments/camera-evaluation
python3 go-sky.py
```






