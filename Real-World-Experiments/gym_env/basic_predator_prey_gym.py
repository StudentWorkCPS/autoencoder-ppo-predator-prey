# Global imports
import os
import gym 
import time
import asyncio
import numpy as np
from threading import Thread

# ROS2 imports
import rclpy
import launch_ros.actions
from std_srvs.srv import Empty


# Local imports
from Debug import info,warn,error
import gym_env.env_utils as env_utils
from gym_env.robot import Robot


class Config:

    def __init__(self,robot_namespaces =lambda type,i: 'env0_{type}{i}',cmd_topic='/cmd_vel',camera_topic='/camera/image_raw') -> None:
        
        self.robot_namespaces = robot_namespaces
        self.cmd_topic = cmd_topic

        self.camera_topic = camera_topic




    

class BasicPredatorPreyEnv(gym.Env):

    metadata = {'render.modes':['human']}

    def __init__(self,predator_num,prey_num,env_num=None, max_steps=180,randomize_initial_state=[],headless=False,world_path='Arena/model.sdf',real_world:bool=False, custom_config=Config()):
        super(BasicPredatorPreyEnv,self)

        self.predator_num = predator_num
        self.prey_num = prey_num
        
        self.env_num = env_num if env_num is not None else 1
        self.use_old_state = env_num is None


        self.headless = headless
        gui_str = 'false' if headless else 'true'
        
        info('Initializing ROS2 node...')
        rclpy.init()
        self.ros_node = rclpy.create_node('predator_prey_gym_node')
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.ros_node)

        self.real_world = real_world

        self.ros2_topic_config = custom_config

        if not real_world:
            info('Gazebo starting...')
            os.system('gnome-terminal -- sh -c "ros2 launch launch_gazebo launch_sim.py gui:='+ gui_str + ' env_num:=' + str(self.env_num) + ' predator_num:='+str(predator_num)+' prey_num:='+str(prey_num)+'; exec bash"')
            info('Gazebo started')
        
            from gazebo_msgs.srv import SetEntityState
            from gazebo_msgs.msg import ModelStates
            
        
            info('Initializing Gazebo services...')

            self.pause_srv = env_utils.create_client_and_wait_for_service(self.ros_node,Empty,'/pause_physics')
            self.unpause_srv = env_utils.create_client_and_wait_for_service(self.ros_node,Empty,'/unpause_physics')
            self.reset_srv = env_utils.create_client_and_wait_for_service(self.ros_node,Empty,'/reset_world')
            self.set_agent_pose_srv = env_utils.create_client_and_wait_for_service(self.ros_node,SetEntityState,'/gazebo/set_entity_state')
        
            info('Initializing ROS2 executor...')
            state_node = rclpy.create_node('state_node')

            self.state_subcription = state_node.create_subscription(ModelStates,'/gazebo/model_states',self.state_callback,1)

        
            self.executor.add_node(state_node)
        


        self.env_states = [{'done':False,'steps':0,'caught':False,'predator_caught':-1,'max_steps':max_steps} for i in range(self.env_num)]

        info('Initializing Robot publishers...')
        
        self.robots : list[Robot] = []
        # For easy access of robots
        self.name_to_robot = {}
        
        # ['predator','prey']
        self.randomize_initial_state = randomize_initial_state
        for eid in range(self.env_num):
            for i in range(predator_num):
                self.setup_robot('predator',i,eid)
            for i in range(prey_num):
                self.setup_robot('prey',i,eid)
        
        #os.system('ros2 run gazebo_ros spawn_entity.py -file robot_test.sdf -entity robot')
        info('Intializing done')
        executor_thread = Thread(target=self.executor.spin, daemon=True)
        executor_thread.start()

        self.timer = self.ros_node.create_rate(10)
        self.last_reset_time = time.time()
        self.last_step_time = time.time()
        self.steps = 0
        self.max_steps = max_steps
        self.state_updated = True
        self.fps = 0
        

        
        wait_robots = [robot for robot in self.robots]
        info('Waiting for robots to be spawned')
        while len(wait_robots) > 0:
            self.timer.sleep()
            for i in reversed(range(len(wait_robots))):
                robot = wait_robots[i]
                if robot.observation is not None:
                    wait_robots.remove(robot)
            
            print('Waiting for robots to be spawned' if not self.real_world else f'Waiting for robots ({wait_robots}) to communicate')
        info('Robots spawned')
            

        
    def setup_robot(self,robot_type,idx,eid=0):

        if  robot_type == 'predator':
            initial_state = env_utils.State(-1.0,(idx - self.predator_num/2 + 1/2)*0.4,0.0,0.0,0.0,0.0)
        else:
            initial_state = env_utils.State(1.0,(idx - self.prey_num/2 + 1/2)*0.4,0.0,0.0,0.0,np.pi)

        
        robot_name = 'env{}_{}{}'.format(eid,robot_type,idx)
        print('Setting up robot',robot_type,self.randomize_initial_state)

        robot_namespace = self.ros2_topic_config.robot_namespaces(robot_type,idx)
        cmd_topic = robot_namespace + self.ros2_topic_config.cmd_topic
        camera_topic = robot_namespace + self.ros2_topic_config.camera_topic

        robot = Robot(self.ros_node,robot_name,initial_state,self.executor, robot_type in self.randomize_initial_state,env_id=eid,real_world=self.real_world,cmd_topic=cmd_topic,camera_topic=camera_topic,namespace=robot_namespace)
        # Standard access to robots
        self.robots.append(robot)
        # Fast access to robots
        self.name_to_robot[robot_name] = robot        

    # Update Position and current velocity of all robots
    def state_callback(self,msg):
        
        for i in range(len(msg.name)):
            name = msg.name[i]
            if name in self.name_to_robot:

                vels = env_utils.twist_to_vels(msg.twist[i])
                robot : Robot = self.name_to_robot[name] 
                robot.current_state = env_utils.pose_to_state(msg.pose[i],robot.current_state)

                robot.set_current_vels(vels)

        self.state_updated = True
 

    def step(self,action):
        self.fps = 1/(time.time() - self.last_step_time)
        print('FPS',self.fps)
        self.last_step_time = time.time()

        for i in range(len(self.robots)):
            #print("publishing robot " + str(i),action[i])
            if self.env_states[self.robots[i].env_id]['done']:
                action[i] = [0.0,0.0]

            self.robots[i].publish_vels(action[i])

        # Wait for results
        loop = True
        info('Waiting for results')
        while loop:
            
            time.sleep(0.01)
            loop = False
            for i in range(len(self.robots)):
            
                if self.robots[i].new_img is False:
                    loop = True
                    break
        info('Results received')
        
        observations = []
        states = []
        vels = []
        done = False
        caught = False
        
        for i in range(len(self.robots)):
            observations.append(self.robots[i].get_observation())
            states.append(self.robots[i].current_state)
            vels.append(self.robots[i].last_vels)

        
        # Test wether the prey is caught   
        
        for env_id in range(self.env_num):
                done = False
                caught = False
                predator_caught = -1
                offset = env_id * (self.predator_num + self.prey_num)

                for i in range(self.predator_num):
                    for j in range(self.prey_num):
                        id = offset + i
                        id2 = offset + self.predator_num + j
                        #print('Checking if predator {} caught prey {}'.format(id,id2))
                        #print('Distance',len(states),id,id2,env_id)
                        if env_utils.dist(states[id],states[id2]) < 0.25:
                            done = True
                            caught = True
                            predator_caught = i
                            break 
                    # If caught, break the outer loop as well (https://stackoverflow.com/questions/189645/how-can-i-break-out-of-multiple-loops)
                    else:
                        continue
                    break
        
                                    # 6 FPS * 30 seconds

                env_state = self.env_states[env_id]
                
                if self.state_updated:
                    done = done or env_state['steps'] >= env_state['max_steps']
                    env_state['predator_caught'] = predator_caught
                    env_state['done'] = done
                    env_state['caught'] = caught

                env_state['steps'] = env_state['steps'] + 1

        
        if self.use_old_state:
            return observations , {'states':states,'velocities':vels,'predator_caught':predator_caught, 'FPS':self.fps} , done , caught
    
        return observations , {'states':states,'velocities':vels,'predator_caught':predator_caught, 'FPS':self.fps} , self.env_states 

    def pause(self):
        # Pause simulation
        info('Pausing simulation...')
        future = self.pause_srv.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.ros_node, future)

    def unpause(self):
        # Unpause simulation
        info('Unpausing simulation...')
        future = self.unpause_srv.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.ros_node, future)

    def reset(self):
        #print('Last Rest Seconds',time.time() - self.last_reset_time)
        info('Resetting simulation...')
        
        #self.pause()
        #Set robots to initial position

        info('Resetting robots...')
        futures = []
        envs_to_reset = set()
        for i in range(len(self.robots)):
            robot = self.robots[i]
            
            if self.env_states[robot.env_id]['done']:
                model_state = robot.get_initial_state()
                info('Resetting robot: '+robot.robot_name)
                future = self.set_agent_pose_srv.call_async(model_state)
                rclpy.spin_until_future_complete(self.ros_node, future)
                envs_to_reset.add(robot.env_id)

        for env_id in envs_to_reset:
            self.env_states[env_id]['predator_caught'] = -1
            self.env_states[env_id]['steps'] = 0
            self.env_states[env_id]['done'] = False
            self.env_states[env_id]['caught'] = False
            self.env_states[env_id]['max_steps'] = self.max_steps

        
        # Unpause simulation
        #self.unpause()
        self.last_reset_time = time.time()
        self.state_updated = False
        self.steps = 0
        return [robot.observation for robot in self.robots] 


    def stop(self):
        info('Stopping simulation...')
        self.ros_node.destroy_node()
        #rclpy.
        rclpy.shutdown()
        os.system('killall -9 gazebo & killall -9 gzserver & killall -9 gzclient')


    def render(self,mode='human',close=False):
        print('rendering')


        return