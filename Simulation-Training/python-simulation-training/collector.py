import gym_env.env_utils as env_utils
from gym_env.basic_predator_prey_gym import BasicPredatorPreyEnv
from gym_env.robot import Robot

import time
import numpy as np

import utils
import cv2
        
import argparse


env = BasicPredatorPreyEnv(predator_num=3,prey_num=1,headless=False,model='turtlebot4-2.0')

# Stages Needed for the collector:
# 1. Randomize the robots => Only take photos when at least one robot is in the image
# 2. Corner Images => For every Corner on robot
# 3. Corner with Robot Images
# 4. Near robots => Only take photos when at least one robot is in the image
# 5. Near robots with robots => Only take photos when at least one robot is in the image



class Stage():

    def __init__(self, image_count,name) -> None:
        self.robots : list(Robot) = env.robots
        self.last_obs = [self.robots[i].observation for i in range(len(self.robots))]
        self.name = name
        self.image_count = image_count

        self.placer = lambda robot,i: robot.set_state(env_utils.State(env_utils.row(i,4),env_utils.row(i,4),0,0,0,0))
        self.can_make_photo = lambda robot,i: True

        self.n_digits = len(str(image_count))
    
        utils.create_dir_if_not_exists(f'collector/{self.name}')
        utils.clear_dir(f'collector/{self.name}')

    def make_photo(self,idx,number):
        
        while (self.robots[idx].observation == self.last_obs[idx]).all():
            time.sleep(0.1)
        
        img = cv2.cvtColor(self.robots[idx].observation,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'collector/{self.name}/img-{utils.number_to_n_digits(number,self.n_digits)}.png',img)

        self.last_obs[idx] = self.robots[idx].observation

    def make_photos(self):
        total_images = 0
        while total_images < self.image_count :

            for i,robot in enumerate(self.robots):
                self.placer(robot,i)

            
            env.env_states[0]['done'] = True
            env.reset()

            time.sleep(0.1)

            for i,robot in enumerate(self.robots):
                if self.can_make_photo(robot,i):
                    self.make_photo(i,total_images)
                    total_images += 1
                    print(f'Image {total_images}/{self.image_count} made in ' + self.name + ' stage')
            

    def run(self):
        self.setup()

        self.make_photos()

    def setup(self):
        for i,robot in enumerate(self.robots):
            robot.set_intial_state(env_utils.State.zero_state().out_of_arena(i))
        
        env.reset()
        

        

    # Builder Functions for the stage
    
    def only_n_robots(self,n):
        
        current_placer = self.placer
        def can_make_photo(robot,i):
            return i < n
        
        def placer(robot,i):
            if i < n:
                current_placer(robot,i)

        self.can_make_photo = can_make_photo    
        self.placer = placer    

        return self

    # Placing Conditions

    def place_corner(self,yaw_sigma=np.pi,pos_sigma=0.2):
        corner_x = [1.5,1.5,-1.5,-1.5]
        corner_y = [1.5,-1.5,1.5,-1.5]
        
        def corner_placer(robot,i):
            x = corner_x[i] + np.random.uniform(-pos_sigma,pos_sigma)
            y = corner_y[i] + np.random.uniform(-pos_sigma,pos_sigma)
            yaw = np.arctan2(corner_y[i],corner_x[i])  + np.random.uniform(-yaw_sigma,yaw_sigma)

            robot.set_intial_state(env_utils.State(x,y,0,0,0,yaw))


        self.placer = corner_placer

        return self
        
    def randomize(self):
        randomizer = lambda robot,i: robot.set_intial_state(env_utils.State(0,0,0,0,0,0).randomize())
        self.placer = randomizer

        return self

    

    # Photo Conditions
    
    def robot_in_view(self):

        def can_make_photo(robot,i):
            return env_utils.has_color(robot.observation,env_utils.RED) or env_utils.has_color(robot.observation,env_utils.GREEN)

        self.can_make_photo = can_make_photo

        return self

    def prey_in_view(self):

        def can_make_photo(robot,i):
            return env_utils.has_color(robot.observation,env_utils.GREEN)

        self.can_make_photo = can_make_photo

        return self    



def main():
    
    args = argparse.ArgumentParser()
    
    args.add_argument('--images',type=int,default=10000)
    
    args = args.parse_args()
    images = int(args.images)

    stages = [  
                Stage(images * 0.2,'corner').place_corner(yaw_sigma=np.pi,pos_sigma=0.1),
                Stage(images * 0.75,'random').randomize().robot_in_view(),
                Stage(images * 0.05,'only_1').randomize().only_n_robots(1), 
            ]
    


    for stage in stages:
        stage.run()

        print(f'Finished {stage.name} stage')
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

    env.close()