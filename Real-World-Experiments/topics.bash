#!/bin/bash

ros2 topic list | grep 'image_raw' | grep -v 'compressed' | grep -v 'theora'
ros2 topic list | grep cmd_vel
ros2 topic list | grep small_image