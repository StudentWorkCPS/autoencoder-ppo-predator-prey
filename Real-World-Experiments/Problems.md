## Multpile Discovery Server
- cd /etc/turtlebot4_discovery
- open setup.bash 
    ROS_DISCOVERY_SERVER="[IP1]:[PORT1];...;[IPN]:[PORTN]"
- open fastdd_discovery_super_client.xml
    prefix have to follow the rule: 44.53.[position-of-server-at-ROS_DISCOVERY_SERVER-2-digits].5f.45.50.52.4f.53.49.4d.41 
    like for "134.34.226.63:11811;134.34.225.128:11811;134.34.225.232:11811" it would be
        -


- ros2 daemon stop; ros2 daemon start



## Camera Problems

- As the camera is on the dock the camera service is stoped 
- so if you want to use it on the dock you have to restart it:
   ros2 service call /[TURTLE-BOT-NAMESPACE]/oakd/start_camera std_srvs/srv/Trigger

ros2 service call /turtlebot4_2/oakd/start_camera std_srvs/srv/Trigger


## Restart turtlebot4 

sudo systemctl restart turtlebot4


### Multiple Discovery server 

1. Multiple discovery server on localhost:

https://docs.ros.org/en/galactic/Tutorials/Advanced/Discovery-Server/Discovery-Server.html

- work when not souring "/etc/turtlebot4_discovery/setup.bash"

- Also works when using the IP-Adress of the Computer 

2. Trying to connect to a single turtlebot 

https://turtlebot.github.io/turtlebot4-user-manual/setup/discovery_server.html

works 

3. Multiple discovery-servers (Turtlebot+Local-dds)

- start a local:

fastdds discovery --server-id 1 --ip-address 127.0.0.1 --port 11811

- New terminal:
 
export ROS_DISCOVERY_SERVER=";127.0.0.1:11811"

ros2 run demo_nodes_cpp talker --ros-args --remap __node:=talker_1

- New terminal:

export ROS_DISCOVERY_SERVER=";127.0.0.1:11811"

ros2 run demo_nodes_cpp listener --ros-args --remap __node:=listener_1

=> chatter is printed to terminal

- New terminal

export ROS_DISCOVERY_SERVER=";127.0.0.1:11811"

rqt_graph

 => shows both nodes but not the topic 

- New terminal 

export ROS_DISCOVERY_SERVER="134.34.225.232:11811;127.0.0.1:11811"

rqt_graph

 => show node from turtlebot and localhost but no topics 


ros2 node info /turtlebot4_3/turtlebot4_node

=> shows complete information including topics 


ros2 topic info /turtlebot4_3/ip

=> Type: std_msgs/msg/String Publisher count: 1 Subscription count: 0

ros2 topic echo /turtlebot4_3/ip

=> data: 134.34.225.232

echo $FASTRTPS_DEFAULT_PROFILES_FILE

=> not set 
Why is that not expected:

from prior test we had problem echo topics if FASTRTPS_DEFAULT_PROFILES_FILE was not set.

hypothese: but ros-daemon was running in the background (from Test 2.) probably uses correct FASTRTPS_DEFAULT_PROFILES_FILE 


- new terminal

ros2 daemon stop;

- previous terminal 

ros2 topic echo /turtlebot4_3/ip
=> WARNING: topic [/turtlebot4_3/ip] does not appear to be published yet
Could not determine the type for the passed topic

ros2 topic list => doesn't show anything

ros2 daemon stop; 

rqt_graph 
=> still shows all node and not topics

ros2 topic echo /chatter
=> topic [/chatter] does not appear to be published yet

- New terminal

export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4_discovery/fastdds_discovery_super_client.xml

ros2 daemon stop;ros2 daemon start;

- previous terminal

ros2 topic echo /turtlebot4_3/ip

=> data: 134.34.225.232 (works)

ros2 topic echo /chatter
=> WARNING: topic [/chatter] does not appear to be published yet

echo $ROS_DISCOVERY_SERVER
=> 134.34.225.232:11811;127.0.0.1:11811

edit /etc/turtelbot4_discovery/fastdds_discovery_super_client.xml

=> add to discoveryServersList
<RemoteServer prefix="44.53.01.5f.45.50.52.4f.53.49.4d.41">
    <metatrafficUnicastLocatorList>
        <locator>
         <udpv4>
            <address>127.0.0.1</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>

=> terminal 

ros2 daemon stop;ros2 daemon start;

=> 

ros2 topic echo /chatter

=> data: 'Hello World: 2220' (Works :) )

ros2 topic echo /turtlebot4_3/ip

=> data: 134.34.225.232 (works :)) 

- new terminal

export FASTRTPS_DEFAULT_PROFILES_FILE=

rqt_graph => show only nodes but not topics 

export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/turtlebot4_discovery/fastdds_discovery_super_client.xml

rqt_graph => show only nodes and also topics

4. Multiple Discovery Server (Turtlebot2 + Turtlebot3)

- ssh to turtlebot4 => 2

sudo nano /usr/sbin/discovery

=> change to

fastdds discovery -i 1 -p 11811

systemctl restart discovery

nano /etc/turtlebot4/setup.bash

change to 

export ROS_DISCOVERY_SERVER=;127.0.0.1:11811

nano /etc/turtlebot4/fastdds_discovery_super_client.bash

change to prefix to 44.53.01.5f.45.50.52.4f.53.49.4d.41

turtlebot4-daemon-restart
turtlebot4-service-restart


ros2 topic list 
=> shows all topic of turltebot but not create3 ?

- Back to PC

change /etc/turtlebot4_discovery/fastdds_discovery_super_client.xml discerySewrverList

<RemoteServer prefix="44.53.01.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>134.34.226.63</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>

export ROS_DISCOVERY_SERVER="134.34.225.232:11811;134.34.226.63:11811"

ros2 topic list

=> shows both turtlebot topics

rqt_graph

=> only shows turtlebot3

echo $FASTRTPS_DEFAULT_PROFILES_FILE

=> is set


- Other experiment with setup (turtlebot3=Id:0,turtlebot2=Id:1)

export ROS_DISCOVERY_SERVER=";134.34.226.63:11811"

ros2 daemon stop


- Next experiment 

etc/turtlebot_discovery/setup.bash is not sources all robot are reseted to normal 127.0.0.1:11811 Ip server with ID=0

 1. Terminal 

    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/swarm_lab/henri-grotzeck-BA/ROS2-predator-prey-real-world/turtlebot2/default.xml

    export ROS_DISCOVERY_SERVER=134.34.226.63:11811

    ros2 topic echo /turtlebot4_2/ip std_msgs/msg/String --no-daemon => data: 134.34.226.63 (Works :))

    ros2 topic list --no-daemon
        =>  /parameter_events
            /rosout

 2. Terminal 

    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/swarm_lab/henri-grotzeck-BA/ROS2-predator-prey-real-world/turtlebot3/default.xml

    export ROS_DISCOVERY_SERVER=134.34.225.232 

    ros2 topic echo /turtlebot4_3/ip std_msgs/msg/String --no-daemon => data: 134.34.225.232 (Works :))

    ros2 topic list --no-daemon
        =>  /parameter_events
            /rosout


    => Running Custom-Python-node also work with this behavior

### Test Multiserver setup for turtlebot 

<participant profile_name="super_client_profile" is_default_profile="true">
            <rtps>
                <builtin>
                <discovery_config>
                        <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
                        <discoveryServersList>
                            <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>127.0.0.1</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>
                        </discoveryServersList>
                    </discovery_config>
                </builtin>
            </rtps>
</participant>
<participant profile_name="server_profile" >
            <rtps>
                <prefix>44.53.01.5f.45.50.52.4f.53.49.4d.41</prefix>
                <builtin>
                    <discovery_config>
                        <discoveryProtocol>Server</discoveryProtocol>
                        <discoveryServersList>
                            <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>127.0.0.1</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>
                        </discoveryServersList>
                    </discovery_config>
                    <metatrafficUnicastLocatorList>
                        <locator>
                            <udpv4>
                            <!-- placeholder server UDP address -->
                                <address>127.0.0.1</address>
                                <port>11888</port>
                            </udpv4>
                        </locator>
                    </metatrafficUnicastLocatorList>
                </builtin>
            </rtps>
</participant>

## Problem with a single Turtlebot 

when rebooting the robot 