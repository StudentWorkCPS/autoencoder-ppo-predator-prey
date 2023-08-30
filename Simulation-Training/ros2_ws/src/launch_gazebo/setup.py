from setuptools import setup
import os
from glob import glob

package_name = 'launch_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name),glob('launch_gazebo/*s*.py')),
        (os.path.join('share',package_name,'launch'),glob('launch/*.py')),
        (os.path.join('share',package_name,'worlds'),glob('worlds/*.world')),
        (os.path.join('share',package_name,'worlds','ignition'),glob('worlds/ignition/*.sdf')),
        (os.path.join('share',package_name,'objects'),glob('objects/*.sdf') + glob('objects/*.config')),
    ],
    install_requires=['setuptools','ros_ign_interfaces','ros_ign_gazebo'],
    zip_safe=True,
    maintainer='Henri Grotzeck',
    maintainer_email='henri_grotzeck@gmx.de',
    description='',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_robot = launch_gazebo.spawn_robot:main',   
        ],
    },
)

