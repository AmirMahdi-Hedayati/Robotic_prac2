import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/amirmahdi/Mothari_robotics/Robotic_prac2/robotic_course/install/robot_local_localization'
