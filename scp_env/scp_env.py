from turtle import pos
import time
import subprocess
import gymnasium as gym
import numpy as np
import socket
import json
import time
import mss
from Xlib import display
import cv2
from sympy import rad
from input_controller import InputController
from gymnasium import spaces
import random
import mss
import cv2
import numpy as np
import subprocess
import re
import orjson
import os

from .comunicacion import Comunicacion

class Scp_env:
    
    def __init__(self, host = "localhost", port = 7900, agent_id = 0, num_agents = 0):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.num_agents = num_agents
        
        self.comunicacion = Comunicacion(host, port, agent_id, num_agents)