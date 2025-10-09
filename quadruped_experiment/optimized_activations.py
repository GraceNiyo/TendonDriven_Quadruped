
import numpy as np 
import matplotlib.pyplot as plt
import generate_desired_kinematics as gdk

import mujoco
from scipy.optimize import minimize



def compute_inverse_torque(model, data, qpos, qvel, qacc):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qacc[:] = qacc
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse.copy()
