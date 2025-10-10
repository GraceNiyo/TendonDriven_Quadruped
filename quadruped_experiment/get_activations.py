
import numpy as np
import matplotlib.pyplot as plt
import mujoco 
import mujoco.viewer
import time
import desired_kinematic as dsk

import force_length_velocity_functions as flv
import os
import numpy as np
from scipy.optimize import minimize

def solve_tendon_forces(RT, tau):
    m = RT.shape[1]
    def obj(F): return np.sum(F**2)
    cons = {'type': 'eq', 'fun': lambda F: RT @ F - tau}
    min_force = 0
    bounds = [(min_force, None)] * m
    F0_guess = np.ones(m)
    res = minimize(obj, F0_guess, constraints=cons, bounds=bounds)
    if res.success:
        return res.x
    else:
        raise RuntimeError('QP solver failed')


def invert_flv(F_desired, FL, FV, FP, F0):
    a = np.zeros_like(F_desired)
    valid_indices = (FL > 0) & (FV > 0)
    a[valid_indices] = (F_desired[valid_indices] / F0[valid_indices] - FP[valid_indices]) / (FL[valid_indices] * FV[valid_indices])
    return np.clip(a, 0, 1)


def fill_actuator_moments(num_muscles, num_joints,actuator_moments):

    if num_joints == 8:
        muscle_joint_map = [
            [0, 1],    # muscle 0 → joints 0,1
            [0],       # muscle 1 → joint 0
            [0, 1],    # muscle 2 → joints 0,1
            [2, 3],    # muscle 3 → joints 2,3
            [2],       # muscle 4 → joint 2
            [2, 3],    # muscle 5 → joints 2,3
            [4, 5],    # muscle 6 → joints 4,5
            [4],       # muscle 7 → joint 4
            [4, 5],    # muscle 8 → joints 4,5
            [6, 7],    # muscle 9 → joints 6,7
            [6],       # muscle 10 → joint 6
            [6, 7],    # muscle 11 → joints 6,7
        ]
    elif num_joints == 9:
        muscle_joint_map = [
        [1, 2],    # muscle 0 → joints 1,2
        [1],       # muscle 1 → joint 1
        [1, 2],    # muscle 2 → joints 1,2
        [3, 4],    # muscle 3 → joints 3,4
        [3],       # muscle 4 → joint 3
        [3, 4],    # muscle 5 → joints 3,4
        [5, 6],    # muscle 6 → joints 5,6
        [5],       # muscle 7 → joint 5
        [5, 6],    # muscle 8 → joints 5,6
        [7, 8],    # muscle 9 → joints 7,8
        [7],       # muscle 10 → joint 7
        [7, 8],    # muscle 11 → joints 7,8
    ] 
    elif num_joints == 10:
        muscle_joint_map = [
        [2, 3],    # muscle 0 → joints 2,3
        [2],       # muscle 1 → joint 2
        [2, 3],    # muscle 2 → joints 2,3
        [4, 5],    # muscle 3 → joints 4,5
        [4],       # muscle 4 → joint 4
        [4, 5],    # muscle 5 → joints 4,5
        [6, 7],    # muscle 6 → joints 6,7
        [6],       # muscle 7 → joint 6
        [6, 7],    # muscle 8 → joints 6,7
        [8, 9],    # muscle 9 → joints 8,9
        [8],       # muscle 10 → joint 8
        [8, 9],    # muscle 11 → joints 8,9
    ]
    else:
        raise ValueError("Unsupported number of joints. Created a new muscle_joint_map for this number of joints.")


    
    moment_matrix = np.zeros((num_muscles, num_joints))
    nonzero_values = actuator_moments[actuator_moments != 0]
    counter = 0

    for muscle_idx, joint_indices in enumerate(muscle_joint_map):
        for joint_idx in joint_indices:
            if counter < len(nonzero_values):
                moment_matrix[muscle_idx, joint_idx] = nonzero_values[counter]
                counter += 1
            else:
                break  # all values used

    return moment_matrix

def compute_and_save_activations(model_path, omega, dt, duration_in_seconds,activation_folder):
    """
    Computes muscle activations for a given model and saves them to a file.

    Parameters:
    model_path (str): Path to the MuJoCo model XML file.
    omega (float): Frequency for cyclical movements.
    dt (float): Time step for simulation.
    duration_in_seconds (int): Duration of the simulation in seconds.

    Returns:
    str: Path to the saved activation file.
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    RB, RF, LB, LF = dsk.create_cyclical_movements_fcn(omega, attempt_length=duration_in_seconds, dt=dt)

    # Get joint positions, velocities, and accelerations
    all_qpos = np.concatenate((RB[:, 0:2], RF[:, 0:2], LB[:, 0:2], LF[:, 0:2]), axis=1)
    all_qvel = np.concatenate((RB[:, 2:4], RF[:, 2:4], LB[:, 2:4], LF[:, 2:4]), axis=1)
    all_qacc = np.concatenate((RB[:, 4:6], RF[:, 4:6], LB[:, 4:6], LF[:, 4:6]), axis=1)

    # Initialize variables
    n_steps = all_qpos.shape[0]
    n_joints = len(model.dof_jntid)
    n_muscles = model.nu

    # joint_torques = np.zeros((n_steps, n_joints))
    F0 = np.ones(n_muscles) * 1000

    muscel_forces = np.zeros((n_steps, n_muscles))
    activations = np.zeros((n_steps, n_muscles))

    # Compute activations
    for t in range(n_steps):
        data.qpos[-8:] = all_qpos[t, :]
        data.qvel[-8:] = all_qvel[t, :]
        data.qacc[-8:] = all_qacc[t, :]

        mujoco.mj_inverse(model, data)

        joint_torque = data.qfrc_inverse
        actuator_moment_values = data.actuator_moment

        moment_matrix = fill_actuator_moments(n_muscles, n_joints, actuator_moment_values)
        # print(f"Moment matrix at time step {t}:\n{moment_matrix}")
        actuator_moments_transpose = moment_matrix.T

        L_t = data.actuator_length
        V_t = data.actuator_velocity

        FL_t = np.array([flv.compute_FL(L) for L in L_t])
        FP_t = np.array([flv.compute_FP(L) for L in L_t])
        FV_t = np.array([flv.compute_FV(V) for V in V_t])

        F_t = solve_tendon_forces(actuator_moments_transpose, joint_torque)
        # a_t = ca.invert_flv(F_t, FL_t, FV_t, FP_t, F0)

        a_t = F_t / F0

        muscel_forces[t, :] = F_t
        activations[t, :] = np.clip(a_t, 0, 1)

    # Save activations to a file
    activation_file = f"activation_{omega}.txt"

    os.makedirs(activation_folder, exist_ok=True)
    activation_file = os.path.join(activation_folder, f"activation_{omega}.txt")
    np.savetxt(activation_file, activations)

    return activation_file, all_qpos, all_qvel, all_qacc



