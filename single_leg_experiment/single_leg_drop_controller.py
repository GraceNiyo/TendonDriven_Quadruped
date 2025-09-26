# Author : Grace Niyo
# Date : September 25,2025

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import compute_ground_reaction_force as compute_grf
import compute_model_com_velocity as compute_com_vel
from spindle_model import gamma_driven_spindle_model_

# Load the MuJoCo model from an XML file
path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'

save_data = True
base_data_dir = '../all_data/single_leg_experiment/Simulation_09_25_2025/025kg_Data/'
if save_data:
    os.makedirs(base_data_dir, exist_ok=True)

# Load model 
try:
    model = mujoco.MjModel.from_xml_path(path_to_model)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading MuJoCo model from {path_to_model}: {e}")
    exit()

foot_geom_id = model.geom("rbfoot").id
if foot_geom_id == -1:
    print("Error: 'rbfoot' geom not found in model.")
    exit()

floor_geom_id = model.geom("floor").id
if floor_geom_id == -1:
    print("Error: 'floor' geom not found in model.")
    exit()

torso_id = model.body("torso").id
if torso_id == -1:
    print("Error: 'torso' body not found in model.")
    exit()

# Sim parameters
sim_duration = 5.0       # seconds
init_hold_time = 2.0     #  2 seconds before release
drop_height = 0.1    # raise above ground
muscle_activation = [0.2,0.14,0.14]  


# System type: "no_collateral", "with_collateral", or "beta"
system_type = "with_collateral" # Options: "no_collateral", "with_collateral", "beta"
spindle_gain = 1   # gain range 1 to 10

# Data logs
drop_data = {
    'joint_position': [],
    'joint_velocity': [],
    'com_position': [],
    'com_velocity': [],
    'ground_contact_force': [],
    'muscle_activation':[],
    'muscle_length': [],
    'muscle_velocity': [],
    'muscle_force': [],
    'Ia_feedback': [],
    'II_feedback': [],
}

II = np.zeros(model.nu)  
Ia = np.zeros(model.nu) 


with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    # Set the camera to fixed view if available
    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0

    while viewer.is_running() and time.time() - start < sim_duration:
        step_start = time.time()
        t = time.time() - start

        if t < init_hold_time:

            rootx_joint_id = model.joint("rootx").id
            rootz_joint_id = model.joint("rootz").id
            rbthigh_joint_id = model.joint("rbthigh").id
            rbshin_joint_id = model.joint("rbshin").id

            data.qpos[rootx_joint_id] = 0.0  
            data.qpos[rootz_joint_id] = -0.0146945 + drop_height
            data.qpos[rbthigh_joint_id] = 0.179306
            data.qpos[rbshin_joint_id] = 0.178366
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[:] = muscle_activation
            mujoco.mj_forward(model, data) 

        else:
            if system_type == "with_collateral":
                alpha_drive = muscle_activation + Ia + II
                data.ctrl[:] = alpha_drive
                for m in range(model.nu):
                    current_actuator_length = data.actuator_length[m]
                    current_actuator_velocity = data.actuator_velocity[m]
                    muscle_tendon_lengthrange = model.actuator_lengthrange[m].tolist()  
                            
                    Ia_feedback, II_feedback = gamma_driven_spindle_model_(
                                    actuator_length=current_actuator_length,
                                    actuator_velocity=current_actuator_velocity,
                                    actuator_lengthrange=muscle_tendon_lengthrange,
                                    gamma_dynamic= muscle_activation[m] * alpha_drive[m],
                                    gamma_static=  muscle_activation[m] * alpha_drive[m]
                                )
                    Ia[m] = Ia_feedback
                    II[m] = II_feedback
    
            elif system_type == "no_collateral":
                alpha_drive = muscle_activation + Ia + II
                data.ctrl[:] = alpha_drive
                for m in range(model.nu):
                    current_actuator_length = data.actuator_length[m]
                    current_actuator_velocity = data.actuator_velocity[m]
                    muscle_tendon_lengthrange = model.actuator_lengthrange[m].tolist()  
                            
                    Ia_feedback, II_feedback = gamma_driven_spindle_model_(
                                    actuator_length=current_actuator_length,
                                    actuator_velocity=current_actuator_velocity,
                                    actuator_lengthrange=muscle_tendon_lengthrange,
                                    gamma_dynamic= muscle_activation[m] ,
                                    gamma_static=  muscle_activation[m]
                                )
                    Ia[m] = Ia_feedback
                    II[m] = II_feedback

            elif system_type == "beta":
                beta_drive = (muscle_activation * spindle_gain) + Ia + II
                data.ctrl[:] = beta_drive
                for m in range(model.nu):
                    current_actuator_length = data.actuator_length[m]
                    current_actuator_velocity = data.actuator_velocity[m]
                    muscle_tendon_lengthrange = model.actuator_lengthrange[m].tolist()  
                            
                    Ia_feedback, II_feedback = gamma_driven_spindle_model_(
                                    actuator_length=current_actuator_length,
                                    actuator_velocity=current_actuator_velocity,
                                    actuator_lengthrange=muscle_tendon_lengthrange,
                                    beta_drive= beta_drive[m]
        
                                )
                    Ia[m] = Ia_feedback
                    II[m] = II_feedback
            else:
                print("Invalid system type specified. Must be 'no_collateral', 'with_collateral', or 'beta'.")
                exit()

            mujoco.mj_step(model, data)

        viewer.sync()  

        contact_force = compute_grf.get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id)
        com_vel = compute_com_vel.compute_model_com_velocity(model, data)

        drop_data['joint_position'].append(data.qpos.copy())
        drop_data['joint_velocity'].append(data.qvel.copy())
        drop_data['com_velocity'].append(com_vel)
        drop_data['com_position'].append(data.subtree_com[torso_id].copy())
        drop_data['ground_contact_force'].append(contact_force.copy())
        drop_data['muscle_activation'].append(data.ctrl.copy())
        drop_data['muscle_length'].append(data.actuator_length.copy())
        drop_data['muscle_velocity'].append(data.actuator_velocity.copy())
        drop_data['muscle_force'].append(data.actuator_force.copy())
        drop_data['Ia_feedback'].append(Ia.copy())
        drop_data['II_feedback'].append(II.copy())


        # Real-time pacing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Save data
if save_data:
        if system_type == "no_collateral":
            base_filename = f"no_collateral_{abs(drop_height):.3f}_gain_{spindle_gain:.1f}"
        elif system_type == "with_collateral":
            base_filename = f"with_collateral_{abs(drop_height):.3f}_gain_{spindle_gain:.1f}"
        elif system_type == "beta":
            base_filename = f"beta_{abs(drop_height):.3f}_gain_{spindle_gain:.1f}"
        else:
            print("Invalid system type specified. Must be 'no_collateral', 'with_collateral', or 'beta'.")
            exit()

        for data_key, data_list in drop_data.items():

            if len(data_list) == 0:
                print(f"Warning: {data_key} list is empty, skipping...")
                continue

            if isinstance(data_list[0], (float, int, np.ndarray)) and np.ndim(data_list[0]) == 0:
                data_to_save = np.array(data_list).reshape(-1, 1) 
            else:
                data_to_save = np.array(data_list)

            file_path = os.path.join(base_data_dir, f"{base_filename}_{data_key}.txt")
            np.savetxt(file_path, data_to_save, fmt='%.8f', delimiter='\t') 
