import numpy as np
import mujoco 
import mujoco.viewer
import time
import os
import compute_model_com_velocity as compute_com_vel
from spindle_model import gamma_driven_spindle_model_


path_to_model = "quadruped_experiment/quadruped_ws_onground.xml"
activation_folder = "../quadruped_activation_data_10_09_2025"


# Run the simulation with one of the activation files
model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)
torso_id = model.body("body").id

system_types=["feedforward","with_collateral", "no_collateral", "beta"]

# Set muscle activations 
omega = 0.7 # 
activation_file = f"{activation_folder}/activation_{omega}.txt"
muscle_activation_array = (np.loadtxt(activation_file)) 
activation_scaler = 5.0
muscle_activation_array *= activation_scaler

# Spindle gain value
spindle_gain = 0.1

# Simulation parameters
n_steps = muscle_activation_array.shape[0]
timestep = model.opt.timestep
delay_duration = 2.0 # seconds
delay_steps = int(delay_duration / timestep)
duration  = muscle_activation_array.shape[0]

#  director to save data
base_data_dir = "../all_data/quadruped_experiment/10_09_2025"
save_data=False

# ============= Control loop ============= #
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:
    
    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0


    # Loop through each system type
    for system_type in system_types:
        if not viewer.is_running():
            print("Viewer closed. Terminating simulation batch.")
            break
            
        print(f"\n--- Running {system_type} ---")
        
        # Reset model and data for each system type
        mujoco.mj_resetData(model, data)
        if viewer.is_running():
            viewer.sync()

        # Initialize data storage for current system type
        drop_data = {
            'joint_position': [],
            'joint_velocity': [],
            'com_position': [],
            'com_velocity': [],
            'ground_contact_force': [],
            'muscle_activation': [],
            'muscle_length': [],
            'muscle_velocity': [],
            'muscle_force': [],
            'Ia_feedback': [],
            'II_feedback': [],
        }

        II = np.zeros(model.nu)
        Ia = np.zeros(model.nu)
        idx = 0

        # Run simulation for current system type
        while viewer.is_running() and idx - delay_steps < duration:
            step_start = time.time()

            if idx < delay_steps:
                data.ctrl[:] = 0.0
                data.qpos[:] = [-0.0868229, -0.309702, -0.0598717, -0.434487, -0.0598717, -0.434487, -0.0598717, -0.434487, -0.0598717, -0.434487]
                data.qvel[:] = 0.0
            else:
                muscle_activation = muscle_activation_array[idx - delay_steps,:]
                if system_type == "feedforward":
                    data.ctrl[:] = muscle_activation

                elif system_type == "with_collateral":
                    alpha_drive = muscle_activation + Ia + II
                    data.ctrl[:] = np.clip(alpha_drive, 0, 1)
                    for m in range(model.nu):
                        Ia[m], II[m] = gamma_driven_spindle_model_(
                            actuator_length=data.actuator_length[m],
                            actuator_velocity=data.actuator_velocity[m],
                            actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                            gamma_dynamic=muscle_activation[m] * alpha_drive[m],
                            gamma_static=muscle_activation[m] * alpha_drive[m]
                        )

                elif system_type == "no_collateral":
                    alpha_drive = np.clip(muscle_activation + Ia + II, 0, 1)
                    data.ctrl[:] = alpha_drive
                    for m in range(model.nu):
                        Ia[m], II[m] = gamma_driven_spindle_model_(
                            actuator_length=data.actuator_length[m],
                            actuator_velocity=data.actuator_velocity[m],
                            actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                            gamma_dynamic=muscle_activation[m],
                            gamma_static=muscle_activation[m]
                        )

                elif system_type == "beta":
                    beta_drive = np.clip(muscle_activation + Ia + II, 0, 1)
                    data.ctrl[:] = beta_drive
                    for m in range(model.nu):
                        Ia[m], II[m] = gamma_driven_spindle_model_(
                            actuator_length=data.actuator_length[m],
                            actuator_velocity=data.actuator_velocity[m],
                            actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                            gamma_dynamic=beta_drive[m],  
                            gamma_static=beta_drive[m]
                        )

                mujoco.mj_step(model, data)

            # Update viewer
            if viewer.is_running():
                viewer.sync()

            # Collect data
            com_vel = compute_com_vel.compute_model_com_velocity(model, data)

            drop_data['joint_position'].append(data.qpos.copy())
            drop_data['joint_velocity'].append(data.qvel.copy())
            drop_data['com_velocity'].append(com_vel)
            drop_data['com_position'].append(data.subtree_com[torso_id].copy())
            drop_data['ground_contact_force'].append(data.sensordata.copy())
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
            idx += 1

            if idx - delay_steps >= duration:
                print(f"Completed {system_type} simulation")
                break

        # Save data for current system type
        if save_data:
            print(f"Saving data for {system_type}...")  
            filename = f"{system_type}_omega_{omega}"
            
            for data_key, data_list in drop_data.items():
                if len(data_list) > 0:
                    file_path = os.path.join(base_data_dir, f"{filename}_{data_key}.txt")
                    print(f"Saving {data_key} to {file_path}")
                    np.savetxt(file_path, np.array(data_list), fmt="%.8f", delimiter="\t")
                    print(f"Saved {data_key} ({len(data_list)} rows)")
                else:
                    print(f"Skipping {data_key}: empty list")
