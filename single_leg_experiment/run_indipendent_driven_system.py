import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import spindle_model as spindle_model
from spindle_model import gamma_driven_spindle_model_

from generate_smooth_force_profile import generate_smooth_force_profile
import compute_global_com
import compute_ground_reaction_force as compute_grf



# --- Configuration Parameters ---
path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'
base_data_dir = '../all_data/single_leg_experiment/Simulation_09_15_2025/1kg_Data/'
os.makedirs(base_data_dir, exist_ok=True)

# Muscle activation levels (M0, M1, M2)
M0 = 0.3
M1 = 0.1
M2 = 0.32
muscle_activations = np.array([M0, M1, M2])
# gamma_static = muscle_activations
# gamma_dynamic = muscle_activations
# collateral_input = muscle_activations
gamma_range = np.arange(0, 1.1, 0.1)


all_simulation_results = []

# durations for each new phase
phase_durations = [
    3.0, # Phase 1: Only muscle activation, no external force (0 to 2s)
    # 3.0, # Phase 2: Muscle activation + Ia feedback, no external force (2 to 5s)
    1.0, # Phase 3: Muscle activation + Ia feedback + external force (the pulse) (5 to 6s)
    3.0, # Phase 4: Muscle activation + Ia feedback, no external force (6 to 9s)
    # 2.0  # Phase 5: Only muscle activation, no external force (9 to 11s)
]

# Calculate cumulative time thresholds for phase transitions
cumulative_times = np.cumsum(phase_durations)
total_sim_duration = cumulative_times[-1]


# Apply force from 0 N down to -10 N (in z-direction)
force_vector = np.arange(-3,-3.1,-0.1)


# --- Model Loading and Initialization ---
try:
    model = mujoco.MjModel.from_xml_path(path_to_model)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading MuJoCo model from {path_to_model}: {e}")
    exit()

# Get IDs from the model
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
if body_id == -1:
    print("Error: 'torso' body not found in model.")
    exit()
force_location_on_model = data.xpos[body_id]

torso_id = model.body("torso").id
if torso_id == -1:
    print("Error: 'torso' body not found in model.")
    exit()

# touch_sensor_id = model.sensor("rbfoot_touch_sensor").id
# if touch_sensor_id == -1:
#     print("Error: 'rbfoot_touch_sensor' not found in model.")
#     exit()

foot_geom_id = model.geom("rbfoot").id
if foot_geom_id == -1:
    print("Error: 'rbfoot' geom not found in model.")
    exit()

floor_geom_id = model.geom("floor").id
if floor_geom_id == -1:
    print("Error: 'floor' geom not found in model.")
    exit()



# --- Main Simulation Loop ---

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui = True) as viewer:

    # Set the camera to fixed view if available
    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0

    # Loop through each force magnitude
    for f in force_vector:
        for gamma_dyn in gamma_range:
            for gamma_stat in gamma_range:
                print(f"Running simulation with force: {f} N, gamma_dynamic: {gamma_dyn}, gamma_static: {gamma_stat}")

                torque = np.array([0, 0, 0]) # No torque applied
                applied_force = np.zeros(3)  # Initialize applied force

                gamma_dynamic_vec = np.full(model.nu, gamma_dyn)
                gamma_static_vec = np.full(model.nu, gamma_stat)


                force_values = generate_smooth_force_profile(
                    duration=phase_durations[1],  # Duration of the force pulse
                    timestep=model.opt.timestep,
                    rise_time=0.1,  # 100 ms time to reach peak force
                    decay_time=0.1,  # 100 ms time to decay to zero force
                    peak_force=f  # Peak force in Newtons (negative for downward force)
                )


                # Reset the model and data to initial state for each new simulation run
                mujoco.mj_resetData(model, data)
                if viewer.is_running():
                    viewer.sync()


                simulation_start_data_time = data.time

                # Initialize data storage for the current simulation run
                run_data = {
                    'force': f,
                    'joint_position': [],
                    'joint_velocity': [],
                    'com_position': [],
                    'com_velocity': [],
                    'ground_contact_force': [],
                    'time': [],
                    'external_applied_force': [],
                    'muscle_activation':[],
                    'muscle_length': [],
                    'muscle_velocity': [],
                    'muscle_force': [],
                    'Ia_feedback': [],
                    'II_feedback': [],
                }

                # Initialize lists to store Ia and II signals 
                II_a = np.zeros(model.nu)  # Initialize II feedback for each actuator
                Ia_a = np.zeros(model.nu)  
                
        

                total_steps = int(total_sim_duration / model.opt.timestep) + 1 
                # Run the simulation for the calculated total number of steps
                for step_count in range(total_steps):
                    step_start_real_time = time.time() # will be removed later, used for debugging now

                    # Check if the viewer is still running. If closed, terminate the current run.
                    if viewer.is_running():
                        viewer.sync() # Update the viewer display
                    else:
                        print("Viewer closed. Terminating current simulation run and proceeding to next or finishing.")
                        break # Break out of the current simulation run loop


                    current_sim_time_relative = data.time - simulation_start_data_time

                    # --- Simulation Phase Logic ---

                    if current_sim_time_relative < cumulative_times[0]: 
                        # Only muscle activation, no external force
                        data.ctrl[:] = muscle_activations
                        data.qfrc_applied[:] = 0.0 
       
                    elif current_sim_time_relative < cumulative_times[1]: 
                        # Muscle activation + Ia feedback + external force pulse

                        for m in range(model.nu): # Calculate Ia feedback using the spindle model
                            current_actuator_length = data.actuator_length[m]
                            current_actuator_velocity = data.actuator_velocity[m]
                            muscle_tendon_lengthrange = model.actuator_lengthrange[m].tolist()  

                        
                            Ia_feedback, II_feedback = gamma_driven_spindle_model_(
                                actuator_length=current_actuator_length,
                                actuator_velocity=current_actuator_velocity,
                                actuator_lengthrange=muscle_tendon_lengthrange,
                                gamma_dynamic= gamma_dynamic_vec[m],
                                gamma_static= gamma_static_vec  [m]
                            )
                            Ia_a[m] = Ia_feedback
                            II_a[m] = II_feedback

                        # Apply the external force pulse
                        force_index = int((current_sim_time_relative - cumulative_times[0]) / model.opt.timestep)
                        if force_index >= len(force_values):
                            force_index = len(force_values) - 1

                        applied_force = np.array([0, 0, force_values[force_index]]) 
                        # data.qfrc_applied[:] = applied_force
                
                       
                        data.ctrl[:] = np.clip(muscle_activations + Ia_a + II_a, 0, 1)
                        mujoco.mj_applyFT(model, data, applied_force, torque, force_location_on_model, body_id, data.qfrc_applied)

                    else:
                        # current_sim_time_relative < cumulative_times[2]: 
                        # Muscle activation + Ia feedback, no external force
                        for m in range(model.nu): # Calculate Ia feedback using the spindle model
                            
                            current_actuator_length = data.actuator_length[m]
                            current_actuator_velocity = data.actuator_velocity[m]
                            muscle_tendon_lengthrange = model.actuator_lengthrange[m].tolist()  # as a list


                            Ia_feedback, II_feedback = gamma_driven_spindle_model_(
                                actuator_length=current_actuator_length,
                                actuator_velocity=current_actuator_velocity,
                                actuator_lengthrange=muscle_tendon_lengthrange,
                                gamma_dynamic= gamma_dynamic_vec[m],
                                gamma_static= gamma_static_vec[m]
                            )
                            Ia_a[m] = Ia_feedback
                            II_a[m] = II_feedback

                        data.ctrl[:] =   np.clip(muscle_activations + Ia_a + II_a, 0, 1)
                        data.qfrc_applied[:] = 0.0 # No external force
                        applied_force = np.zeros(3)  # Reset applied force to zero

                    # else: # Only muscle activation, no external force
                    #     data.ctrl[:] = muscle_activations
                    #     data.qfrc_applied[:] = 0.0

                    # --- Update the model and data ---

                    mujoco.mj_step(model, data)


                    # --- Collect Data for the Current Step ---
                    contact_force = compute_grf.get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id)

                    run_data['external_applied_force'].append(applied_force)
                    run_data['joint_position'].append(data.qpos.copy())
                    run_data['joint_velocity'].append(data.qvel.copy())
                    run_data['com_velocity'].append(data.cvel[torso_id].copy())
                    run_data['com_position'].append(data.subtree_com[torso_id].copy())
                    run_data['ground_contact_force'].append(contact_force.copy())
                    run_data['time'].append(data.time)
                    run_data['muscle_activation'].append(data.ctrl.copy())
                    run_data['muscle_length'].append(data.actuator_length.copy())
                    run_data['muscle_velocity'].append(data.actuator_velocity.copy())
                    run_data['muscle_force'].append(data.actuator_force.copy())
                    run_data['Ia_feedback'].append(Ia_a.copy())
                    run_data['II_feedback'].append(II_a.copy())


                    # syncronize to model real_time simulation
                    # time_elapsed_real = time.time() - step_start_real_time
                    # time_to_sleep = model.opt.timestep - time_elapsed_real
                    # if time_to_sleep > 0:
                    #     time.sleep(time_to_sleep)

                # --- End of Simulation Run ---

                base_filename = f"independent_drive_force_{abs(f):.1f}_gamma_dynamic_{gamma_dyn:.2f}_gamma_static_{gamma_stat:.2f}"
                    # save each data array to a separate .txt file
                for data_key, data_list in run_data.items():
                    if data_key == 'force':
                        continue

                    if isinstance(data_list[0], (float, int, np.ndarray)) and np.ndim(data_list[0]) == 0:
                        data_to_save = np.array(data_list).reshape(-1, 1) 
                    else:
                        data_to_save = np.array(data_list)

                    file_path = os.path.join(base_data_dir, f"{base_filename}_{data_key}.txt")
                    np.savetxt(file_path, data_to_save, fmt='%.8f', delimiter='\t') 
            # print(f"Saved {data_key} to {file_path}")
        
            all_simulation_results.append(run_data)

print("\nAll simulations completed.")
print(f"Total {len(all_simulation_results)} simulation runs performed.")
print(f"All data saved to: {base_data_dir}")