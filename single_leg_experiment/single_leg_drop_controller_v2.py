import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from lxml import etree

import compute_ground_reaction_force as compute_grf
import compute_model_com_velocity as compute_com_vel
from spindle_model import gamma_driven_spindle_model_


# ============= Function to modify xml mass ============= #
def modify_and_save_model(xml_path, mass_dict, output_dir):
    """
    Modify the masses of specific geoms in the XML and save a new file.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Update mass for all matching geoms
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name in mass_dict:
            geom.set("mass", str(mass_dict[name]))
            print(f"Updated mass of {name} to {mass_dict[name]}")

    # create unique filename based on masses
    mass_tag = "_".join(f"{k}{v}" for k, v in mass_dict.items())
    new_xml_path = os.path.join(output_dir, f"single_leg_{mass_tag}.xml")

    os.makedirs(output_dir, exist_ok=True)
    tree.write(new_xml_path, pretty_print=True)
    print(f"Saved modified XML to {new_xml_path}")
    return new_xml_path


# ============= Control loop ============= #

def run_simulation_batch(xml_path, drop_height=0.1, sim_duration=5.0, init_hold_time=2.0,
                        muscle_activation=[0.2, 0.14, 0.14], system_types=["beta","alpha_gamma_co_activation_with_collateral","alpha_gamma_co_activation_no_collateral","independent_with_collateral","independent_no_collateral"],
                        save_data=True, gamma_drive = 1.0, base_data_dir="./sim_data"):
    
    print(f"Running batch simulation for all system types")
    print(f"Base data dir: {base_data_dir}")
    print(f"Absolute path: {os.path.abspath(base_data_dir)}")
    
    os.makedirs(base_data_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    foot_geom_id = model.geom("rbfoot").id
    floor_geom_id = model.geom("floor").id
    torso_id = model.body("torso").id

    # Launch viewer once for all system types
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
                'time': [],
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
            start = time.time()

            # Run simulation for current system type
            while viewer.is_running() and time.time() - start < sim_duration:
                step_start = time.time()
                t = time.time() - start

                if t < init_hold_time:
                    # Hold in position before release
                    data.qpos[model.joint("rootx").id] = 0.0
                    data.qpos[model.joint("rootz").id] = -0.0146945 + drop_height
                    data.qpos[model.joint("rbthigh").id] = 0.179306
                    data.qpos[model.joint("rbshin").id] = 0.178366
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    data.ctrl[:] = muscle_activation
                    mujoco.mj_forward(model, data)
                else:
                    if system_type == "alpha_gamma_co_activation_with_collateral":
                        alpha_drive = np.clip(muscle_activation + Ia + II, 0, 1)
                        data.ctrl[:] = alpha_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=muscle_activation[m] * alpha_drive[m],
                                gamma_static=muscle_activation[m] * alpha_drive[m]
                            )

                    elif system_type == "alpha_gamma_co_activation_no_collateral":
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
                        beta_drive = np.clip(np.array(muscle_activation) + Ia + II, 0, 1)
                        data.ctrl[:] = beta_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=beta_drive[m],  
                                gamma_static=beta_drive[m]
                            )
                    elif system_type == "independent_with_collateral":
                        alpha_drive = np.clip(muscle_activation + Ia + II, 0, 1)    
                        data.ctrl[:] = alpha_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic= alpha_drive[m] * gamma_drive,
                                gamma_static= alpha_drive[m] * gamma_drive
                            )
                    elif system_type == "independent_no_collateral":
                        alpha_drive = np.clip(muscle_activation + Ia + II, 0, 1)
                        data.ctrl[:] = alpha_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=gamma_drive,
                                gamma_static=gamma_drive   
                            )

                    mujoco.mj_step(model, data)

                # Update viewer
                if viewer.is_running():
                    viewer.sync()

                # Collect data
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
                drop_data['time'].append(t)


                # Real-time pacing
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

            # Save data for current system type
            if save_data and len(drop_data['joint_position']) > 0:
                    print(f"Saving data for {system_type}...")  
                    
                    # Create filename with spindle gain for independent systems
                    if "independent" in system_type:
                        filename = f"{system_type}_drop_{drop_height:.2f}_gamma_drive_{gamma_drive:.1f}"
                    else:
                        filename = f"{system_type}_drop_{drop_height:.2f}"
                    
                    for data_key, data_list in drop_data.items():
                        if len(data_list) > 0:
                            file_path = os.path.join(base_data_dir, f"{filename}_{data_key}.txt")
                            print(f"Saving {data_key} to {file_path}")
                            np.savetxt(file_path, np.array(data_list), fmt="%.8f", delimiter="\t")
                            print(f"Saved {data_key} ({len(data_list)} rows)")
                        else:
                            print(f"Skipping {data_key}: empty list")


# ========================== #
if __name__ == "__main__":
    xml_template = "../Working_Folder/single_leg_experiment/single_leg.xml"
    mass_scenarios = [
        {"torso": 0.125, "RB_HIP": 0.03125, "rbthigh": 0.03125, "RB_KNEE": 0.03125, "rbshin": 0.03125},
        {"torso": 0.25, "RB_HIP": 0.0625, "rbthigh": 0.0625, "RB_KNEE": 0.0625, "rbshin": 0.0625},
        {"torso": 0.375, "RB_HIP": 0.09375, "rbthigh": 0.09375, "RB_KNEE": 0.09375, "rbshin": 0.09375},
        {"torso": 0.5, "RB_HIP": 0.125, "rbthigh": 0.125, "RB_KNEE": 0.125, "rbshin": 0.125}
    ]
    drop_heights = np.arange(0.00, 0.55, 0.05)  # m above the ground
    gamma_drives = np.arange(0.1, 1.1, 0.1)  
    
    for mass_dict in mass_scenarios:
        total_mass = sum(mass_dict.values())
        folder_name = f"{int(total_mass*1000):03d}mg_Data"  
        output_dir = os.path.join("../all_data/single_leg_experiment/leg_drop_data_10_18_2025/soft_floor", folder_name)
        
        # print(f"\n=== Processing mass scenario: {total_mass} kg ===")
        # print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XML
        new_xml = modify_and_save_model(xml_template, mass_dict, output_dir)
        
        for drop_height in drop_heights:
            print(f"\n--- Drop Height: {drop_height:.2f} m ---")
            

            non_independent_systems = ["beta", "alpha_gamma_co_activation_with_collateral"] #, "alpha_gamma_co_activation_no_collateral"]
            if non_independent_systems:
                run_simulation_batch(new_xml, drop_height=drop_height, base_data_dir=output_dir, 
                                   system_types=non_independent_systems, save_data=True)
            
            # independent_systems = ["independent_with_collateral", "independent_no_collateral"]
            # independent_systems = ["independent_with_collateral"]

            # for gamma_drive in gamma_drives:
            #     print(f"  Spindle Gain: {gamma_drive}")
            #     run_simulation_batch(new_xml, drop_height=drop_height, base_data_dir=output_dir,
            #                        system_types=independent_systems, gamma_drive=gamma_drive,save_data=True)