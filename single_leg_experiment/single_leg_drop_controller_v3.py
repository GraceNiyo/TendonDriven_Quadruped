import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from lxml import etree

import compute_ground_reaction_force as compute_grf
import compute_model_com_velocity as compute_com_vel
import allometric_scaler
from spindle_model import gamma_driven_spindle_model_


# ============= Function to modify xml mass and geometry ============= #
def modify_and_save_model(xml_path, total_mass, output_dir):
    """
    Modify the masses and geometry dimensions of geoms in the XML and save a new file.
    Uses allometric scaling based on total mass to update geometry dimensions.
    
    Mass distribution (based on biological proportions):
    - Torso: 50% of total mass
    - Upper segment (RB_HIP + rbthigh): 25% of total mass (12.5% each)
    - Lower segment (RB_KNEE + rbshin): 25% of total mass (12.5% each)
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    print(f"Total mass: {total_mass} kg, using for allometric scaling")
    
    # Calculate mass distribution based on biological proportions
    torso_mass = total_mass * 0.5  # 50%
    segment_mass = total_mass * 0.125  # 12.5% each for 4 segments
    
    mass_dict = {
        "torso": torso_mass,
        "RB_HIP": segment_mass, 
        "rbthigh": segment_mass,
        "RB_KNEE": segment_mass,
        "rbshin": segment_mass
    }
    
    print(f"Mass distribution - Torso: {torso_mass:.5f}, Segments: {segment_mass:.5f} each")
    
    # Get scaled dimensions from allometric scaler
    # Output: [L, new_torso_width, new_geom_thigh_0, new_geom_thigh_1, new_geom_shin_0, new_geom_shin_1]
    scaled_params = allometric_scaler.allometric_scaler(total_mass)
    L = scaled_params[0]  # scaled size 
    torso_width = scaled_params[1]  # New torso size[0] 
    thigh_radius = scaled_params[2]  # New thigh capsule radius
    thigh_half_length = scaled_params[3]  # New thigh capsule half-length
    shin_radius = scaled_params[4]  # New shin capsule radius  
    shin_half_length = scaled_params[5]  # New shin capsule half-length
    
    print(f"Allometric scaling factor: {L:.4f}")
    print(f"Scaled dimensions - Torso width: {torso_width:.5f}, Thigh: {thigh_radius:.5f}/{thigh_half_length:.5f}, Shin: {shin_radius:.5f}/{shin_half_length:.5f}")

    # Helper function to scale space-separated numeric values
    def scale_numeric_attribute(attr_value, scale_factor):
        """Scale space-separated numeric values by scale_factor"""
        if attr_value:
            values = attr_value.split()
            scaled_values = [str(float(val) * scale_factor) for val in values]
            return " ".join(scaled_values)
        return attr_value

    # Update mass and comprehensively scale ALL geometry
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        
        # Update mass if specified in mass_dict
        if name in mass_dict:
            geom.set("mass", str(mass_dict[name]))
            print(f"Updated mass of {name} to {mass_dict[name]}")
        
        # Scale ALL geom sizes by L factor (comprehensive scaling)
        size_attr = geom.get("size")
        if size_attr:
            if name == "torso":
                # Use precise allometric scaling for torso
                current_size = size_attr.split()
                new_size = f"{torso_width} {float(current_size[1]) * L} {float(current_size[2]) * L}"
                geom.set("size", new_size)
                print(f"Updated torso size to: {new_size}")
            elif name == "rbthigh":
                # Use precise allometric scaling for thigh  
                new_size = f"{thigh_radius} {thigh_half_length}"
                geom.set("size", new_size)
                print(f"Updated thigh size to: {new_size}")
            elif name == "rbshin":
                # Use precise allometric scaling for shin
                new_size = f"{shin_radius} {shin_half_length}"
                geom.set("size", new_size)
                print(f"Updated shin size to: {new_size}")
            else:
                # Scale all other geoms (RB_HIP, RB_KNEE, rbfoot, etc.) by L factor
                scaled_size = scale_numeric_attribute(size_attr, L)
                geom.set("size", scaled_size)
                print(f"Updated {name} size to: {scaled_size}")
        
        # Scale geom positions by L factor 
        pos_attr = geom.get("pos")
        if pos_attr:
            scaled_pos = scale_numeric_attribute(pos_attr, L)
            geom.set("pos", scaled_pos)

    # Scale ALL site sizes and positions by L factor
    for site in root.findall(".//site"):
        name = site.get("name") or "unnamed_site"
        
        # Scale site sizes
        size_attr = site.get("size") 
        if size_attr:
            scaled_size = scale_numeric_attribute(size_attr, L)
            site.set("size", scaled_size)
            print(f"Updated site {name} size to: {scaled_size}")
        
        # Scale site positions
        pos_attr = site.get("pos")
        if pos_attr:
            scaled_pos = scale_numeric_attribute(pos_attr, L)
            site.set("pos", scaled_pos)

    # Scale ALL body positions by L factor
    for body in root.findall(".//body"):
        pos_attr = body.get("pos")
        if pos_attr:
            scaled_pos = scale_numeric_attribute(pos_attr, L)
            body.set("pos", scaled_pos)
    
    # Scale joint positions by L factor
    for joint in root.findall(".//joint"):
        pos_attr = joint.get("pos") 
        if pos_attr:
            scaled_pos = scale_numeric_attribute(pos_attr, L)
            joint.set("pos", scaled_pos)
    
    print(f"Applied comprehensive geometric scaling factor L={L:.4f} (muscle forces kept at default values)")

    # Create unique filename based on total model mass and size L
    model_mass_tag = f"mass{total_mass:.3f}" 
    size_tag = f"L{L:.3f}"
    new_xml_path = os.path.join(output_dir, f"single_leg_{model_mass_tag}_{size_tag}.xml")

    os.makedirs(output_dir, exist_ok=True)
    tree.write(new_xml_path, pretty_print=True)
    print(f"Saved modified XML to {new_xml_path}")
    return new_xml_path


# ============= Control loop ============= #

def run_simulation_batch(xml_path, drop_height=0.1, relative_drop_height=None, sim_duration=5.0, init_hold_time=2.0,
                        muscle_activation=[0.2, 0.14, 0.14], system_types=["beta","alpha_gamma_co_activation_with_collateral","alpha_gamma_co_activation_no_collateral","independent_with_collateral","independent_no_collateral","feedforward"],
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

                    elif system_type == "feedforward":
                        data.ctrl[:] = muscle_activation

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
                    
                    # Use relative drop height for filename (better for cross-model comparison)
                    drop_height_for_filename = relative_drop_height if relative_drop_height is not None else drop_height
                    
                    # Create filename with spindle gain for independent systems
                    if "independent" in system_type:
                        filename = f"{system_type}_drop_{drop_height_for_filename:.2f}_gamma_drive_{gamma_drive:.1f}"
                    else:
                        filename = f"{system_type}_drop_{drop_height_for_filename:.2f}"
                    
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
    xml_template = "../Working_Folder/single_leg_experiment/unit_single_leg.xml"
    # Total mass scenarios (mass distribution calculated automatically)
    mass_scenarios = [0.25] #, 0.5, 0.75, 1,1.25, 1.5, 1.75, 2.0]
    
    # Relative drop heights as fractions of body length (for L=1.0 reference model)
    relative_drop_heights = np.arange(0.00, 0.55, 0.05)  # 0.05, 0.1, 0.15, ... 0.5 body lengths
    
    gamma_drives = np.arange(1.0, 0, -0.1)  
    
    for total_mass in mass_scenarios:
        # Calculate L factor for this mass to scale drop heights appropriately
        L = total_mass ** (1/3)
        
        # Scale drop heights by model size (biomechanically equivalent drops)
        absolute_drop_heights = relative_drop_heights * L
        
        folder_name = f"{int(total_mass*1000):03d}mg_Data"  
        output_dir = os.path.join("../all_data/single_leg_experiment/leg_drop_data_02_25_2026_test/checklast", folder_name)
        
        print(f"\n=== Processing mass scenario: {total_mass} kg (L={L:.3f}) ===")
        print(f"Relative drop heights: {relative_drop_heights}")
        print(f"Absolute drop heights: {absolute_drop_heights}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XML with automatic mass distribution
        new_xml = modify_and_save_model(xml_template, total_mass, output_dir)
        
        for i, drop_height in enumerate(absolute_drop_heights):
            relative_height = relative_drop_heights[i]
            print(f"\n--- Drop Height: {drop_height:.3f} m ({relative_height:.2f} body lengths) ---")
            

            non_independent_systems = ["beta"]#","alpha_gamma_co_activation_with_collateral", "alpha_gamma_co_activation_no_collateral","feedforward"]
            for system_type in non_independent_systems:
                run_simulation_batch(new_xml, drop_height=drop_height, relative_drop_height=relative_height, 
                                   base_data_dir=output_dir, system_types=[system_type], save_data=True, sim_duration=5.0)
            
            # independent_systems = ["independent_with_collateral"] #, "independent_no_collateral"]
            # independent_systems = ["independent_with_collateral"]
            
            # for system_type in independent_systems:
            #     for gamma_drive in gamma_drives:
            #         print(f"  Spindle Gain: {gamma_drive}")
            #         run_simulation_batch(new_xml, drop_height=drop_height, relative_drop_height=relative_height,
            #                             base_data_dir=output_dir, system_types=[system_type], 
            #                             gamma_drive=gamma_drive,save_data=True, sim_duration=5.0)