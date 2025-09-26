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
def run_simulation(xml_path, drop_height=0.1, sim_duration=5.0, init_hold_time=2.0,
                   muscle_activation=[0.2, 0.14, 0.14], system_type="with_collateral",
                   spindle_gain=1, save_data=True, base_data_dir="./sim_data"):

    os.makedirs(base_data_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    foot_geom_id = model.geom("rbfoot").id
    floor_geom_id = model.geom("floor").id
    torso_id = model.body("torso").id

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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        if model.ncam > 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = 0

        while viewer.is_running() and time.time() - start < sim_duration:
            step_start = time.time()
            t = time.time() - start

            if t < init_hold_time:
                # hold in position before release
                data.qpos[model.joint("rootx").id] = 0.0
                data.qpos[model.joint("rootz").id] = -0.0146945 + drop_height
                data.qpos[model.joint("rbthigh").id] = 0.179306
                data.qpos[model.joint("rbshin").id] = 0.178366
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
                data.ctrl[:] = muscle_activation
                mujoco.mj_forward(model, data)
            else:
                if system_type == "with_collateral":
                    alpha_drive = muscle_activation + Ia + II
                    data.ctrl[:] = alpha_drive
                    for m in range(model.nu):
                        Ia[m], II[m] = gamma_driven_spindle_model_(
                            actuator_length=data.actuator_length[m],
                            actuator_velocity=data.actuator_velocity[m],
                            actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                            gamma_dynamic=muscle_activation[m] * alpha_drive[m],
                            gamma_static=muscle_activation[m] * alpha_drive[m]
                        )

                elif system_type == "no_collateral":
                    alpha_drive = muscle_activation + Ia + II
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
                    beta_drive = (muscle_activation * spindle_gain) + Ia + II
                    data.ctrl[:] = beta_drive
                    for m in range(model.nu):
                        Ia[m], II[m] = gamma_driven_spindle_model_(
                            actuator_length=data.actuator_length[m],
                            actuator_velocity=data.actuator_velocity[m],
                            actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                            beta_drive=beta_drive[m]
                        )
                else:
                    raise ValueError("Invalid system_type")

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

            # real-time pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Save data
    if save_data:
        filename = f"{system_type}_drop_{drop_height:.2f}"
        for data_key, data_list in drop_data.items():
            if len(data_list) > 0:
                np.savetxt(os.path.join(base_data_dir, f"{filename}_{data_key}.txt"),
                           np.array(data_list), fmt="%.8f", delimiter="\t")


# ========================== #
if __name__ == "__main__":
    xml_template = "../Working_Folder/single_leg_experiment/single_leg.xml"
    # Example sweep of different torso + thigh masses
    mass_scenarios = [
        {"torso": 0.125, "RB_HIP": 0.03125, "rbthigh": 0.03125, "RB_KNEE": 0.03125, "rbshin": 0.03125},
        {"torso": 2.0, "RB_HIP": 0.5, "rbthigh": 1.0, "RB_KNEE": 0.2, "rbshin": 0.8},
    ]

    for mass_dict in mass_scenarios:
        # compute total mass
        total_mass = sum(mass_dict.values())
        folder_name = f"{int(total_mass*1000):03d}kg_Data"   # e.g. 0.25 -> "025kg_Data"
        output_dir = os.path.join("../all_data/single_leg_experiment/Simulation_09_25_2025", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # save XML there
        new_xml = modify_and_save_model(xml_template, mass_dict, output_dir)

        # run sim and save data in same folder
        run_simulation(new_xml, drop_height=0.1, base_data_dir=output_dir, system_type="no_collateral")

