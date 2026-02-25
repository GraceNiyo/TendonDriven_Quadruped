import numpy as np
import mujoco
import mujoco.viewer
import time
import os

import compute_model_com_velocity as compute_com_vel
from spindle_model import gamma_driven_spindle_model_
import compute_ground_reaction_force as compute_grf


path_to_model = "quadruped_experiment/quadruped_ws_onground.xml"
activation_folder = "../quadruped_activation_data_10_09_2025"


def run_quadruped(
    path_to_model,
    activation_folder,
    omega=0.7,
    activation_scaler=5.0,
    delay_duration=2.0,
    after_sim_delay=2.0,
    base_data_dir="../all_data/quadruped_experiment/Very_soft_floor_Data_10cm_compliance_10_14_2025/check",
    save_data=False,
    gamma_drive=0.1,
    system_types=None,
):
    if system_types is None:
        system_types = [
            "feedforward",
            "beta",
            "alpha_gamma_co_activation_with_collateral",
            "alpha_gamma_co_activation_no_collateral",
            "independent_with_collateral",
            "independent_no_collateral",
        ]

    # Run the simulation with one of the activation files
    model = mujoco.MjModel.from_xml_path(path_to_model)
    data = mujoco.MjData(model)

    body_id = model.body("body").id
    if body_id == -1:
        print("Error: 'body' body not found in model.")
        return

    rbfoot_geom_id = model.geom("rbfoot").id
    if rbfoot_geom_id == -1:
        print("Error: 'rbfoot' geom not found in model.")
        return

    rffoot_geom_id = model.geom("rffoot").id
    if rffoot_geom_id == -1:
        print("Error: 'rffoot' geom not found in model.")
        return

    lbfoot_geom_id = model.geom("lbfoot").id
    if lbfoot_geom_id == -1:
        print("Error: 'lbfoot' geom not found in model.")
        return

    lffoot_geom_id = model.geom("lffoot").id
    if lffoot_geom_id == -1:
        print("Error: 'lffoot' geom not found in model.")
        return

    floor_geom_id = model.geom("floor").id
    if floor_geom_id == -1:
        print("Error: 'floor' geom not found in model.")
        return

    # Set muscle activations
    activation_file = f"{activation_folder}/activation_{omega}.txt"
    muscle_activation_array = np.loadtxt(activation_file)
    muscle_activation_array *= activation_scaler

    # Simulation parameters
    timestep = model.opt.timestep
    delay_steps = int(delay_duration / timestep)
    after_sim_delay_steps = int(after_sim_delay / timestep)
    duration = muscle_activation_array.shape[0]

    # directory to save data
    if save_data:
        os.makedirs(base_data_dir, exist_ok=True)

    # ============= Control loop ============= #
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:

        if model.ncam > 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = 1  # 0 world camera , 1: body track camera

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
                "joint_position": [],
                "joint_velocity": [],
                "com_position": [],
                "com_velocity": [],
                "ground_contact_force": [],
                "sensor_data": [],
                "muscle_activation": [],
                "muscle_length": [],
                "muscle_velocity": [],
                "muscle_force": [],
                "Ia_feedback": [],
                "II_feedback": [],
            }

            II = np.zeros(model.nu)
            Ia = np.zeros(model.nu)
            idx = 0
            # Run simulation for current system type
            while viewer.is_running() and idx - delay_steps - duration < after_sim_delay_steps:
                step_start = time.time()

                if idx < delay_steps:
                    data.ctrl[:] = 0.0
                    # data.qpos[:] = [
                    #     -0.102056,
                    #     -0.308312,
                    #     -0.0163205,
                    #     -0.0451933,
                    #     -0.487662,
                    #     -0.0804189,
                    #     -0.398441,
                    #     -0.0451933,
                    #     -0.487662,
                    #     -0.0804189,
                    #     -0.398441,
                    # ]
                    # data.qvel[:] = 0.0
                    mujoco.mj_step(model, data)
                elif idx - delay_steps < duration:
                    muscle_activation = muscle_activation_array[idx - delay_steps, :]

                    if system_type == "feedforward":
                        data.ctrl[:] = muscle_activation

                    elif system_type == "alpha_gamma_co_activation_with_collateral":
                        alpha_drive = muscle_activation + Ia + II
                        data.ctrl[:] = np.clip(alpha_drive, 0, 1)
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=muscle_activation[m] * alpha_drive[m],
                                gamma_static=muscle_activation[m] * alpha_drive[m],
                            )

                    elif system_type == "alpha_gamma_co_activation_no_collateral":
                        alpha_drive = np.clip(muscle_activation + II , 0, 1)
                        data.ctrl[:] = alpha_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=muscle_activation[m],
                                gamma_static=muscle_activation[m],
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
                                gamma_static=beta_drive[m],
                            )

                    elif system_type == "independent_with_collateral":
                        alpha_drive = np.clip(muscle_activation + Ia + II , 0, 1)
                        data.ctrl[:] = alpha_drive
                        for m in range(model.nu):
                            Ia[m], II[m] = gamma_driven_spindle_model_(
                                actuator_length=data.actuator_length[m],
                                actuator_velocity=data.actuator_velocity[m],
                                actuator_lengthrange=model.actuator_lengthrange[m].tolist(),
                                gamma_dynamic=alpha_drive[m] * gamma_drive,
                                gamma_static=alpha_drive[m] * gamma_drive,
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
                                gamma_static=gamma_drive,
                            )

                    mujoco.mj_step(model, data)
                else:
                    # After simulation delay phase - remove activations
                    data.ctrl[:] = 0.0
                    mujoco.mj_step(model, data)

                # Update viewer
                if viewer.is_running():
                    viewer.sync()

                # Collect data
                com_vel = compute_com_vel.compute_model_com_velocity(model, data)

                grf = np.concatenate(
                    [
                        compute_grf.get_ground_reaction_force(model, data, rbfoot_geom_id, floor_geom_id),
                        compute_grf.get_ground_reaction_force(model, data, rffoot_geom_id, floor_geom_id),
                        compute_grf.get_ground_reaction_force(model, data, lbfoot_geom_id, floor_geom_id),
                        compute_grf.get_ground_reaction_force(model, data, lffoot_geom_id, floor_geom_id),
                    ]
                )

                drop_data["joint_position"].append(data.qpos.copy())
                drop_data["joint_velocity"].append(data.qvel.copy())
                drop_data["com_velocity"].append(com_vel)
                drop_data["com_position"].append(data.subtree_com[body_id].copy())
                drop_data["ground_contact_force"].append(grf)
                drop_data["sensor_data"].append(data.sensordata.copy())
                drop_data["muscle_activation"].append(data.ctrl.copy())
                drop_data["muscle_length"].append(data.actuator_length.copy())
                drop_data["muscle_velocity"].append(data.actuator_velocity.copy())
                drop_data["muscle_force"].append(data.actuator_force.copy())
                drop_data["Ia_feedback"].append(Ia.copy())
                drop_data["II_feedback"].append(II.copy())

                # Real-time pacing
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                idx += 1

                if idx - delay_steps - duration >= after_sim_delay_steps:
                    print(f"Completed {system_type} simulation with after-sim delay")
                    break

            # Save data for current system type
            if save_data:
                print(f"Saving data for {system_type}...")
                if "independent" in system_type:
                    filename = f"{system_type}_omega_{omega}_gamma_drive_{gamma_drive:.1f}"
                else:
                    filename = f"{system_type}_omega_{omega}"

                for data_key, data_list in drop_data.items():
                    if len(data_list) > 0:
                        file_path = os.path.join(base_data_dir, f"{filename}_{data_key}.txt")
                        np.savetxt(file_path, np.array(data_list), fmt="%.8f", delimiter="\t")
                    else:
                        print(f"Skipping {data_key}: empty list")


# ========================== #
if __name__ == "__main__":

    path_to_model = "quadruped_experiment/quadruped_ws_onground.xml"
    activation_folder = "../quadruped_activation_data_10_09_2025"

    # User-editable parameters
    omega = 0.7
    activation_scaler = 5.0
    delay_duration = 5.0
    after_sim_delay = delay_duration

    base_data_dir = "../all_data/quadruped_experiment/quadruped_experiment_02_05_2026/hard_floor/II_only"
    save_data = False

    
    # non_independent_systems = ["feedforward","beta", "alpha_gamma_co_activation_with_collateral", "alpha_gamma_co_activation_no_collateral"]
    non_independent_systems = ["beta"]
    # non_independent_systems = []

    # independent_systems = ["independent_with_collateral", "independent_no_collateral"]
    # independent_systems = ["independent_no_collateral"]
    independent_systems = []

    # Gamma activation level
    gamma_drives = np.arange(1, -0.1, -0.1)  # e.g.,  np.array([0.8])

    # ---- Run non-independent systems (no gamma needed) ----
    if non_independent_systems:
        run_quadruped(
            path_to_model=path_to_model,
            activation_folder=activation_folder,
            omega=omega,
            activation_scaler=activation_scaler,
            delay_duration=delay_duration,
            after_sim_delay=after_sim_delay,
            base_data_dir=base_data_dir,
            save_data=save_data,
            system_types=non_independent_systems,
        )

    # ---- Run independent systems (loop over gamma values) ----
    if independent_systems:
        if np.isscalar(gamma_drives):
            gamma_list = [float(gamma_drives)]
        else:
            gamma_list = list(gamma_drives)

        for gamma_drive in gamma_list:
            print(f"\n  Spindle Gain: {gamma_drive}")
            run_quadruped(
                path_to_model=path_to_model,
                activation_folder=activation_folder,
                omega=omega,
                activation_scaler=activation_scaler,
                delay_duration=delay_duration,
                after_sim_delay=after_sim_delay,
                base_data_dir=base_data_dir,
                save_data=save_data,
                gamma_drive=float(gamma_drive),
                system_types=independent_systems,
            )

        
           