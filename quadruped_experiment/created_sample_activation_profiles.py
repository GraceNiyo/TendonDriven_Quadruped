import numpy as np
import mujoco 
import time
import get_activations
import os 

path_to_model = "quadruped_experiment/quadruped_ws_inair.xml"

activation_folder = "../quadruped_activation_data_10_09_2025" 
os.makedirs(activation_folder, exist_ok=True)

for omega in np.arange(0.1, 1.1, .1):  # Increment omega from 0.1 to 1.2 (inclusive)
    try: 
        activation_file, desired_qpos, desired_qvel, desired_qacc = get_activations.compute_and_save_activations(
            path_to_model, 
            omega, 
            dt=0.001, 
            duration_in_seconds=10,
            activation_folder=activation_folder)

        muscle_activations = np.loadtxt(activation_file)
        print(f"Simulated omega: {omega}, Activation file: {activation_file}")

    except Exception as e:
        print(f"An error occurred while simulating omega {omega}: {e}")
        continue

    


