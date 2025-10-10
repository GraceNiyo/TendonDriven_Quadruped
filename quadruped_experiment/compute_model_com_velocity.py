import mujoco
import numpy as np

def compute_model_com_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Computes the center of mass (CoM) velocity of the entire MuJoCo model.

    Args:
        model: The MuJoCo model object.
        data: The MuJoCo data object, containing the simulation state.

    Returns:
        A 3D NumPy array representing the model's CoM linear velocity.
    """
    # Initialize total momentum vector
    total_momentum = np.zeros(3)

    # Calculate total mass of the model
    # Note: model.body_mass includes the world body at index 0, but its mass is 0,
    # so we can simply sum all masses.
    total_mass = np.sum(model.body_mass)

    # Loop through all bodies (from index 1, excluding the world body)
    for i in range(1, model.nbody):
        body_mass = model.body_mass[i]
        
        # data.cvel stores a 6D spatial vector: [angular_x, angular_y, angular_z, linear_x, linear_y, linear_z].
        # We need the last 3 elements for the translational velocity.
        body_vel = data.cvel[i, 3:6]
        
        # Calculate the linear momentum for this body (mass * velocity)
        # and add it to the total momentum.
        total_momentum += body_mass * body_vel

    # Calculate the model's CoM velocity (total momentum / total mass)
    if total_mass > 0:
        model_com_velocity = total_momentum / total_mass
    else:
        # Handle the case of a massless model to avoid division by zero
        model_com_velocity = np.zeros(3)
        
    return model_com_velocity