import mujoco
import numpy as np


def get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id):
    """
    Calculates the 3D ground reaction force vector on the foot in the world frame.
    """
    # Initialize a vector to store the total contact force for this step
    grf_vector = np.zeros(3)
    
    # Iterate through all contacts in the simulation
    for i in range(data.ncon):
        contact = data.contact[i]

        # Check if the contact involves the foot and the floor
        geom1_is_foot = contact.geom1 == foot_geom_id
        geom2_is_floor = contact.geom2 == floor_geom_id
        
        geom1_is_floor = contact.geom1 == floor_geom_id
        geom2_is_foot = contact.geom2 == foot_geom_id

        if (geom1_is_foot and geom2_is_floor) or \
           (geom1_is_floor and geom2_is_foot):
            
            # Buffer to store the 6D contact force (force + torque)
            contact_force_buffer = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, contact_force_buffer)
            
            # Rotate the force from the contact frame to the world frame
            contact_frame_rot = contact.frame.reshape(3, 3)
            force_in_world = contact_frame_rot.T @ contact_force_buffer[:3]
            
            # mj_contactForce calculates force of geom1 on geom2.
            # We want the force ON the foot.
            if geom1_is_floor:
                grf_vector += force_in_world
            else:
                grf_vector -= force_in_world
    
    return grf_vector