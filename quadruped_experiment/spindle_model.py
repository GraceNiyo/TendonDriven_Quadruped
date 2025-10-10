""" 
Author: Grace Niyo
Date: 2025-07-30
Summary: 
-- Spindle model based on Enander et al. (2022) " A model for self-organization of sensorimotor function: the spinal monosynatic reflex."
-- Muscle parameters matches MuJoCo's interanal derivation and description of their muscle model.
"""

import numpy as np

def gamma_driven_spindle_model_(actuator_length, actuator_velocity, actuator_lengthrange, vmax = 1.6, gamma_static = None, gamma_dynamic = None, beta_drive = None, muscle_lengthrange=(0.5, 1.5)):

    """
    Args: 
        actuator_length (float): Current length of the actuator.
        actuator_velocity (float): Current velocity of the actuator.
        actuator_lengthrange (tuple): Range of feasible lengths of the actuator’s transmission.
        vmax (float): Shortening velocity at which muscle force drops to zero, in units of L0 per second.
        gamma_static (float): Static gamma motor neuron activation.
        gamma_dynamic (float): Dynamic gamma motor neuron activation.
        muscle_lengthrange (tuple): Operating length range of the muscle, in units of L0.
        """


    min_length, max_length = actuator_lengthrange

    if (muscle_lengthrange[1] - muscle_lengthrange[0]) <= 0:
        raise ValueError("The 'muscle_lengthrange' parameter must have a positive span (max > min).")
        
    
    optimal_muscle_length , tendon_slack_length = compute_l0_and_lt((min_length, max_length), muscle_lengthrange)

    muscle_length_lo = (actuator_length - tendon_slack_length) / optimal_muscle_length # muscle length in lo/s units
    normalized_muscle_length = (muscle_length_lo - muscle_lengthrange[0]) / (muscle_lengthrange[1] - muscle_lengthrange[0])  # scale muscle length between 0 to 1
    normalized_muscle_length = np.clip(normalized_muscle_length, 0.0, 1.0)  # Ensure length is within [0, 1]

    normalized_muscle_velocity = actuator_velocity / (optimal_muscle_length * vmax)
    normalized_muscle_velocity = np.clip(normalized_muscle_velocity, -1.0, 1.0)  # Ensure velocity is within [-1, 1]

    Ia, II = 0.0, 0.0 # Initialize outputs
    
    
    if beta_drive is not None:

        if gamma_static is not None or gamma_dynamic is not None:
            raise ValueError("If beta_drive is provided, gamma_static and gamma_dynamic should not be provided.")
        
        beta_drive_clipped = np.clip(beta_drive, 0.0, 1.0)
        II = (((normalized_muscle_length - 0.2)* 1.25) + beta_drive_clipped) / (2.0) # spindle group II afferent
        II = np.clip(II, 0.0, 1.0) # Ensure II is within [0, 1]

        Ia = ((( 1.5 + np.log10( beta_drive_clipped + 0.1))) * normalized_muscle_velocity + beta_drive_clipped + (II * 0.2)) / 2 # spindle group I afferent
        Ia = np.clip(Ia, 0.0, 1.0) # Ensure Ia is within [0, 1]

    else:
        if gamma_static is None or gamma_dynamic is None:
            raise ValueError("If beta_drive is None, both gamma_static and gamma_dynamic must be provided.")
        


        gamma_static_clipped = np.clip(gamma_static, 0.0, 1.0)
        gamma_dynamic_clipped = np.clip(gamma_dynamic, 0.0, 1.0)

        II = (((normalized_muscle_length - 0.2)* 1.25) + gamma_static_clipped) / (2.0) # spindle group II afferent
        II = np.clip(II, 0.0, 1.0) # Ensure II is within [0, 1]

        Ia = ((( 1.5 + np.log10( gamma_dynamic_clipped + 0.1))) * normalized_muscle_velocity + gamma_dynamic_clipped + (II * 0.2)) / 2 # spindle group I afferent
        Ia = np.clip(Ia, 0.0, 1.0) # Ensure Ia is within [0, 1]

    return Ia, II

def compute_l0_and_lt(actuator_lengthrange, normalized_muscle_lengthrange=(0.5, 1.5)):
    """
    Computes L0 (optimal muscle fiber length) and LT (tendon slack length)
    based on MuJoCo's internal derivation method.

    Args:
        actuator_lengthrange (tuple): Range of feasible lengths of the actuator’s transmission.
        muscle_lengthrange (tuple): Operating length range of the muscle, in units of L0.
    """
    
    min_actuator_length, max_actuator_length = actuator_lengthrange
    min_LM_over_L0, max_LM_over_L0 = normalized_muscle_lengthrange

    if (max_LM_over_L0 - min_LM_over_L0) <= 0:
        raise ValueError("The 'muscle_lengthrange' must have a positive span (max > min).")

    # Equation 1: (min_actuator_length - LT) / L0 = min_LM_over_L0
    # Equation 2: (max_actuator_length - LT) / L0 = max_LM_over_L0

    # Solve for L0:
    L0 = (max_actuator_length - min_actuator_length) / (max_LM_over_L0 - min_LM_over_L0)

    # Substitute L0 back into Eq. 1 to solve for LT:
    LT = min_actuator_length - (min_LM_over_L0 * L0)

    return L0, LT

if __name__ == "__main__":
    # T_RB_M0
    test_muscle_params = {
        "lengthrange": (0.459547, 0.768536), 
        "vmax": 1.6,
        "muscle_lengthrange": (0.5, 1.5)  # range for muscle length in L0 units
    }

    # Hypothetical current state
    current_actuator_length = 0.55  # actuator length at qpos0
    current_actuator_velocity = 0

    # Scenario 1: Using Beta Drive
    print("\n--- Scenario 1: Using Beta Drive ---")
    beta_level = 0.3
    try:
        ia_beta, ii_beta = gamma_driven_spindle_model_(
            actuator_length=current_actuator_length,
            actuator_velocity=current_actuator_velocity,
            actuator_lengthrange=test_muscle_params["lengthrange"],
            beta_drive=beta_level,
            muscle_lengthrange=test_muscle_params["muscle_lengthrange"]
        )
        print(f"Beta Drive: {beta_level:.2f}")
        print(f"Ia Signal (Beta): {ia_beta:.4f}")
        print(f"II Signal (Beta): {ii_beta:.4f}")
    except ValueError as e:
        print(f"Error in Beta Drive scenario: {e}")

    # Scenario 2: Using Separate Gamma Drives
    print("\n--- Scenario 2: Using Separate Gamma Drives ---")
    gamma_s_level = 0
    gamma_d_level = 0
    try:
        ia_gamma, ii_gamma = gamma_driven_spindle_model_(
            actuator_length=current_actuator_length,
            actuator_velocity=current_actuator_velocity,
            actuator_lengthrange=test_muscle_params["lengthrange"],
            vmax=test_muscle_params["vmax"],
            gamma_static=gamma_s_level,
            gamma_dynamic=gamma_d_level,
            muscle_lengthrange=test_muscle_params["muscle_lengthrange"]
        )
        print(f"Gamma Static: {gamma_s_level:.2f}, Gamma Dynamic: {gamma_d_level:.2f}")
        print(f"Ia Signal (Gamma): {ia_gamma:.4f}")
        print(f"II Signal (Gamma): {ii_gamma:.4f}")
    except ValueError as e:
        print(f"Error in Gamma Drive scenario: {e}")

    # Scenario 3: Error case (beta_drive AND gamma_static/dynamic provided)
    print("\n--- Scenario 3: Error Case (Beta Drive AND Gamma Provided) ---")
    try:
        gamma_driven_spindle_model_(
            actuator_length=current_actuator_length,
            actuator_velocity=current_actuator_velocity,
            actuator_lengthrange=test_muscle_params["lengthrange"],
            vmax=test_muscle_params["vmax"],
            beta_drive=0.5,
            gamma_static=0.5,
            gamma_dynamic=0.5,
            muscle_lengthrange=test_muscle_params["muscle_lengthrange"]
        )
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    # Scenario 4: Error case (neither beta_drive nor gamma provided)
    print("\n--- Scenario 4: Error Case (Neither Beta Drive Nor Gamma Provided) ---")
    try:
        gamma_driven_spindle_model_(
            actuator_length=current_actuator_length,
            actuator_velocity=current_actuator_velocity,
            actuator_lengthrange=test_muscle_params["lengthrange"],
            vmax=test_muscle_params["vmax"],
            muscle_lengthrange=test_muscle_params["muscle_lengthrange"]
        )
    except ValueError as e:
        print(f"Correctly caught error: {e}")


   