# Author: Grace Niyo
# Date 2026-02-25
def allometric_scaler( M, Torsal_p = 0.4, Thigh_p= 0.35, Shin_p = 0.25):
    
    # Function takes in the mass of the model and the segment proportion. Return the alommetric equivalent size and new parameters for the torso, thigh and shin for MuJoCo modeling. Reference paper:  Lindstedt and Hoppeler (2023), "Allometry: revealing evolution's engineering principles"

   # Under geometric similarity, the length of a segment (L) scales with body mass (M) to the power of 1/3.
    a = 1
    b = 1/3  
    L = a*M**(b)  

    # Unit model parameters 
    Torsal_p = 0.4
    Thigh_p= 0.35
    Shin_p = 0.25
    geom_torso = [0.2, 0.1, 0.05]  # box half_length, half_width, half_depth
    geom_thigh = [0.04497, 0.13003]  # capsule radius and half-length
    geom_shin = [0.03783, 0.08717]   # capsule radius and half-length

    scaled_torso_length = L*Torsal_p
    scaled_thigh_length = L*Thigh_p
    scaled_shin_length = L*Shin_p

    new_torso_width = scaled_torso_length /2

    # now given the new scaled_thigh legnth we can calculate the new thigh geom parameters
    new_geom_thigh_0 = geom_thigh[0] * (scaled_thigh_length / ((geom_thigh[0] + geom_thigh[1])*2))
    new_geom_thigh_1 = geom_thigh[1] * (scaled_thigh_length / ((geom_thigh[0] + geom_thigh[1])*2))     

    # now given the new scaled_shin legnth we can calculate the new shin geom parameters
    new_geom_shin_0 = geom_shin[0] * (scaled_shin_length / ((geom_shin[0] + geom_shin[1])*2))
    new_geom_shin_1 = geom_shin[1] * (scaled_shin_length / ((geom_shin[0] + geom_shin[1])*2))

    # return value for thigh and shin with 5 front digits for better readability
    
    return [L, new_torso_width, round(new_geom_thigh_0, 5), round(new_geom_thigh_1, 5), round(new_geom_shin_0, 5), round(new_geom_shin_1, 5)]

print(allometric_scaler(0.25))














