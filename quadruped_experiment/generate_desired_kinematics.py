'''
Generate  desired kinematics for the quadruped robot 
'''
import numpy as np


def positions_to_kinematics_fcn(q0, q1, dt=0.01):
	kinematics=np.transpose(
	np.concatenate(
		(
			[[q0],
			[q1],
			[np.gradient(q0)/dt],
			[np.gradient(q1)/dt],
			[np.gradient(np.gradient(q0)/dt)/dt],
			[np.gradient(np.gradient(q1)/dt)/dt]]),
		axis=0
		)
	)
	return kinematics

# def combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF):
# 	attempt_kinematics = np.concatenate(
# 		(
# 			attempt_kinematics_RB[:,0:2], attempt_kinematics_RF[:,0:2], attempt_kinematics_LB[:,0:2], attempt_kinematics_LF[:,0:2],
# 			attempt_kinematics_RB[:,2:4], attempt_kinematics_RF[:,2:4], attempt_kinematics_LB[:,2:4], attempt_kinematics_LF[:,2:4],
# 			attempt_kinematics_RB[:,4:6], attempt_kinematics_RF[:,4:6], attempt_kinematics_LB[:,4:6], attempt_kinematics_LF[:,4:6]
# 		),
# 		axis=1
# 	)
# 	return attempt_kinematics
def combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF):
   
    joint_positions = np.concatenate(
        (
            attempt_kinematics_RB[:, 0:2], 
            attempt_kinematics_RF[:, 0:2], 
            attempt_kinematics_LB[:, 0:2], 
            attempt_kinematics_LF[:, 0:2]
        ),
        axis=1
    )
    
    joint_velocities = np.concatenate(
        (
            attempt_kinematics_RB[:, 2:4], 
            attempt_kinematics_RF[:, 2:4], 
            attempt_kinematics_LB[:, 2:4], 
            attempt_kinematics_LF[:, 2:4]
        ),
        axis=1
    )
    
    joint_accelerations = np.concatenate(
        (
            attempt_kinematics_RB[:, 4:6], 
            attempt_kinematics_RF[:, 4:6], 
            attempt_kinematics_LB[:, 4:6], 
            attempt_kinematics_LF[:, 4:6]
        ),
        axis=1
    )
    
    return joint_positions, joint_velocities, joint_accelerations

def sinusoidal_CPG_fcn(w = 1, phi = 0, lower_band = -1, upper_band = 1, attempt_length = 5 , dt=0.01):
	number_of_attempt_samples = int(np.round(attempt_length/dt))
	q0 = np.zeros(number_of_attempt_samples)
	for ii in range(number_of_attempt_samples):
		q0[ii]=np.sin((2*np.pi*w*ii/(number_of_attempt_samples/attempt_length))+phi)
	q0 = (q0+1)/2 # normalize 0-1
	q0 = q0 * (upper_band-lower_band)
	q0 = q0 + lower_band
	return q0


def create_cyclical_movements_fcn(omega=1.5, attempt_length=10, dt=0.01):
	q0a = sinusoidal_CPG_fcn(w = omega, phi = 0, lower_band = -.8, upper_band = .6, attempt_length = attempt_length , dt=dt)
	q1a = sinusoidal_CPG_fcn(w = omega, phi = np.pi/2, lower_band = -1, upper_band = .8, attempt_length = attempt_length , dt=dt)

	q0b = sinusoidal_CPG_fcn(w = omega, phi = np.pi, lower_band = -.8, upper_band = .6, attempt_length = attempt_length , dt=dt)
	q1b = sinusoidal_CPG_fcn(w = omega, phi = -np.pi/2, lower_band = -1, upper_band = .8, attempt_length = attempt_length , dt=dt)
	
	attempt_kinematics_RB = positions_to_kinematics_fcn(q0a, q1a, dt)
	attempt_kinematics_RF = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LB = positions_to_kinematics_fcn(q0b, q1b, dt)
	attempt_kinematics_LF = positions_to_kinematics_fcn(q0a, q1a, dt)
	joint_pos,joint_vel,joint_acc = combine_4leg_kinematics(attempt_kinematics_RB, attempt_kinematics_RF, attempt_kinematics_LB, attempt_kinematics_LF)
	return joint_pos, joint_vel, joint_acc