import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    ## STUDENT CODE GOES HERE

    J = calcJacobian(q_in)

    # Compute the end effector velocities using the Jacobian
    dp = dq.reshape((7, 1))
    velocity = J @ dq
    velocity = velocity.reshape((6, 1))

    assert velocity.shape == (6, 1)

    return velocity
