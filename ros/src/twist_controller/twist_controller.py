GAS_DENSITY = 2.858
ONE_MPH = 0.44704


#import rospy
from pid import PID


class Controller(object):
    def __init__(self, yaw_controller, p_brake):
        # TODO: Implement
        # defining 2 PID controllers
        self.throttle_pid = PID(kp=0.15, ki=0.0025, kd=0.0, mn=0.0, mx=1.0)
        self.brake_pid = PID(kp=p_brake, ki=0.0, kd=0.0, mn=0.0, mx=10000.)
        self.yaw_controller = yaw_controller

    #def control(self, delta_time, lin_vel_err, ang_vel_err):
    def control(self, delta_time, linear_velocity, angular_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        #rospy.logwarn(lin_vel_err)
        lin_vel_err = linear_velocity - current_velocity
        throttle_ = self.throttle_pid.step(lin_vel_err, delta_time)
        brake_ = self.brake_pid.step(-lin_vel_err, delta_time)
        if lin_vel_err > -0.1:
            #self.brake_pid.reset()
            brake_ = 0.
        elif lin_vel_err<=-0.1:
            self.throttle_pid.reset()
            throttle_ = 0.
        else:
            throttle_ = 0.
            brake_ = 0.
            
        steer_ = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        return throttle_, brake_, steer_
