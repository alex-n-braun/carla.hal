GAS_DENSITY = 2.858
ONE_MPH = 0.44704


#import rospy
from pid import PID


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        # defining 2 PID controllers
        self.throttle_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=0.0, mx=1.0)
        self.brake_pid = PID(kp=1.0, ki=0.0, kd=0.0, mn=0.0, mx=1.0)
        self.steer_pid = PID(kp=1.0, ki=0.0, kd=0.0)
        #self.steer_pid = PID(kp=0.2, ki=0.001, kd=0.5)

    def control(self, delta_time, lin_vel_err, ang_vel_err):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        #rospy.logwarn(lin_vel_err)
        throttle_ = self.throttle_pid.step(lin_vel_err, delta_time)
        brake_ = self.brake_pid.step(-lin_vel_err, delta_time)
        steer_ = self.steer_pid.step(ang_vel_err, delta_time);
        return throttle_, 0., steer_
