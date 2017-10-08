GAS_DENSITY = 2.858
ONE_MPH = 0.44704


import rospy
from pid import PID
from lowpass2 import LowPassFilter2

import os
import json

class Controller(object):
    def __init__(self, yaw_controller, p_brake):
        # TODO: Implement
        # defining 2 PID controllers
        self.throttle_pid = PID(kp=0.4, ki=0.03, kd=0.1, mn=-1.0, mx=1.0)
        self.brake_pid = PID(kp=p_brake, ki=0.0, kd=0.0, mn=0.0, mx=100000.)
        self.yaw_controller = yaw_controller
        self.des_speed_filter = LowPassFilter2(0.75, 0.) 
        self.throttle_brake_offs = -1.0
        self.throttle_direct = 0.0
        self.standstill_velocity = 1.0
        self.standstill_brake = -2.0*p_brake*self.throttle_brake_offs
        self.standstill_filter_time = 1.0
        self.standstill_filter = LowPassFilter2(self.standstill_filter_time, 0.0)
        # take 2 secs to reach max speed, init speed = 0.

    #def control(self, delta_time, lin_vel_err, ang_vel_err):
    def control(self, delta_time, linear_velocity, angular_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        #rospy.logwarn(lin_vel_err)
        
        filt_linear_velocity = self.des_speed_filter.filt(linear_velocity, delta_time)
        filt_lin_vel_err = filt_linear_velocity - current_velocity
        lin_vel_err = linear_velocity - current_velocity
        
        #throttle_ = self.throttle_pid.step(filt_lin_vel_err, delta_time) + self.throttle_direct * linear_velocity
        #if throttle_ < 0.0:
        #    throttle_ = 0.0
        
        #brake_ = self.brake_pid.step(-(lin_vel_err-self.throttle_brake_offs), delta_time)
        #rospy.logwarn(self.throttle_pid.get_state())
        
        if lin_vel_err > self.throttle_brake_offs:
            #self.brake_pid.reset()
            if linear_velocity < self.standstill_velocity and lin_vel_err < 0.0:
                brake_ = self.standstill_filter.filt(self.standstill_brake, delta_time)
                throttle_ = 0.
                self.throttle_pid.reset()
                rospy.logwarn("--- standstill --- "+str(brake_))
            else:
                throttle_ = self.throttle_pid.step(filt_lin_vel_err, delta_time) + self.throttle_direct * linear_velocity
                if throttle_ < 0.0:
                    throttle_ = 0.0
                brake_ = 0.
                self.standstill_filter.init(0.)
        elif lin_vel_err <= self.throttle_brake_offs:
            self.throttle_pid.decay(delta_time, 0.5)
            throttle_ = 0.
            brake_ = self.brake_pid.step(-(lin_vel_err-self.throttle_brake_offs), delta_time)
            self.standstill_filter.init(brake_)
            rospy.logwarn("--- brake --- "+str(brake_))
        else:
            throttle_ = 0.
            brake_ = 0.
            
        steer_ = self.yaw_controller.get_steering(filt_linear_velocity, angular_velocity, current_velocity)
        
        #rospy.logwarn(str(linear_velocity)+" "+str(current_velocity)+" "+str(throttle_)+" "+str(brake_)+" "+str(steer_))
        
        return throttle_, brake_, steer_
    
    def init(self, curr_lin_vel):
        self.throttle_pid.reset()
        self.brake_pid.reset()
        self.des_speed_filter.init(curr_lin_vel)
        self.standstill_filter.init(0.0)

    def reload_params(self):
        #rospy.logwarn(os.path.dirname(os.path.realpath(__file__)))
        fn=os.path.dirname(os.path.realpath(__file__))+"/controlparams.json"
        with open(fn) as json_data:
            d = json.load(json_data)
            speed_err = d['speed_err']
            self.throttle_pid.set_params(speed_err['p'], speed_err['i'], speed_err['d']);
            self.throttle_direct = d['throttle_direct']
            self.des_speed_filter.set_tau(d['lowpass_linvel'])
            self.throttle_brake_offs = d['throttle_brake_offs']
            standstill = d['standstill']
            self.standstill_velocity = standstill['velocity']
            self.standstill_brake = standstill['brake']
            self.standstill_filter_time = standstill['filter_time']
            self.standstill_filter.set_tau(self.standstill_filter_time)
            