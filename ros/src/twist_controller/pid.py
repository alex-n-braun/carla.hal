
import math

MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = 0.
        self.last_error = None

    def reset(self):
        self.int_val = 0.0
        self.last_error = None
        
    def decay(self, delta_time, tau):
        self.last_error = None
        self.int_val = self.int_val * math.exp(-delta_time/tau)

    def step(self, error, sample_time):
        integral = self.int_val + error * sample_time * self.ki;
        if self.last_error:
            derivative = (error - self.last_error) / sample_time;
        else:
            derivative = 0.0

        y = self.kp * error + self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val
    
    def get_state(self):
        return [self.int_val, self.last_error]
    
    def set_params(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        
