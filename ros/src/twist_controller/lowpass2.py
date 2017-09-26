
class LowPassFilter2(object):
    def __init__(self, tau, init_val):
        self.tau = tau
        self.last_val = init_val

    def get(self):
        return self.last_val

    def filt(self, val, ts):
        a = 1. / (self.tau / ts + 1.)
        b = 1. - a
        val = a * val + b * self.last_val

        self.last_val = val
        return val
    
    def init(self, val):
        self.last_val = val
        
    def set_tau(self, tau):
        self.tau = tau
        
