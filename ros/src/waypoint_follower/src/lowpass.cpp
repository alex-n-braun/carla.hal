 
#include "lowpass.h"

namespace waypoint_follower
{
    Lowpass::Lowpass(double tau_, double init_val_)
    {
        tau = tau_;
        last_val = init_val_;
    }
    
    double Lowpass::get()
    {
        return last_val;
    }
    
    double Lowpass::filt(double val_, double ts_)
    {
        double a_( 1. / (tau / ts_ + 1.) );
        double b_( 1. - a_ );
        double new_val_( a_ * val_ + b_ * last_val );
        
        last_val = new_val_;
        return new_val_;
    }
    
    void Lowpass::init(double val_)
    {
        last_val = val_;
    }
    
    void Lowpass::set_tau(double tau_)
    {
        tau = tau_;
    }
}