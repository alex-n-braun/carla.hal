#ifndef LOWPASS_H
#define LOWPASS_H

namespace waypoint_follower
{
class Lowpass
{
private:
    double tau;
    double last_val;
public:
    Lowpass(double tau_, double init_val_);
    double get();
    double filt(double val_, double ts_);
    void init(double val_);
    void set_tau(double tau_);
};
}

#endif  // LOWPASS_H
