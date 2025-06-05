attitude_ic = {
    'ic/psi-true-deg': 180,                # Heading (degrees)
    'ic/phi-deg': 0,                       # Roll angle (level)
    'ic/theta-deg': 0,                     # Pitch angle (level)
}

attitude_rate_ic = {
    'ic/p-rad_s': 0.0,                     # Roll rate (rad/s)
    'ic/q-rad_s': 0.0,                     # Pitch rate (rad/s)
    'ic/r-rad_s': 0.0,                     # Yaw rate (rad/s)
}

rate_ic = {
    'ic/u-fps': 100.0,                      # Forward velocity (m/s)
    'ic/v-fps': 0.0,                      # Sideways velocity (m/s)
    'ic/w-fps': 0.0,                      # Vertical velocity (m/s)
}

position_ic = {
    'ic/h-agl-ft': 4000.0,                        # Z position (ft)
}

aero_ic = {
    'ic/alpha-deg': 0.0,              # True angle of attack (degrees)
    'ic/beta-deg': 0.0,               # True sideslip angle (degrees)
}

other_ic = {
    'fcs/throttle-cmd-norm': 0.5,             # Throttle position (normalized)
    'fcs/mixture-cmd-norm': 1,             # Mixture position (normalized)
    'propulsion/magneto_cmd-norm': 3,             # Magneto position (1.0 for on)
    'propulsion/start_cmd': 1
}

ic = {
    'attitude': attitude_ic,
    # 'attitude_rate': attitude_rate_ic,
    'rate': rate_ic,
    'position': position_ic,
    # 'aero': aero_ic,
    'other': other_ic
}

type_randomization_variance = {
    'attitude': 10,
    'attitude_rate': 5,
    'rate': 20,
    'position': 100,
    'aero': 10,
}

