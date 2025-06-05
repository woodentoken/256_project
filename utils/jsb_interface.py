import jsbsim
import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
import ipdb
from utils.constants import P
import flightgear_python
from flightgear_python.fg_if import FDMConnection
import time
from pprint import pprint
from flightgear_python.fg_if import TelnetConnection


def main():
    telnet()

def fdm_callback(fdm_data, event_pipe):
    if event_pipe.child_poll():
        phi_rad_child, = event_pipe.child_recv()
        fdm_data['phi_rad'] = phi_rad_child
    fdm_data.alt_m = fdm_data.alt_m + 0.1
    return fdm_data

def JSB():
    fdm = jsbsim.FGFDMExec(None)
    jsbsim.FGJSBBase().debug_lvl=0
    fdm.load_script('scripts/c1723.xml')
    fdm.run_ic()

    times = []
    altitude = []

    while fdm.run():
        times.append(fdm.get_sim_time())
        altitude.append(fdm[P['v_ms']])  # Convert meters to feet
        pass

    print('Simulation finished.')

    figure, ax = plt.subplots()
    ax.plot(times, altitude)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (ft)')
    plt.show()


def local_connection():
    fdm_conn = FDMConnection()
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    fdm_conn.start()

    phi_rad_parent = 0.0
    while True:
        phi_rad_parent += 0.1
        fdm_event_pipe.parent_send((phi_rad_parent,))
        time.sleep(1.0)

def telnet():
    """
    Start FlightGear with `--telnet=socket,bi,60,localhost,5500,tcp`
    """
    telnet_conn = TelnetConnection('localhost', 5500)
    telnet_conn.connect()  # Make an actual connection
    telnet_props = telnet_conn.list_props('/', recurse_limit=0)
    pprint(telnet_props)  # List the top-level properties, no recursion

    while True:
        alt_ft = telnet_conn.get_prop('/position/altitude-ft')
        print(f'Altitude: {alt_ft:.1f}ft')
        telnet_conn.set_prop('/position/altitude-ft', alt_ft + 20.0)
        time.sleep(0.1)


if __name__=='__main__':
    main()
