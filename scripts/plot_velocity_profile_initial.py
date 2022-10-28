import numpy as np
import matplotlib.pyplot as plt
import plonk
from scipy import stats
R = np.linspace(10,300,100)




snap = plonk.load_snap('ICS_T200K.h5')
snap.set_units(position='au', density='g/cm^3',smoothing_length='au',velocity='km/s')
sink = snap.sinks[0]
snap['rad'] = np.sqrt((snap['x'] - sink['position'][0]) ** 2 + (snap['y']-sink['position'][1]) ** 2)
snap['velmag'] = np.sqrt((snap['v_x'] - sink['velocity'][0]) ** 2 + (snap['v_y']-sink['velocity'][1]) ** 2)
snap.add_quantities('disc')
snap.set_central_body(0)
accreted_mask = snap['smoothing_length'] > 0
snap = snap[accreted_mask]
def calculate_number_in_bin(binned_quantity,mean_quantity):
    return stats.binned_statistic(binned_quantity, mean_quantity, 'count', bins=100)

count = calculate_number_in_bin(snap['rad'],snap['m'])
mass_in_bin = np.cumsum(count[0]) * snap['mass'][0]




def calculate_kep_vel(R):
    ''' The Keplerian velocity in km/s
    '''
    return (((1*0.8)/(R))**0.5 * 2.978E6)/1e5

def calculate_self_gravity(mass_in_bin,count):
    return (((1*(0.8+mass_in_bin.magnitude))/(count[1][1:]))**0.5 * 2.978E6)/1e5


plt.plot(R,calculate_self_gravity(mass_in_bin,count),
        label=r'Analytical Solution for $V_{\rm kep}$ with self gravity',c='k',
        linestyle = 'solid')
plt.plot(R,calculate_kep_vel(R),
        label=r'Analytical Solution for $V_{\rm kep}$ without self gravity',c='k',
        linestyle = 'dashed')

plt.scatter(snap['rad'] ,snap['velmag'],s=0.25,c='red',label = 'IC data')
plt.ylabel('Velocity $(\\rm km\,s^{-1})$',fontsize=15)
plt.xlabel('R [AU]',fontsize=15)
plt.legend()

plt.savefig('inital_velocity_profile.png',dpi=200)
