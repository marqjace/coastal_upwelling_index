# Cumulative Upwelling Index Calculation Script

# Created on 2024-10-07 by Jace Marquardt - Oregon State University
# Wind Stress Functions by Filipe Fernandes (https://github.com/ocefpaf)
# Last Updated 2024-12-18 by Jace Marquardt

# Imports 
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import colors # type: ignore
from datetime import datetime, timedelta
import pandas as pd # type: ignore
import metpy # type: ignore
from metpy.calc import wind_components # type: ignore
from metpy.units import units # type: ignore


# Open first dataset up to the 45-day realtime
ds1 = pd.read_csv('C:/Users/marqjace/cui/nwpo3h2024/nwpo3h2024.txt', sep='\s+', header=1, parse_dates=[['#yr', 'mo', 'dy', 'hr', 'mn']])

# Create "time" column and convert it to Datetime
ds1['time'] = pd.to_datetime(ds1['#yr_mo_dy_hr_mn'], format='%Y %m %d %H %M')

# Replace "MM" values with NaNs
ds1.replace('MM', np.nan, inplace=True)

# Open second dataset with 45-day realtime values
ds2 = pd.read_csv('C:/Users/marqjace/cui/nwpo3h2024/realtime.txt', sep='\s+', header=1, parse_dates=[['#yr', 'mo', 'dy', 'hr', 'mn']])

# Create "time" column and convert it to Datetime
ds2['time'] = pd.to_datetime(ds2['#yr_mo_dy_hr_mn'], format='%Y %m %d %H %M')

# Replace "MM" values with NaNs
ds2.replace('MM', np.nan, inplace=True)

# Modify ds2
ds2 = ds2.sort_values(by=['time'], ascending=True)
ds2 = ds2.reset_index(drop=True)
ds2 = ds2.drop(['hPa.1'], axis=1)

# Combine and modify datasets
frames = [ds1,ds2]
ds = pd.concat(frames)
ds = ds.sort_values(by=['time'])
ds = ds.reset_index(drop=True)
ds = ds.set_index(['time'])
ds = ds.drop(['#yr_mo_dy_hr_mn','sec', 'sec.1', 'hPa', 'degC.1', 'degC.2', 'ft', 'degT.1', 'm', 'm/s.1'], axis=1)


# Specify path for export. Creates new file with combined dataset.
path = 'C:/Users/marqjace/cui/nwpo3h2024/2024_tmp.txt'

# Export dataset to text file (keep header row and index column)
with open(path, 'w') as f:
    ds_string = ds.to_string()
    f.write(ds_string)

# Create variables for wind direction, wind speed, and air temperature
wdir = ds['degT'].values
wspeed = ds['m/s'].values
temp = ds['degC'].values

# Calculate U,V wind components
ucomp = []
vcomp = []

for dir, speed in zip(wdir, wspeed):
    u, v = metpy.calc.wind_components(float(speed) * units('m/s'), float(dir) * units.deg)
    ucomp.append(u.magnitude)
    vcomp.append(v.magnitude)

# Convert to DataFrame
ucomp = pd.DataFrame(ucomp, columns=['ucomp'], index=ds.index)
vcomp = pd.DataFrame(vcomp, columns=['vcomp'], index=ds.index)

# Set new columns in dataset
ds['ucomp'] = ucomp
ds['vcomp'] = vcomp

# Assign new variable "vcomp_new"
vcomp_new = ds['vcomp']


# Wind Stress Functions adapted from: https://github.com/pyoceans/python-airsea/blob/master/airsea/windstress.py
kappa = 0.4  # von Karman's constant
Charnock_alpha = 0.011  # Charnock constant for open-ocean
R_roughness = 0.11  # Limiting roughness Reynolds for aerodynamically smooth flow
g = 9.8  # Acceleration due to gravity [m/s^2]

def cdn(sp, z, drag='largepond', Ta=temp):
    """Computes neutral drag coefficient and wind speed at 10 m."""
    sp, z, Ta = np.asarray(sp), np.asarray(z), np.asarray(Ta)
    tol = 0.00001

    if drag == 'largepond':
        a = np.log(z / 10.) / kappa
        u10o = np.zeros(sp.shape)
        cd = 1.15e-3 * np.ones(sp.shape)
        u10 = sp / (1 + a * np.sqrt(cd))
        ii = np.abs(u10 - u10o) > tol

        while np.any(ii):
            u10o = u10
            cd = (4.9e-4 + 6.5e-5 * u10o)
            cd[u10o < 10.15385] = 1.15e-3
            u10 = sp / (1 + a * np.sqrt(cd))
            ii = np.abs(u10 - u10o) > tol

    else:
        raise ValueError('Unknown method')

    return cd, u10

def stress(sp, z=10., drag='largepond', rho_air=1.22, Ta=temp):
    """Computes the neutral wind stress."""
    z, sp = np.asarray(z), np.asarray(sp)
    Ta, rho_air = np.asarray(Ta), np.asarray(rho_air)

    if drag == 'largepond':
        cd, sp = cdn(sp, z, 'largepond')
    else:
        raise ValueError('Unknown method')

    tau = rho_air * (cd * sp * np.abs(sp))

    return tau


# Apply the windstress function to convert m/s to N/m^2
vcomp_Nm2 = stress(vcomp_new, drag='largepond', z=10)

# Create new "stress" column in dataset
ds['stress'] = vcomp_Nm2

# Calculate daily means
ds = ds.resample('D').mean(numeric_only=True)

# Round values to 3 decimal places
ds['stress'] = np.round(ds['stress'], 3)

# Define new variable "stress"
stress = ds['stress']

# Create new columns for positive and negative wind stress
ds['positive_stress'] = ds['stress'].apply(lambda x: x if x > 0 else 0)
ds['negative_stress'] = ds['stress'].apply(lambda x: x if x < 0 else 0)

# Specify path for export. Creates new file with combined dataset.
path = 'C:/Users/marqjace/cui/nwpo3h2024/2024_with_stress.txt'

# Export dataset to text file (keep header row and index column)
with open(path, 'w') as f:
    ds_string = ds.to_string()
    f.write(ds_string)

# Parameters
rolling_window = 30  # Number of days for rolling window
threshold = 0.05      # Minimum difference threshold for upwelling detection
positive_duration = 7  # Minimum number of days of positive dominance to end upwelling

# Calculate the 90-day rolling sum for both positive and negative stress
rolling_positive = ds['stress'].where(ds['stress'] > 0).rolling(window=rolling_window, min_periods=1).sum()
rolling_negative = ds['stress'].where(ds['stress'] < 0).rolling(window=rolling_window, min_periods=1).sum()

# Initialize variables for tracking upwelling season
upwelling_periods = []  # List to store identified upwelling periods
upwelling_start = None  # Variable to track the start of the upwelling season
is_upwelling = False  # Flag for tracking if we are in an upwelling period

# Iterate over the dataset to find upwelling periods based on the 90-day rolling sums
for date in ds.index:
    cum_neg = rolling_negative.loc[date]
    cum_pos = rolling_positive.loc[date]
    
    # Check if the cumulative negative stress is greater than positive over the last 90 days
    if cum_neg - cum_pos > threshold:  # Adjusted condition for upwelling
        if not is_upwelling:  # If not already in an upwelling season
            upwelling_start = date  # Mark start of the upwelling season
            is_upwelling = True  # Set flag to indicate upwelling has started
    else:
        if is_upwelling:  # If we were in an upwelling period
            # Mark the end of the upwelling period
            upwelling_periods.append((upwelling_start, date))
            upwelling_start = None  # Reset for the next potential upwelling period
            is_upwelling = False  # End the upwelling season

# After the loop, check if we ended in an upwelling season
if is_upwelling and upwelling_start is not None:
    upwelling_periods.append((upwelling_start, ds.index[-1]))

# Print the identified upwelling periods
if upwelling_periods:
    for i, (start, end) in enumerate(upwelling_periods, 1):
        print(f"Upwelling season {i}: From {start.date()} to {end.date()}")
else:
    print("No upwelling season detected.")



mask = np.logical_and(ds.index >= start, ds.index <= end)

# Apply the mask to the dataset using the index
new_time = ds.index[mask]

# Create a new windstress variable for the upwelling season
upwell_stress = stress[mask]

# Calculate the cumulative wind stress for the upwelling season
cumulative_sum = np.cumsum(upwell_stress)

# Calculate the current minimum cumulative value 
cumsum_min = cumulative_sum.min()
cumsum_min = round(cumsum_min, 3)
cumsum_min = str(cumsum_min)


# Create plot
fig, ax = plt.subplot_mosaic([['upper', 'upper', 'upper'],
                               ['lower', 'lower', 'lower'],
                                     ['lower', 'lower', 'lower']],
                              figsize=(10, 12), layout="constrained", dpi=300)

colors = c=np.where(stress<0, 'b', 'r')
ax['upper'].scatter(ds.index, stress, c=colors, s=1)

# Fill the area under the curve
ax['upper'].fill_between(ds.index, stress, where=(stress >= 0), interpolate=True, color='r', alpha=1)
ax['upper'].fill_between(ds.index, stress, where=(stress < 0), interpolate=True, color='b', alpha=1)

ax['upper'].set_xticklabels('')
ax['upper'].set_xticks((datetime(2024,1,1), datetime(2024,2,1), datetime(2024,3,1), datetime(2024,4,1), datetime(2024,5,1), datetime(2024,6,1), datetime(2024,7,1), 
            datetime(2024,8,1), datetime(2024,9,1), datetime(2024,10,1), datetime(2024,11,1), datetime(2024,12,1)))
ax['upper'].set_xticklabels(('J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'))
ax['upper'].set_xlim(datetime(2024,1,1), datetime(2025,1,1))
ax['upper'].xaxis.set_label_position('top') 
ax['upper'].xaxis.tick_top()
ax['upper'].tick_params(width=1, top=True, right=True, bottom=False, direction='in', which='both')

sec = ax['upper'].secondary_xaxis(location=0)
sec.set_xticks([datetime(2024,1,1), datetime(2024,3,1), datetime(2024,4,30), datetime(2024,6,29), datetime(2024,8,28), datetime(2024,10,27), datetime(2024,12,26)], labels=['0', '60', '120', '180', '240', '300', '360'])
sec.tick_params(width=1, top=False, right=False, bottom=True, direction='in', which='both')

ax['upper'].set_yticks((-0.3, 0.0, 0.3, 0.6, 0.9))
ax['upper'].set_yticklabels((-0.3, 0.0, 0.3, 0.6, 0.9))
ax['upper'].set_ylim(-0.3, 0.9)

ax['upper'].set_ylabel(r'Wind Stress (N/m$^2$)')
ax['upper'].set_title('2024')

ax['lower'].plot(new_time, cumulative_sum, c='b', alpha=0.75)
ax['lower'].set_xticklabels('')
ax['lower'].set_xticks((datetime(2024,1,1), datetime(2024,2,1), datetime(2024,3,1), datetime(2024,4,1), datetime(2024,5,1), datetime(2024,6,1), datetime(2024,7,1), 
            datetime(2024,8,1), datetime(2024,9,1), datetime(2024,10,1), datetime(2024,11,1), datetime(2024,12,1)))
ax['lower'].set_xticklabels(('J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'))
ax['lower'].set_xlim(datetime(2024,1,1), datetime(2025,1,1))
ax['lower'].xaxis.set_label_position('top') 
ax['lower'].xaxis.tick_top()
ax['lower'].tick_params(width=1, top=True, right=True, bottom=False, direction='in', which='both')
ax['lower'].set_yticks((0, -1, -2, -3, -4, -5, -6))
ax['lower'].set_yticklabels(('0', '', '-2', '', '-4', '', '-6'))

sec2 = ax['lower'].secondary_xaxis(location=0)
sec2.set_xticks([datetime(2024,1,1), datetime(2024,3,1), datetime(2024,4,30), datetime(2024,6,29), datetime(2024,8,28), datetime(2024,10,27), datetime(2024,12,26)], labels=['0', '60', '120', '180', '240', '300', '360'])
sec2.tick_params(width=1, top=False, right=False, bottom=True, direction='in', which='both')

ax['upper'].axvspan(xmin=datetime(2024,3,30), xmax=datetime(2024,10,10), color='k', alpha=0.05)
ax['lower'].axvspan(xmin=datetime(2024,3,30), xmax=datetime(2024,10,10), color='k', alpha=0.05)

ax['lower'].set_ylabel(r'Cumulative Wind Stress (N/m$^2$Days)')
sec2.set_xlabel(r'Yearday 2024')

ax['lower'].text((new_time.max() + timedelta(days=5)), cumulative_sum.min(), cumsum_min + ' ' + r'N/m$^2$Days')

plt.savefig('C:/Users/marqjace/cui/nwpo3h2024/cui_2024.png')