# Cumulative Upwelling Index Calculation Script

# Created on 2024-10-07 by Jace Marquardt - Oregon State University
# Wind Stress Functions by Filipe Fernandes (https://github.com/ocefpaf)
# Last Updated 2025-04-06 by Jace Marquardt

# Imports 
import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import colors # type: ignore
from datetime import datetime, timedelta
import pandas as pd # type: ignore
import metpy # type: ignore
import metpy.calc # type: ignore
from metpy.units import units # type: ignore
from scipy.signal import find_peaks # type: ignore

# Year that you want to examine
year = 2025

# Open first dataset up to the 45-day realtime (found at https://www.ndbc.noaa.gov/station_history.php?station=nwpo3)
ds1 = pd.read_csv(f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/nwpo3_{year}.txt', sep='\s+', header=1)
print(f'\nOpening historical dataset...')

ds1.columns = ds1.columns.str.strip() # Strip whitespace from column names
ds1.rename(columns={'#yr': 'year', 'mo': 'month', 'dy': 'day', 'hr': 'hour', 'mn': 'minute', 'degT': 'wdir'}, inplace=True) # Rename columns for clarity
ds1['time'] = pd.to_datetime(ds1[['year', 'month', 'day', 'hour', 'minute']]) # Convert to datetime
ds1.replace('MM', np.nan, inplace=True) # Replace "MM" values with NaNs

# Open second dataset with 45-day realtime values (found at https://www.ndbc.noaa.gov/station_realtime.php?station=nwpo3)
realtime_filepath = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/realtime.txt'

if os.path.exists(realtime_filepath):
    ds2 = pd.read_csv(f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/realtime.txt', sep='\s+', header=1)
    print(f'Opening realtime dataset...')

    ds2.columns = ds2.columns.str.strip() # Strip whitespace from column names
    ds2.rename(columns={'#yr': 'year', 'mo': 'month', 'dy': 'day', 'hr': 'hour', 'mn': 'minute', 'degT': 'wdir'}, inplace=True) # Rename columns for clarity
    ds2['time'] = pd.to_datetime(ds2[['year', 'month', 'day', 'hour', 'minute']]) # Convert to datetime
    ds2.replace('MM', np.nan, inplace=True) # Replace "MM" values with NaNs

    # Modify ds2
    ds2.replace('MM', np.nan, inplace=True) # Replace "MM" values with NaNs
    ds2.sort_values(by=['time'], ascending=True, inplace=True) # Sort by time
    ds2.reset_index(drop=True, inplace=True) # Reset index
    ds2.drop(['hPa.1'], axis=1, inplace=True) # Drop unnecessary columns

    # Combine and modify datasets
    frames = [ds1,ds2]
    ds = pd.concat(frames) # Concatenate datasets
    ds.sort_values(by=['time'], inplace=True) # Sort by time
    ds.reset_index(drop=True, inplace=True) # Reset index
    ds.set_index(['time'], inplace=True) # Set time as index
    ds.drop(['year', 'month', 'day', 'hour', 'minute', 'sec', 'sec.1', 'hPa', 'degC.1', 'degC.2', 'ft', 'degT.1', 'm', 'm/s.1', 'nmi', 'deg'], axis=1, inplace=True) # Drop unnecessary columns
    print(f"Concatenating datasets...")
else:
    # Combine and modify datasets
    frames = [ds1]
    ds = pd.concat(frames) # Concatenate datasets
    ds.sort_values(by=['time'], inplace=True) # Sort by time
    ds.reset_index(drop=True, inplace=True) # Reset index
    ds.set_index(['time'], inplace=True) # Set time as index
    ds.drop(['year', 'month', 'day', 'hour', 'minute', 'sec', 'sec.1', 'hPa', 'degC.1', 'degC.2', 'ft', 'degT.1', 'm', 'm/s.1', 'nmi', 'deg'], axis=1, inplace=True) # Drop unnecessary columns
    print(f"No realtime dataset found.")

# Export dataset to text file (keep header row and index column)
path = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/{year}_tmp.txt'

with open(path, 'w') as f:
    ds_string = ds.to_string()
    f.write(ds_string)
print(f"\nExporting concatenated dataset as '{year}_tmp.txt'...")

# Create variables for wind direction, wind speed, and air temperature
wdir = ds['wdir'].values
wspeed = ds['m/s'].values
temp = ds['degC'].values

# Calculate U,V wind components
ucomp = []
vcomp = []

for dir, speed in zip(wdir, wspeed):
    u, v = metpy.calc.wind_components(float(speed) * units('m/s'), float(dir) * units.deg)
    ucomp.append(u.magnitude)
    vcomp.append(v.magnitude)
print(f"\nCalculating wind components...")

# Convert to DataFrame
ucomp = pd.DataFrame(ucomp, columns=['ucomp'], index=ds.index)
vcomp = pd.DataFrame(vcomp, columns=['vcomp'], index=ds.index)
ds['ucomp'] = ucomp # Add ucomp to dataset
ds['vcomp'] = vcomp # Add vcomp to dataset
vcomp_new = ds['vcomp'] # Create new variable for vcomp


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
print(f"\nCalculating wind stress...")

ds['stress'] = vcomp_Nm2 # Add wind stress to dataset
ds = ds.resample('D').mean(numeric_only=True) # Resample to daily mean values
ds['stress'] = np.round(ds['stress'], 3) # Round to 3 decimal places
stress = ds['stress'] # Create new variable for stress

# Create new columns for positive and negative wind stress
ds['positive_stress'] = ds['stress'].apply(lambda x: x if x > 0 else 0)
ds['negative_stress'] = ds['stress'].apply(lambda x: x if x < 0 else 0)

# Specify path for export. Creates new file with combined dataset.
path = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/{year}_with_stress.txt'

# Export dataset to text file (keep header row and index column)
with open(path, 'w') as f:
    ds_string = ds.to_string()
    f.write(ds_string)
print(f"\nExporting new dataset as '{year}_with_stress.txt'...")

# Parameters
rolling_window = 120  # Adjusted rolling window size for upwelling detection
threshold = -3.25    # Minimum difference threshold for upwelling detection
min_upwelling_duration = 30  # Minimum number of consecutive days for upwelling

# Calculate the rolling sums for positive and negative stress
rolling_positive = ds['stress'].where(ds['stress'] > 0).rolling(window=rolling_window, min_periods=1).sum()
rolling_negative = ds['stress'].where(ds['stress'] < 0).rolling(window=rolling_window, min_periods=1).sum()

# Initialize variables for tracking upwelling periods
upwelling_periods = []
upwelling_start = None
is_upwelling = False

print(f"\nCalculating upwelling periods...")

# Iterate over the dataset to find upwelling periods
for date in ds.index:
    cum_neg = rolling_negative.loc[date]
    cum_pos = rolling_positive.loc[date]
    diff = cum_neg - cum_pos  # Difference between negative and positive stress

    # Check if the cumulative negative stress exceeds the threshold
    if diff < threshold:
        if not is_upwelling:
            upwelling_start = date  # Mark the start of the upwelling period
            is_upwelling = True
    else:
        if is_upwelling:
            # Check if the upwelling period meets the minimum duration
            if (date - upwelling_start).days >= min_upwelling_duration:
                upwelling_periods.append((upwelling_start, date))
            upwelling_start = None
            is_upwelling = False

# Handle case where the last period is still ongoing
if is_upwelling and upwelling_start is not None:
    upwelling_periods.append((upwelling_start, ds.index[-1]))

# Print the identified upwelling periods
if upwelling_periods:
    for i, (start, end) in enumerate(upwelling_periods, 1):
        print(f"Upwelling season {i}: From {start.date()} to {end.date()}")
else:
    print("No upwelling season detected.")
    # Set default values for start and end if no upwelling periods are detected
    start, end = ds.index[0], ds.index[-1]

# Define start and end for the mask
if upwelling_periods:
    start, end = upwelling_periods[0]  # Example: using the first upwelling period

# Create a mask for values between start and end
mask = np.logical_and(ds.index >= start, ds.index <= end)
new_time = ds.index[mask] # Apply the mask to the dataset using the index
upwell_stress = stress[mask] # Create a new windstress variable for the upwelling season

# Calculate the cumulative wind stress for the upwelling season
cumulative_sum = np.cumsum(upwell_stress)
print(f"\nCalculating cumulative wind stress...")

# Initialize a dictionary to store cumsum_min for each date
cumsum_min_by_date = {}

# Iterate over each date in the dataset index
for current_date in ds.index:
    # Filter cumulative_sum to include only values up to the current date
    filtered_cumulative_sum = cumulative_sum[(new_time >= start) & (new_time <= current_date)]
    
    # Calculate the minimum cumulative value for the filtered range
    if filtered_cumulative_sum.size > 0:  # Check if the array is not empty
        cumsum_min = filtered_cumulative_sum.min()
        cumsum_min = round(cumsum_min, 3)
        cumsum_min_by_date[current_date] = cumsum_min

    # Convert date to yearday format
    yearday = current_date.strftime('%j')  # '%j' gives the day of the year as a zero-padded decimal number (001-366)

# Save the cumsum_min values for each date to a text file
output_path = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/{year}/{year}.txt'

with open(output_path, 'w') as file:
    file.write(f"# Spring Transition Day: {start}\n# Fall Transition Day: {end}\n# Column 1: Yearday\n# Column 2: Northward Wind Stress (N/m^2 Days)\n")
    for date in ds.index:
        # Convert date to yearday format
        yearday = date.strftime('%j')  # '%j' gives the day of the year as a zero-padded decimal number (001-366)
        
        # Check if the date is within the start and end range
        if start <= date <= end:
            cumsum_min = cumsum_min_by_date.get(date, "NaN")  # Get cumsum_min or default to NaN
        else:
            cumsum_min = "NaN"  # Assign NaN for dates outside the range
        
        # Write the yearday and cumsum_min to the file
        file.write(f"{yearday}\t{cumsum_min}\n")

print(f"\nWind stress values saved to '{year}.txt'")


# Create timestamp for the figure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'cui_{year}_{timestamp}.png'

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
ax['upper'].set_xticks((datetime(year,1,1), datetime(year,2,1), datetime(year,3,1), datetime(year,4,1), datetime(year,5,1), datetime(year,6,1), datetime(year,7,1), 
            datetime(year,8,1), datetime(year,9,1), datetime(year,10,1), datetime(year,11,1), datetime(year,12,1)))
ax['upper'].set_xticklabels(('J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'))
ax['upper'].set_xlim(datetime(year,1,1), datetime(year+1,1,1))
ax['upper'].xaxis.set_label_position('top') 
ax['upper'].xaxis.tick_top()
ax['upper'].tick_params(width=1, top=True, right=True, bottom=False, direction='in', which='both')

sec = ax['upper'].secondary_xaxis(location=0)
sec.set_xticks([datetime(year,1,1), datetime(year,3,1), datetime(year,4,30), datetime(year,6,29), datetime(year,8,28), datetime(year,10,27), datetime(year,12,26)], labels=['0', '60', '120', '180', '240', '300', '360'])
sec.tick_params(width=1, top=False, right=False, bottom=True, direction='in', which='both')

ax['upper'].set_yticks((-0.3, 0.0, 0.3, 0.6, 0.9))
ax['upper'].set_yticklabels((-0.3, 0.0, 0.3, 0.6, 0.9))
ax['upper'].set_ylim(-0.3, 0.9)

ax['upper'].set_ylabel(r'Wind Stress (N/m$^2$)')
ax['upper'].set_title(f'{year}')

ax['lower'].plot(new_time, cumulative_sum, c='b', alpha=0.75)
ax['lower'].set_xticklabels('')
ax['lower'].set_xticks((datetime(year,1,1), datetime(year,2,1), datetime(year,3,1), datetime(year,4,1), datetime(year,5,1), datetime(year,6,1), datetime(year,7,1), 
            datetime(year,8,1), datetime(year,9,1), datetime(year,10,1), datetime(year,11,1), datetime(year,12,1)))
ax['lower'].set_xticklabels(('J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'))
ax['lower'].set_xlim(datetime(year,1,1), datetime(year+1,1,1))
ax['lower'].xaxis.set_label_position('top') 
ax['lower'].xaxis.tick_top()
ax['lower'].tick_params(width=1, top=True, right=True, bottom=False, direction='in', which='both')
ax['lower'].set_yticks((0, -1, -2, -3, -4))
ax['lower'].set_yticklabels((0, -1, -2, -3, -4))

sec2 = ax['lower'].secondary_xaxis(location=0)
sec2.set_xticks([datetime(year,1,1), datetime(year,3,1), datetime(year,4,30), datetime(year,6,29), datetime(year,8,28), datetime(year,10,27), datetime(year,12,26)], labels=['0', '60', '120', '180', '240', '300', '360'])
sec2.tick_params(width=1, top=False, right=False, bottom=True, direction='in', which='both')

ax['upper'].axvspan(xmin=start, xmax=end, color='k', alpha=0.05)
ax['lower'].axvspan(xmin=start, xmax=end, color='k', alpha=0.05)

ax['lower'].set_ylabel(r'Cumulative Wind Stress (N/m$^2$Days)')
sec2.set_xlabel(f'Yearday {year}')

ax['lower'].text((new_time.max() + timedelta(days=5)), cumulative_sum.min(), str(round(cumulative_sum.min(),3)) + ' ' + r'N/m$^2$Days')

plt.savefig(f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/coastal_upwelling_index/figures/{filename}', dpi=300)
print(f"Figure saved as '{filename}'")
print(f"\nDone!")