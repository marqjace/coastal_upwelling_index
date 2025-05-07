# Function to find the upwelling season start and end dates
# Author: Jace Marquardt
# Last updated 2025-05-07

import pandas as pd

def find_upwell(df, min_duration='7D', upwell_threshold=-0.075, max_threshold=-0.1, downwell_threshold=0.025):
    """
    Find upwelling and downwelling events in a dataset based on slope, duration, and stress threshold.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'time' as index and 'stress' as a column.
        min_duration (str or pd.Timedelta): Minimum duration for stress to remain negative.
        stress_threshold (float): Minimum stress value (negative) for an event to be considered valid.
    
    Returns:
        pd.DatetimeIndex: Start dates of upwelling events.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Find where stress is zero or crosses zero
    zero_crossings = df['stress'] == 0
    zero_crossings |= (df['stress'].shift(1) > 0) & (df['stress'] < 0)  # Detect zero crossings

    # Create a column to store the last zero crossing
    df['last_zero'] = pd.Series(df.index.where(zero_crossings), index=df.index).ffill()

    # Find where stress is negative
    negative_stress = df['stress'] < 0

    # Group consecutive negative periods
    df['negative_group'] = (negative_stress != negative_stress.shift()).cumsum()  # Group consecutive True values
    negative_groups = df.loc[negative_stress].groupby('negative_group')

    # Filter groups by minimum duration and stress threshold
    min_duration = pd.to_timedelta(min_duration)
    valid_groups = []
    for group_id, group in negative_groups:
        duration = group.index[-1] - group.index[0]
        min_stress = group['stress'].min()
        if duration >= min_duration and min_stress <= upwell_threshold or min_stress <= max_threshold:
            valid_groups.append(group)

    # Extract start dates of valid groups
    start_dates = pd.DatetimeIndex([group['last_zero'].iloc[0] for group in valid_groups])

    if not start_dates.empty:  # Ensure there is at least one start date
        first_start_date = start_dates[0]  # Get the first start date

    # Create a mask to filter the dataset
    mask = df.index >= first_start_date

    # Apply the mask to the dataset
    upwelling_period = df[mask].copy()  # Ensure it's a copy to avoid SettingWithCopyWarning

    # Find where stress is positive
    positive_stress = upwelling_period['stress'] > 0

    # Group consecutive positive periods
    upwelling_period.loc[:, 'positive_group'] = (positive_stress != positive_stress.shift()).cumsum()  # Group consecutive True values
    positive_groups = upwelling_period.loc[positive_stress].groupby('positive_group')

    valid_groups2 = []
    for group_id, group in positive_groups:
        duration = group.index[-1] - group.index[0]
        max_stress = group['stress'].max()
        if duration >= min_duration and max_stress >= downwell_threshold:
            valid_groups2.append(group)

    # Extract end dates of valid groups
    end_dates = pd.DatetimeIndex([group['last_zero'].iloc[0] for group in valid_groups2])

    if not end_dates.empty:
        first_end_date = end_dates[0]
        mask = upwelling_period.index <= first_end_date
        upwelling_period = upwelling_period[mask]
        print(f"The upwelling period starts on {first_start_date} and ends on {first_end_date}")
    else:
        print(f"The upwelling period starts on {first_start_date}, but no valid end date was found.")
        first_end_date = None  # Optional: Assign None if no end date is found

    return upwelling_period