
# module to find LLJ in WRF data from tower

import numpy as np
import pandas as pd
import xarray as xr

def findLLJevents(df, u_min=3.0, dh=20.0, dumin = 1.5, duminOverUjet = 0.1, N_height=65, N_dt_per_min=9):
    N_per_time = N_height * N_dt_per_min

    jet_df = pd.DataFrame(columns=['time','is_jet', 'jet_nose_height','jet_nose_top'])

    for (i, (time, height)) in enumerate(df.index):
        if (i % N_per_time == 0):
            profile = df.loc[time]
            heights = profile.index

            # min velocity
            mask1 = profile['windspeed'][140.0] > u_min

            # peak within range
            umax = profile['windspeed'].max(axis=0)
            z_umax = profile['windspeed'].idxmax(axis=0)
            i_z_umax = z_umax / dh
            mask2a = z_umax > heights[0]
            mask2b = z_umax < heights[-1]
            mask2 = mask2a & mask2b

            # dropoff
            profileup = profile['windspeed'][z_umax:heights[-1]]
            diffprofileup = np.diff(profileup)
            min_idx = np.where(diffprofileup > 0.0)[0]
            if len(min_idx) == 0:
                z_locmin = heights[-1]
            else:
                z_locmin = heights[min_idx[0]]
            u_locmin = profile['windspeed'][z_locmin]

            du = umax - u_locmin
            mask3a = du > dumin
            mask3b = du/umax > duminOverUjet
            mask3 = mask3a & mask3b
            
            wd_umax = profile['winddirection'][z_umax]
            surf_ws = profile['windspeed'].loc[0.0]
            surf_wd = profile['winddirection'].loc[0.0]
            hh_ws = profile['windspeed'].loc[140.0]
            hh_wd = profile['winddirection'].loc[140.0]

            isjet = mask1 & mask2 & mask3

            if ~isjet:
                z_umax = -999
                z_locmin = -999
                umax = -999
                wd_umax = -999

            new_jet = pd.DataFrame(columns=['time','is_jet', 'jet_nose_height','jet_nose_top', 'jet_nose_WS','jet_nose_WD','surface_ws','surface_wd','HH_ws','HH_wd'],data=np.array([[time, isjet, z_umax, z_locmin, umax, wd_umax, surf_ws, surf_wd, hh_ws, hh_wd]]))
            jet_df = jet_df.append(new_jet,ignore_index=True)
    
    return jet_df


def findLLJevents_xr_old(time_height, max_height, u_min=3.0, dh=20.0, dumin = 1.5, duminOverUjet = 0.1, N_height=65, skip_times=1):
    
    times = time_height['Time']
    skip_times = 1
    times = times[0::skip_times]
    N_time = len(times)

    is_jet          = np.ndarray(N_time, dtype=bool)
    jet_nose_height = np.ndarray(N_time)
    jet_nose_speed  = np.ndarray(N_time)
        
    heights = time_height.isel(Time=0)['height'].values
    iz_HH = (np.abs(heights - 140.0)).argmin()
    iz_maxheight = (np.abs(heights - max_height)).argmin()
            
    # downsample the velocity profile
    time_height = time_height.isel(bottom_top = slice(0, iz_maxheight))
    heights = time_height.isel(Time=0)['height'].values
            
    for i in time_height['Time']:
        if (i % skip_times == 0):
            idx = int(i / skip_times)
            profile = time_height.isel(Time=i)
            
            # min velocity
            mask1 = profile['windspeed'].isel(bottom_top = iz_HH).values > u_min

            # peak within range
            umax = profile['windspeed'].values.max(axis=0)
            i_z_umax = np.abs(profile['windspeed'].values - umax).argmin()
            z_umax = heights[i_z_umax]
            mask2a = z_umax > heights[0]
            mask2b = z_umax < heights[-1]
            mask2 = mask2a & mask2b

            # dropoff
            profileup = profile['windspeed'].isel(bottom_top = slice(i_z_umax, len(heights))).values
            diffprofileup = np.diff(profileup)
            min_idx = np.where(diffprofileup > 0.0)[0]
            if len(min_idx) == 0:
                i_z_locmin = -1
            else:
                i_z_locmin = min_idx[0]
            z_locmin = heights[i_z_locmin]
            u_locmin = profile['windspeed'].isel(bottom_top = i_z_locmin).values

            du = umax - u_locmin
            mask3a = du > dumin
            mask3b = du/umax > duminOverUjet
            mask3 = mask3a & mask3b

            isjet = mask1 & mask2 & mask3

            if ~isjet:
                z_umax = -999
                umax = -999
                
            is_jet[idx] = isjet
            jet_nose_height[idx] = z_umax
            jet_nose_speed[idx] = umax
    
    jet_id = xr.Dataset(
        data_vars = dict(is_jet=(["Time"], is_jet), jet_nose_height=(["Time"],jet_nose_height), jet_nose_speed=(["Time"],jet_nose_speed)),
        attrs=dict(description="Jet presence, nose height, and nose wind speed"))
    
    return jet_id


def findLLJevents_xr_new(time_height, max_height, u_min=3.0, dumin = 1.5, duminOverUjet = 0.1, N_height=65):
    heights = time_height.isel(Time=0)['height'].values
    iz_HH = (np.abs(heights - 140.0)).argmin()
    iz_maxheight = (np.abs(heights - max_height)).argmin()

    # downsample the dataa
    time_height = time_height.isel(bottom_top = slice(0, iz_maxheight))
    heights = time_height.isel(Time=0)['height']

    N_time = len(time_height['Time'])
    is_jet = np.zeros(N_time, dtype=bool)
    mask3 = np.zeros(N_time, dtype=bool)
    
    mask1 = time_height['windspeed'].isel(bottom_top = iz_HH) > u_min
    
    umax = time_height['windspeed'].max(axis=1)
    i_z_umax = np.abs(time_height['windspeed'] - umax).argmin(axis=1).values
    z_umax = heights[i_z_umax]

    umax_arr = umax.values
    z_umax_arr = z_umax.values

    mask2a = z_umax > heights[0]
    mask2b = z_umax < heights[-1]
    mask2 = mask2a & mask2b
    
    for i in time_height['Time']:
        profileup = time_height['windspeed'].isel(Time=i, bottom_top = slice(i_z_umax[i], len(heights)))
        diffprofileup = np.diff(profileup)
        min_idx = np.where(diffprofileup > 0.0)
        if len(min_idx[0]) == 0:
            i_z_locmin = -1
        else:
            i_z_locmin = min_idx[0][0] + i_z_umax[i]

        z_locmin = heights[i_z_locmin]
        u_locmin = time_height['windspeed'].isel(Time=i, bottom_top = i_z_locmin)

        du = umax[i] - u_locmin
        mask3a = du > dumin
        mask3b = du/umax[i] > duminOverUjet
        mask3[i] = mask3a & mask3b
        
    is_jet = mask1.values & mask2.values & mask3
    
    z_umax_arr[~is_jet] = -999
    umax_arr[~is_jet] = -999
    
    jet_id = xr.Dataset(
        data_vars = dict(is_jet=(["Time"], is_jet), jet_nose_height=(["Time"],z_umax_arr), jet_nose_speed=(["Time"],umax_arr)),
        attrs=dict(description="Jet presence, nose height, and nose wind speed"))
    
    return jet_id

def findLLJevents_xr_lidar(time_height, u_min=3.0, dumin = 1.5, duminOverUjet = 0.1, N_height=65):
    heights = time_height['height'].values
    iz_HH = (np.abs(heights - 140.0)).argmin()

    N_time = len(time_height['datetime'])
    is_jet = np.zeros(N_time, dtype=bool)
    mask3 = np.zeros(N_time, dtype=bool)
    
    mask1 = time_height['windspeed'].isel(z = iz_HH) > u_min
    
    umax = time_height['windspeed'].max(axis=1)
    i_z_umax = np.zeros_like(umax, dtype='int8')
    i_z_umax[~np.isnan(umax)] = np.abs((time_height['windspeed'] - umax)[~np.isnan(umax)]).argmin(axis=1).values
    z_umax = heights[i_z_umax]

    umax_arr = umax
    z_umax_arr = z_umax

    mask2a = z_umax > heights[0]
    mask2b = z_umax < heights[-1]
    mask2 = mask2a & mask2b
    
    for i in range(len(time_height['t'])):
        profileup = time_height['windspeed'].isel(t=i, z = slice(i_z_umax[i], None))
        diffprofileup = np.diff(profileup)
        min_idx = np.where(diffprofileup > 0.0)
        if len(min_idx[0]) == 0:
            i_z_locmin = -1
        else:
            i_z_locmin = min_idx[0][0] + i_z_umax[i]

        z_locmin = heights[i_z_locmin]
        u_locmin = time_height['windspeed'].isel(t=i, z = i_z_locmin)

        du = umax[i] - u_locmin
        mask3a = du > dumin
        mask3b = du/umax[i] > duminOverUjet
        mask3[i] = mask3a & mask3b
        
    is_jet = mask1 & mask2 & mask3
    
    z_umax_arr[~is_jet] = -999
    umax_arr[~is_jet] = -999
    
    jet_id = xr.Dataset(
        data_vars = dict(is_jet=(["Time"], is_jet.data), jet_nose_height=(["Time"],z_umax_arr.data), jet_nose_speed=(["Time"],umax_arr.data)),
        attrs=dict(description="Jet presence, nose height, and nose wind speed"))
    
    return jet_id


def findLLJevents_xr_meanhgt(time_height, max_height,
                         u_min=3.0,
                         dumin=1.5,
                         duminOverUjet=0.1,
                         N_height=65):
    heights = time_height.coords['height'].values
    iz_HH = (np.abs(heights - 140.0)).argmin()
    iz_maxheight = (np.abs(heights - max_height)).argmin()
    # downsample the dataa
    time_height = time_height.isel(height=slice(0,iz_maxheight))
    heights = time_height.coords['height']
    N_time = len(time_height.coords['datetime'])
    is_jet = np.zeros(N_time, dtype=bool)
    mask3 = np.zeros(N_time, dtype=bool)
    
    mask1 = time_height['windspeed'].isel(height=iz_HH) > u_min
    
    umax = time_height['windspeed'].max(axis=1)
    i_z_umax = time_height['windspeed'].argmax(axis=1)
    z_umax = heights[i_z_umax]
    umax_arr = umax.values
    z_umax_arr = z_umax.values
    mask2a = z_umax > heights[0]
    mask2b = z_umax < heights[-1]
    mask2 = mask2a & mask2b
    
    for i in range(time_height.dims['datetime']):
        profileup = time_height['windspeed'].isel(datetime=i, height=slice(i_z_umax[i], len(heights)))
        diffprofileup = np.diff(profileup)
        min_idx = np.where(diffprofileup > 0.0)
        if len(min_idx[0]) == 0:
            i_z_locmin = -1
        else:
            i_z_locmin = min_idx[0][0] + i_z_umax[i]
        z_locmin = heights[i_z_locmin]
        u_locmin = time_height['windspeed'].isel(datetime=i, height=i_z_locmin)
        du = umax[i] - u_locmin
        mask3a = du > dumin
        mask3b = du/umax[i] > duminOverUjet
        mask3[i] = mask3a & mask3b
        
    is_jet = mask1.values & mask2.values & mask3
    z_umax_arr[~is_jet] = -999
    umax_arr[~is_jet] = -999
    jet_id = xr.Dataset(
        data_vars = {
            'is_jet': ('datetime', is_jet),
            'jet_nose_height': ('datetime', z_umax_arr),
            'jet_nose_speed': ('datetime', umax_arr)
        },
        coords={'datetime': time_height.coords['datetime']},
        attrs=dict(description="Jet presence, nose height, and nose wind speed"))
    return jet_id