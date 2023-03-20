import sys
import numpy as np
import xarray as xr
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time 
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
sys.path.append('/Users/jakesteinberg/ECCOv4-py/ECCOv4-py')
import ecco_v4_py as ecco
import warnings
warnings.filterwarnings('ignore')

def seasonal_cycle(x,a,b,c,d,f):
    return a*np.sin((2*np.pi/365)*x+b) + c*np.sin((2*np.pi/(365/2))*x+d) + f 


# -- LOAD -- 
ttt = time.time()

filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_1993_2002.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_1996_2005.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_2000_2009.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_2004_2013.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_2008_2017.nc'

# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_thirddeg_1998_2007.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_thirddeg_2008_2017.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_thirddeg_2008_2013.nc'
# filename = '/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_thirddeg_2013_2018.nc'
x = xr.open_dataset(filename)

ecco_sha = x['sha']
ecco_obpa = x['obpa']
ecco_dep = x['dep']
new_grid_lon = x['i']
new_grid_lat = x['j']
lon_w = new_grid_lon[0] - (new_grid_lon[1]-new_grid_lon[0])/2
lon_e = new_grid_lon[-1] + (new_grid_lon[1]-new_grid_lon[0])/2
lat_s = new_grid_lat[0] - (new_grid_lat[1]-new_grid_lat[0])/2
lat_n = new_grid_lat[-1] + (new_grid_lat[1]-new_grid_lat[0])/2

# -- process -- 
ecco_zeta_rho_month = ecco_sha
ecco_obpa_month = ecco_obpa
# replace land with nans
ecco_zeta_rho_month = ecco_zeta_rho_month.where((ecco_zeta_rho_month > 0) | (ecco_zeta_rho_month < 0), other=np.nan)
ecco_obpa_month = ecco_obpa_month.where((ecco_zeta_rho_month > 0) | (ecco_zeta_rho_month < 0), other=np.nan)
# -- IF DESIRED -- 
# remove section(s) of the pacific 
# - 1 
polo = np.where(new_grid_lon < -75)[0]; pola = np.where(new_grid_lat < 8.5)[0]
ecco_zeta_rho_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
ecco_obpa_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
# - 2
polo = np.where(new_grid_lon < -84)[0]; pola = np.where(new_grid_lat < 13)[0]
ecco_zeta_rho_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
ecco_obpa_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
# - 3
polo = np.where(new_grid_lon < -87)[0]; pola = np.where(new_grid_lat < 17)[0]
ecco_zeta_rho_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
ecco_obpa_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
# - remove hudson bay
polo = np.where(new_grid_lon < -75)[0]; pola = np.where(new_grid_lat > 50)[0]
ecco_zeta_rho_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan
ecco_obpa_month[:,np.arange(pola[0],pola[-1]),np.arange(polo[0],polo[-1])] = np.nan

# -- filter by depth -- 
ecco_obpa_shelf_month = ecco_obpa_month.where((ecco_dep > 0) & (ecco_dep < 200), other=np.nan)
ecco_zeta_rho_deep_month = ecco_zeta_rho_month.where((ecco_dep > 1500), other=np.nan)

# -- collapse two horizontal dimensions to one -- 
# full field (over all depths)
ecco_zeta_rho_month_full = ecco_zeta_rho_month.stack(k=("j", "i")) # .unstack("z")
ecco_obpa_month_full = ecco_obpa_month.stack(k=("j", "i")) # .unstack("z")
# shelf/offshore
ecco_zeta_rho_month_0 = ecco_zeta_rho_deep_month.stack(k=("j", "i")) # .unstack("z")
ecco_obpa_shelf_month_0 = ecco_obpa_shelf_month.stack(k=("j", "i")) # .unstack("z")

# -- remove nan (land) timeseries -- 
# full field 
ecco_zeta_rho_month_full1 = ecco_zeta_rho_month_full[:,~np.isnan(ecco_zeta_rho_month_full[0,:].data)]
ecco_obpa_month_full1 = ecco_obpa_month_full[:,~np.isnan(ecco_obpa_month_full[0,:].data)]
ecco_zeta_rho_month_full_w = ecco_zeta_rho_month_full1.interpolate_na(dim='time')
ecco_obpa_month_full_w = ecco_obpa_month_full1.interpolate_na(dim='time')
# shelf/offshore
ecco_zeta_rho_month_1 = ecco_zeta_rho_month_0[:,~np.isnan(ecco_zeta_rho_month_0[0,:].data)]
ecco_obpa_shelf_month_1 = ecco_obpa_shelf_month_0[:,~np.isnan(ecco_obpa_shelf_month_0[0,:].data)]
# interpolate remaining nans (some locations have nans at times)
ecco_zeta_rho_month_w = ecco_zeta_rho_month_1.interpolate_na(dim='time')
ecco_obpa_shelf_month_w = ecco_obpa_shelf_month_1.interpolate_na(dim='time')

# -- remaining nans -- 
# full field 
stragglers = np.where(np.isnan(ecco_zeta_rho_month_full_w))
stragglers_obpa = np.where(np.isnan(ecco_obpa_month_full_w)) # NONE HERE
for i in range(len(stragglers[0])):
    ecco_zeta_rho_month_full_w[stragglers[0][i],stragglers[1][i]] = ecco_zeta_rho_month_full_w[stragglers[0][i],stragglers[1][i]-1] - 0.001
for i in range(len(stragglers_obpa[0])):
    ecco_obpa_month_full_w[stragglers_obpa[0][i],stragglers_obpa[1][i]] = ecco_obpa_month_full_w[stragglers_obpa[0][i],stragglers_obpa[1][i]-1] - 0.001
# shelf/offshore
stragglers = np.where(np.isnan(ecco_zeta_rho_month_w))
stragglers_obpa = np.where(np.isnan(ecco_obpa_shelf_month_w)) # NONE HERE
# -- at start of end of time series 
for i in range(len(stragglers[0])):
    ecco_zeta_rho_month_w[stragglers[0][i],stragglers[1][i]] = ecco_zeta_rho_month_w[stragglers[0][i],stragglers[1][i]-1] - 0.001
for i in range(len(stragglers_obpa[0])):
    ecco_obpa_shelf_month_w[stragglers_obpa[0][i],stragglers_obpa[1][i]] = ecco_obpa_shelf_month_w[stragglers_obpa[0][i],stragglers_obpa[1][i]-1] - 0.001
    
# -- check -- 
print(np.sum(np.isnan(ecco_zeta_rho_month_w.data)))
print(np.sum(np.isnan(ecco_obpa_shelf_month_w.data)))
# -- show shapes of matrices we're about to operate on -- 
print(np.shape(ecco_zeta_rho_month_w))
print(np.shape(ecco_obpa_shelf_month_w))
print(np.shape(ecco_zeta_rho_month_w)[1]*np.shape(ecco_obpa_shelf_month_w)[1])

# -- time array [datetimes] -- 
time_ord = np.nan*np.ones(np.shape(ecco_zeta_rho_month_w)[0])
for i in range(np.shape(ecco_zeta_rho_month_w)[0]):
    ts = (np.datetime64(str(ecco_zeta_rho_month_w[i,0].time.data)[0:10]) - np.datetime64('0000-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    time_ord[i] = ts/(60*60*24)

# -- remove seasonal cycle -- 
print('removing seasonal cycle')
# full field 
# STERIC HEIGHT
ecco_zeta_rho_month_full_w_wosea = ecco_zeta_rho_month_full_w.copy()
for i in range(np.shape(ecco_zeta_rho_month_full_w)[1]):
    # -- fit seasonal cycle over entire time series 
    fit = curve_fit(seasonal_cycle, time_ord - time_ord[0], ecco_zeta_rho_month_full_w[:,i])
    model = seasonal_cycle((time_ord - time_ord[0]),*fit[0])
    ecco_zeta_rho_month_full_w_wosea[:,i] = ecco_zeta_rho_month_full_w[:,i] - model  
# OBPA SHELF
ecco_obpa_month_full_w_wosea = ecco_obpa_month_full_w.copy()
for i in range(np.shape(ecco_obpa_month_full_w)[1]):
    # -- fit seasonal cycle over entire time series 
    fit = curve_fit(seasonal_cycle, time_ord - time_ord[0], ecco_obpa_month_full_w[:,i])
    model = seasonal_cycle((time_ord - time_ord[0]),*fit[0])
    ecco_obpa_month_full_w_wosea[:,i] = ecco_obpa_month_full_w[:,i] - model
# shelf/offshore
# STERIC HEIGHT
ecco_zeta_rho_month_w_wosea = ecco_zeta_rho_month_w.copy()
for i in range(np.shape(ecco_zeta_rho_month_w)[1]):
    # -- fit seasonal cycle over entire time series 
    fit = curve_fit(seasonal_cycle, time_ord - time_ord[0], ecco_zeta_rho_month_w[:,i])
    model = seasonal_cycle((time_ord - time_ord[0]),*fit[0])
    ecco_zeta_rho_month_w_wosea[:,i] = ecco_zeta_rho_month_w[:,i] - model
# OBPA SHELF
ecco_obpa_shelf_month_w_wosea = ecco_obpa_shelf_month_w.copy()
for i in range(np.shape(ecco_obpa_shelf_month_w)[1]):
    # -- fit seasonal cycle over entire time series 
    fit = curve_fit(seasonal_cycle, time_ord - time_ord[0], ecco_obpa_shelf_month_w[:,i])
    model = seasonal_cycle((time_ord - time_ord[0]),*fit[0])
    ecco_obpa_shelf_month_w_wosea[:,i] = ecco_obpa_shelf_month_w[:,i] - model

# -- COV -- 
print('svd')
# steric height & shelf bottom pressure 
# -- w/o seasonal cycle -- 
covm = np.matrix(ecco_zeta_rho_month_w_wosea/np.nanstd(ecco_zeta_rho_month_w_wosea,axis=0)).transpose()*\
    np.matrix(ecco_obpa_shelf_month_w_wosea/np.nanstd(ecco_obpa_shelf_month_w_wosea,axis=0))
# -- big long step
U,S,V = np.linalg.svd(covm)
# -- variance explained by each mode 
print(S[0]**2/np.sum(S**2))
print(S[1]**2/np.sum(S**2))
# -- EXPANSION COEFFICIENTS -- 
s1_ec = np.matrix(ecco_zeta_rho_month_w_wosea)*np.matrix(U)
s2_ec = np.matrix(ecco_obpa_shelf_month_w_wosea)*np.matrix(V)

# ------------------------------- 
print('correlation maps')
# full field 
# -- correlation maps on full steric and obpa fields (not selected by depth ranges)
re_corr1_full = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
re_corr2_full = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
for i in range(np.shape(ecco_zeta_rho_month_full_w_wosea)[1]): # loop over each location 
    kij = ecco_zeta_rho_month_full_w_wosea.k.data[i]
    loni = np.where(new_grid_lon == kij[1])[0]
    lati = np.where(new_grid_lat == kij[0])[0]
    # -- homogeneous correlation map -- (expansion coefficient of 1st mode of s1 w/ s1 values) -- 
    cor1 = np.corrcoef(np.squeeze(s1_ec[:,0]),ecco_zeta_rho_month_full_w_wosea[:,i])
    re_corr1_full[lati,loni] = cor1[0,1]
    # -- heterogeneous correlation map -- (expansion coefficient of 1st mode of s2 w/ s1 values) -- 
    cor2 = np.corrcoef(np.squeeze(s2_ec[:,0]),ecco_zeta_rho_month_full_w_wosea[:,i])
    re_corr2_full[lati,loni] = cor2[0,1]
    
re_corr3_full = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
re_corr4_full = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
for i in range(np.shape(ecco_obpa_month_full_w_wosea)[1]):
    kij = ecco_obpa_month_full_w_wosea.k.data[i]
    loni = np.where(new_grid_lon == kij[1])[0]
    lati = np.where(new_grid_lat == kij[0])[0]
    # -- homogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s2 w/ s2 values) -- 
    cor3 = np.corrcoef(np.squeeze(s2_ec[:,0]),ecco_obpa_month_full_w_wosea[:,i])
    re_corr3_full[lati,loni] = cor3[0,1]
    # --  heterogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s1 w/ s2 values) -- 
    cor4 = np.corrcoef(np.squeeze(s1_ec[:,0]),ecco_obpa_month_full_w_wosea[:,i])
    re_corr4_full[lati,loni] = cor4[0,1]
    
# shelf/offshore 
# -- correlation maps/analysis -- 
# S1 (variable 1 = steric height)
re_map1 = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
re_corr1 = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
re_map2 = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
re_corr2 = np.nan*np.ones(np.shape(ecco_zeta_rho_month[0,:,:]))
for i in range(len(U[:,0])):
    kij = ecco_zeta_rho_month_w.k.data[i]
    loni = np.where(new_grid_lon == kij[1])[0]
    lati = np.where(new_grid_lat == kij[0])[0]
    # -- homogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s1 w/ s1 values) -- 
    cor1 = np.corrcoef(np.squeeze(s1_ec[:,0]),ecco_zeta_rho_month_w_wosea[:,i])
    re_map1[lati,loni] = U[i,0]
    re_corr1[lati,loni] = cor1[0,1]
    re_map2[lati,loni] = U[i,1] # mode 2 of s1
    
    # -- heterogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s2 w/ s1 values) -- 
    cor2 = np.corrcoef(np.squeeze(s2_ec[:,0]),ecco_zeta_rho_month_w_wosea[:,i])
    re_corr2[lati,loni] = cor2[0,1]
    
re_map3 = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
re_corr3 = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
re_map4 = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
re_corr4 = np.nan*np.ones(np.shape(ecco_obpa_shelf_month[0,:,:]))
for i in range(len(V[:,0])):
    kij = ecco_obpa_shelf_month_w.k.data[i]
    loni = np.where(new_grid_lon == kij[1])[0]
    lati = np.where(new_grid_lat == kij[0])[0]
    # -- homogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s2 w/ s2 values) -- 
    cor3 = np.corrcoef(np.squeeze(s2_ec[:,0]),ecco_obpa_shelf_month_w_wosea[:,i])
    re_map3[lati,loni] = V[0,i] # V[i,0]
    re_corr3[lati,loni] = cor3[0,1]
    re_map4[lati,loni] = V[1,i] # V[i,1]
    
    # --  heterogeneous correlation map -- 
    # -- (expansion coefficient of 1st mode of s1 w/ s2 values) -- 
    cor4 = np.corrcoef(np.squeeze(s1_ec[:,0]),ecco_obpa_shelf_month_w_wosea[:,i])
    re_corr4[lati,loni] = cor4[0,1]


# save mca output for later plotting 
file_out = ('/Users/jakesteinberg/Documents/NASA_OSTST/analysis/eccov5_sha_obpa_halfdeg_' + filename[-12:-8] + '_' + filename[-7:-3] + '_mca.nc')
ds = xr.Dataset(
    data_vars=dict(
        sha_homcor_m1=(["j","i"], re_corr1_full),
        obpa_hetcor_m1=(["j","i"], re_corr4_full),
        dep=(["j","i"], ecco_dep.data),
        mode_frac_cov=(['m'], S[0:4]**2/np.sum(S**2)),
    ),
    coords=dict(
        j=(["j"], new_grid_lat.data),
        i=(["i"], new_grid_lon.data),
        m=(['m'], np.array([1,2,3,4]))
    ),
)
ds.to_netcdf(path=file_out)

# -- did stuff -- 
elapsed = time.time() - ttt
print(elapsed/60)

# ----------------
# -- FINAL PLOT -- 
xbs = [lon_w,lon_e]; ybs = [lat_s,lat_n]
crrcmp = plt.get_cmap('RdBu_r'); tgcmp = plt.get_cmap('viridis'); 
f, ax = plt.subplots(2,3,figsize=(20,10), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=0)})
# --- mode1 pattern ---
ax[0,0].pcolor(new_grid_lon,new_grid_lat,re_map1,transform=ccrs.PlateCarree(), vmin=np.nanmin(re_map1), vmax=np.nanmax(re_map1))
ax[0,0].set_title('S (Steric Height): m=1 of Cov. (frac. sq. cov. exp. = ' + str(np.round(S[0]**2/np.sum(S**2),2)) + ')') # S1
ax[0,0].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[0,0].coastlines()
gl = ax[0,0].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[0,0].set_facecolor('w');
ax[0,0].set_xlim(xbs); ax[0,0].set_ylim(ybs); 
# --- homogeneous correlation map ---
# cc = ax[0,1].pcolor(new_grid_lon,new_grid_lat,re_corr1,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
cc = ax[0,1].pcolor(new_grid_lon,new_grid_lat,re_corr1_full,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
ax[0,1].contour(new_grid_lon,new_grid_lat,re_corr1_full,levels=[0],transform=ccrs.PlateCarree())
ax[0,1].set_title('m=1 Homogeneous Correlation Map')    
ax[0,1].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[0,1].coastlines()
gl = ax[0,1].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[0,1].set_facecolor('w');
ax[0,1].set_xlim(xbs); ax[0,1].set_ylim(ybs); # ax[0,1].set_facecolor('#66C066')
# --- heterogeneous correlation map 
# cc = ax[0,2].pcolor(new_grid_lon,new_grid_lat,re_corr4,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
cc = ax[0,2].pcolor(new_grid_lon,new_grid_lat,re_corr4_full,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
ax[0,2].set_title('m=1 Heterogeneous Correlation Map')    
ax[0,2].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[0,2].coastlines()
gl = ax[0,2].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[0,2].set_facecolor('w');
ax[0,2].set_xlim(xbs); ax[0,2].set_ylim(ybs); # ax[0,2].set_facecolor('#66C066')
# -- repeat for second dataset -- 
ax[1,0].pcolor(new_grid_lon,new_grid_lat,re_map3,transform=ccrs.PlateCarree(), vmin=np.nanmin(re_map3), vmax=np.nanmax(re_map3))
ax[1,0].set_title('P (OBP): Mode 1 of Cov. (frac. sq. cov. exp. = ' + str(np.round(S[0]**2/np.sum(S**2),2)) + ')') # S2
ax[1,0].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[1,0].coastlines()
gl = ax[1,0].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[1,0].set_facecolor('w');
ax[1,0].set_xlim(xbs); ax[1,0].set_ylim(ybs);
# --- homogeneous correlation map ---
# cc = ax[1,1].pcolor(new_grid_lon,new_grid_lat,re_corr3,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
cc = ax[1,1].pcolor(new_grid_lon,new_grid_lat,re_corr3_full,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
ax[1,1].set_title('m=1 Homogeneous Correlation Map')    
ax[1,1].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[1,1].coastlines()
gl = ax[1,1].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[1,1].set_facecolor('w');
ax[1,1].set_xlim(xbs); ax[1,1].set_ylim(ybs);
# --- heterogeneous correlation map 
# cc = ax[1,2].pcolor(new_grid_lon,new_grid_lat,re_corr2,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
cc = ax[1,2].pcolor(new_grid_lon,new_grid_lat,re_corr2_full,vmin=-1,vmax=1,cmap=crrcmp,transform=ccrs.PlateCarree())
ax[1,2].set_title('m=1 Heterogeneous Correlation Map')    
ax[1,2].add_feature(cartopy.feature.LAND, color='#D2B48C',zorder=1); ax[1,2].coastlines()
gl = ax[1,2].gridlines(draw_labels=True); gl.xlabels_top = False; gl.ylabels_right = False; ax[1,2].set_facecolor('w');
ax[1,2].set_xlim(xbs); ax[1,2].set_ylim(ybs);

cbpos = [0.92, 0.3, 0.015, 0.4]; cax = f.add_axes(cbpos); 
cb = f.colorbar(cc, cax=cax, orientation='vertical', extend='both', label='correlation w mode1');
plt.show()
# f.savefig('/Users/jakesteinberg/Documents/NASA_OSTST/meetings/2023_03_02/corr_coeffs_half_deg_08_17.jpg', dpi=450)
