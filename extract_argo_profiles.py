import numpy as np
import matplotlib.pyplot as plt
from argopy import DataFetcher as ArgoDataFetcher 

lon_w = -97.5
lon_e = -82.5
lat_s = 18
lat_n = 30

argo_loader = ArgoDataFetcher()
ds0 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2010-01-01', '2012-12-31']).to_xarray()
ds0_profiles = ds0.argo.point2profile()
print('2012')

ds = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2012-01-01', '2014-12-31']).to_xarray()
ds_profiles = ds.argo.point2profile()
print('2014')

ds2 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2015-01-01', '2015-12-31']).to_xarray()
ds2_profiles = ds2.argo.point2profile()
print('2015')

ds3 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2016-01-01', '2016-12-31']).to_xarray()
ds3_profiles = ds3.argo.point2profile()
print('2016')

ds4 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2017-01-01', '2017-12-31']).to_xarray()
ds4_profiles = ds4.argo.point2profile()
print('2017')

ds5 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2018-01-01', '2018-12-31']).to_xarray()
ds5_profiles = ds5.argo.point2profile()
print('2018')

ds6 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2019-01-01', '2019-12-31']).to_xarray()
ds6_profiles = ds6.argo.point2profile()
print('2019')

ds7 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2020-01-01', '2020-12-31']).to_xarray()
ds7_profiles = ds7.argo.point2profile()
print('2020')

ds8 = argo_loader.region([lon_w, lon_e, lat_s, lat_n, 0, 1000, '2021-01-01', '2021-12-31']).to_xarray()
ds8_profiles = ds8.argo.point2profile()
print('2021')

ds_tot = xr.concat([ds0_profiles,ds_profiles,ds2_profiles,ds3_profiles,ds4_profiles,ds5_profiles,ds6_profiles,ds7_profiles,ds8_profiles],dim='N_PROF')