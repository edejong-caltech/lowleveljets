#!/usr/bin/env python
import os
import numpy
import pandas as pd
import requests

startdate = '2019-09-01'
enddate = '2021-10-01'
outdir = '../sfc'

#analysis = 'namussfc' # CONUS
analysis = 'usfntsfc' # fronts
prefix = 'https://www.wpc.ncep.noaa.gov/archives/sfc'

#------------------------------------------------------------------------------

datetimes = pd.date_range(startdate,enddate,freq='3h')

for yr in datetimes.year.unique():
    os.makedirs(f'{outdir}/{yr}', exist_ok=True)

imgurls = [f'{prefix}/{dt.year}/{analysis}{dt.strftime("%Y%m%d%H")}.gif' for dt in datetimes]

for dt,url in zip(datetimes,imgurls):
    fpath = f'{outdir}/{url[len(prefix)+1:]}'
    if not os.path.isfile(fpath):
        print('\rDownloading '+fpath,end='')
        img_data = requests.get(url).content
        with open(fpath, 'wb') as f:
            f.write(img_data)
print('')
