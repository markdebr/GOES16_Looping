'''
Written by Mark De Bruin

Given a time range and start/end domain, program will automatically
download and subset GOES-16 images to produce a storm-centered
satellite loop. Right now, visible and infrared band presets are
in place.

I realize there are some inefficiencies in this script but I am not being
paid for this so unless they create a major problem, I do not care.

Contact mdebruin2@huskers.unl.edu with any questions about running this script.
'''

import datetime
import time
import xarray as xr
import requests
import netCDF4
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import cv2
import re

start = time.time()

def get_s3_keys(bucket, s3_client, prefix = ''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """
    kwargs = {'Bucket': bucket}
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

# Some weird function I made to make sure every part of the time range will get downloaded        
def hourRound(iterTime):
    iterTimestamp = pd.Timestamp(iterTime)
    rounded = iterTimestamp.round('60min').to_pydatetime()
    if rounded <= iterTime:
        iterTime = rounded + datetime.timedelta(hours=1)
        return iterTime
    else:
        return rounded

# This function takes from the GOES downloading function to instead contruct
# a dictionary of datetimes that are present in the AWS archive.        
def minuteList(band, startTime, endTime, product_name):
    # AWS S3 data explorer: https://noaa-goes16.s3.amazonaws.com/index.html
    # Initialize s3 client. 
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Control product within the function
    bucket_name = 'noaa-goes16'
    
    # Switch date is a day in 2019 when the AWS url convention changed slightly (possibly start of GOES-17)
    switchDate = datetime.datetime(2019, 2, 19, 9)
    
    roundEnd = hourRound(endTime)
    roundEnd = roundEnd - datetime.timedelta(hours=1)
    iterTime = startTime.replace(minute=0)
    finalList = {}
    finalList[iterTime] = []
    
    while iterTime != roundEnd:
        iterTime = iterTime + datetime.timedelta(hours=1)
        finalList[iterTime] = []
    
    for iterTime in list(finalList.keys()):
        year = iterTime.year
        day = iterTime.timetuple().tm_yday
        hour = iterTime.hour
        
        # Check to see if event happened after the switch date to concatenate the url properly
        if iterTime > switchDate:
            keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}'
                      )
        else:
            keys = get_s3_keys(bucket_name,
                           s3_client,
                           prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M3C{band:02.0f}'
                          )
            
        for filetime in keys:
            finalList[iterTime].append(int(filetime[61:63]))
            
    prevdif = 61
    minute = startTime.minute
    hour1list = list(finalList.values())[0]
    for goesmin in hour1list:
        dif = abs(minute-goesmin)
        if dif < prevdif:
            closestmin = goesmin
            prevdif = dif
    minIndexStart = hour1list.index(closestmin)
    hour1list = hour1list[minIndexStart::]
    hour1key = list(finalList.keys())[0]
    finalList[hour1key] = hour1list
    
    prevdif = 61
    hourElist = list(finalList.values())[-1]
    minute = endTime.minute
    for goesmin in hourElist:
            dif = abs(minute-goesmin)
            if dif < prevdif:
                closestmin = goesmin
                prevdif = dif
    minIndexEnd = hourElist.index(closestmin)
    hourElist = hourElist[:minIndexEnd:]
    hourEkey = list(finalList.keys())[-1]
    finalList[hourEkey] = hourElist
    
    return finalList

# This function controls every GOES image download    
def pullGOES(band, eventDate, product_name):
    # AWS S3 data explorer: https://noaa-goes16.s3.amazonaws.com/index.html
    # Initialize s3 client. 
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Control product within the function
    bucket_name = 'noaa-goes16'
    
    # Switch date is a day in 2019 when the AWS url convention changed slightly (possibly start of GOES-17)
    switchDate = datetime.datetime(2019, 2, 19, 9)
    year = eventDate.year
    day = eventDate.timetuple().tm_yday
    hour = eventDate.hour
    minute = eventDate.minute
    
    # Check to see if event happened after the switch date to concatenate the url properly
    if eventDate > switchDate:
        keys = get_s3_keys(bucket_name,
                   s3_client,
                   prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}'
                  )
    else:
        keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M3C{band:02.0f}'
                      )

    minuteList = []
    for filetime in keys:
        minuteList.append(int(filetime[61:63]))
    minIndex = minuteList.index(minute)
    
    # Check to see if event happened after the switch date to concatenate the url properly
    if eventDate > switchDate:
        keys = get_s3_keys(bucket_name,
                   s3_client,
                   prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{band:02.0f}')
    else:
        keys = get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day:03.0f}/{hour:02.0f}/OR_{product_name}-M3C{band:02.0f}')
            
    key = [key for key in keys][minIndex]
    resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    file_name = key.split('/')[-1].split('.')[0]
    nc4_ds = netCDF4.Dataset(file_name, memory = resp.content)
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    DS = xr.open_dataset(store)
    
    return DS

def calc_latlon(ds):
    # The math for this function was taken from 
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    
    # Extracts projection information and calculates lat/lons from x/y coordinates
    # Main function is to subset GOES images starting with lat/lons, efficiency!
    
    x = ds.x
    y = ds.y
    goes_imager_projection = ds.goes_imager_projection
    
    x,y = np.meshgrid(x,y)
    
    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi/180)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat
    
    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2
    
    r_s = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)
    
    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H-s_x)**2 +s_y**2))) * (180/np.pi)
    lon = (l_0 - np.arctan(s_y / (H-s_x))) * (180/np.pi)
    
    ds = ds.assign_coords({
        "lat":(["y","x"],lat),
        "lon":(["y","x"],lon)
    })
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    return ds

# Returns xy boundaries to subset the GOES images
def get_xy_from_latlon(ds, lats, lons):
    lat1, lat2 = lats
    lon1, lon2 = lons

    lat = ds.lat.data
    lon = ds.lon.data
    
    x = ds.x.data
    y = ds.y.data
    
    x,y = np.meshgrid(x,y)
    
    x = x[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]
    y = y[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)] 
    
    return ((min(x), max(x)), (min(y), max(y)))

# Calculates from lat/lon and 1st GOES image the needed x/y coordinate change from
# image to image. Since it needs to download a GOES image for this anyways, it saves
# it to the folder outside of the main image saving loop.
def panning(timeList, center1lat, center1lon, center2lat, center2lon, dlat, dlon, tempPath, band, colors, products, Vmax, Vmin):
    Nframes = sum(len(i) for i in timeList.values())
    startHour = list(timeList.keys())[0]
    startTime = startHour.replace(minute=timeList[startHour][0])
    endHour = list(timeList.keys())[-1]
    endTime = endHour.replace(minute=timeList[endHour][-1])
    
    startGOES = pullGOES(band, startTime, 'ABI-L2-CMIPC')
    endGOES = pullGOES(band, endTime, 'ABI-L2-CMIPC')
    
    latsS = (center1lat-dlat, center1lat+dlat)
    lonsS = (center1lon-dlon, center1lon+dlon)
    latsE = (center2lat-dlat, center2lat+dlat)
    lonsE = (center2lon-dlon, center2lon+dlon)
    startGOES = calc_latlon(startGOES)
    endGOES = calc_latlon(endGOES)
    ((x1s,x2s), (y1s, y2s)) = get_xy_from_latlon(startGOES, latsS, lonsS)
    ((x1e,x2e), (y1e, y2e)) = get_xy_from_latlon(endGOES, latsE, lonsE)
    dx = (x1e - x1s) / Nframes
    dy = (y1e - y1s) / Nframes
    
    startGOES = startGOES.sel(x=slice(x1s, x2s-.00001), y=slice(y2s-.00001, y1s)).CMI.to_dataframe()
    startimage = startGOES['CMI'].to_xarray()
    plt.imsave(os.path.join(tempPath, f"{1}.png"), startimage, format='png', cmap=colors, origin='lower', vmax=Vmax, vmin=Vmin)
    
    return dx, dy, (x1s,x2s), (y1s,y2s), startimage    

# Function to change the time interval of frames used from all frames to all/int frames
def intervalFilter(timeList, interval):
    keys = list(timeList.keys())
    newTimeList = {}
    count = interval
    for key in keys:
        valueList = []
        for value in timeList[key]:
            if count == interval:
                valueList.append(value)
                count = 1
            else:
                count+=1
        newTimeList[key] = valueList
    return newTimeList

# Occasionally images will be one pixel larger than the rest and won't be
# written into the video. This is a fast fix to crop based on the smallest image.
def crop(tempPath):
    smallestX, smallestY = float('inf'), float('inf')
    for file in os.listdir(tempPath):
        dims = cv2.imread(os.path.join(tempPath, file)).shape
        if dims[0] < smallestY:
            smallestY = dims[0]
        if dims[1] < smallestX:
            smallestX = dims[1]
    for file in os.listdir(tempPath):
        img = cv2.imread(os.path.join(tempPath, file))
        if img.shape[0] > smallestY or img.shape[1] > smallestX:
            crop = img[0:smallestY, 0:smallestX]  
            cv2.imwrite(os.path.join(tempPath, file), crop)
    cv2.destroyAllWindows()

# Manual data entry section
#######################
#######################
# Must set the time, nearest time is selected
startTime = '8-07-2023 17:00'
endTime = '8-08-2023 00:30'
#######################
#######################
# Image Settings 
interval = 2
frameRate = 10
# Band 2 for vis, 14 for infrared
band = 2
product = 'ABI-L2-CMIPC'
#######################
#######################
# Panning settings
center1lat = 39
center1lon = -81
center2lat = 39.9
center2lon = -74
dlat = 2.7
dlon = 3.1
#######################
#######################

def main():
    # Run everything here
    start = time.time()
    warnings.filterwarnings('ignore')
    
    if band == 14:
        colors = 'turbo_r'
        Vmax = 300
        Vmin = 195
    else:
        colors = 'gist_gray'
        Vmax = 1.1
        Vmin = -.1
        
    startObject = datetime.datetime.strptime(startTime, '%m-%d-%Y %H:%M')
    endObject = datetime.datetime.strptime(endTime, '%m-%d-%Y %H:%M')
    
    print('Gathering needed times...')
    timeList = minuteList(band, startObject, endObject, product)
    timeList = intervalFilter(timeList, interval)
    
    # Just to take out an empty keys in the dictionary
    for key in list(timeList.keys()):
        if len(timeList[key]) == 0:
            del timeList[key]
    
    frames = 'n'
    Nframes = sum(len(i) for i in timeList.values())
    while frames != 'y':
        frames = input(f'Total Frames: {Nframes}. Proceed? (y/n)')
    
    # Controls the temporary image directory that is created.
    # Will ensure the directory is empty before writing.
    tempPath = os.path.join(os.getcwd(), 'TempImages_GOES16')
    if os.path.isdir(tempPath) == False:
        os.mkdir(tempPath)
    for file in os.listdir(tempPath):
        os.remove(os.path.join(tempPath, file))
    
    # This is the main image saving loop
    iterTime = startObject
    count = 1
    keyList = list(timeList.keys())
    for iterTime in keyList:
        for minute in timeList[iterTime]:
            curTime = iterTime.replace(minute=minute)
            if count==1:
                print('Calibrating coordinate panning...')
                dx, dy, (b2x1, b2x2), (b2y1,b2y2), startimage = panning(timeList, center1lat, center1lon, center2lat, center2lon, dlat, dlon, tempPath, band, colors, product, Vmax, Vmin)
                count+=1
                continue
    
            b2x1+=dx
            b2x2+=dx
            b2y1+=dy
            b2y2+=dy
            band2 = pullGOES(band, curTime, product)
            band2 = band2.sel(x=slice(b2x1, b2x2), y=slice(b2y2, b2y1)).CMI.to_dataframe()
            band2image = band2['CMI'].to_xarray()
            plt.imsave(os.path.join(tempPath, f"{count}.png"), band2image, format='png', cmap=colors, origin='lower', vmax=Vmax, vmin=Vmin)
            print(f'Download Progress: {count}/{sum(len(i) for i in timeList.values())}')
            count+=1
        
    crop(tempPath)
    
    # Everything needed to combine the frames into an mp4 video
    image_folder = tempPath
    video_name = f'{startObject.year}{startObject.month}{startObject.day}.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame = cv2.imread(os.path.join(image_folder, images[1]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, frameRate, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    crop(tempPath)
    cv2.destroyAllWindows()
    video.release()
                    
    end = time.time()
    print("Execution time: ", (end-start) * 10**3, "ms")

if __name__ == '__main__':
    main()
