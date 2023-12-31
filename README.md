# GOES16_Looping

Script used for creating extended storm centered satellite video loops. There is a settings entry section where a start/end time, start/end lat/lon center point, degree width/height, video frame rate, GOES image interval, and band number is required to be entered to create the loop.

Currently, there are two "presets". Band 2 (red visible imagery) and Band 14 (classic infrared). Changing the band number will automatically adjust the colorbar.

Temporary image files are placed in a directory that the script creates in the current working directory called "TempImages_GOES16". You can open this directory during the download process to monitor what the video frames will look like.

For creating loops with a long download time, I recommend a couple of steps. Set the interval to a large number so that only a few total frames will be downloaded across the entire loop. Use these few frames in TempImages to adjust your time range and lat/lon center points to make your loop look perfect. When your settings are exactly as you want, set your interval lower to download to full number of frames you will need to create your video loop.

Depending on internet speeds, download time could take at least a few seconds per image. If there are any ideas to optimize this, let me know.

WARNING: Depending on the versions of packages in your environment, the images produced may be flipped initially. If this happens, find the two plt.imsave calls and change origin='lower' to origin='upper'.

WARNING: Things get weird if you are indexing points near the edge of the GOES-16 CONUS domain. To alleviate this, you can change the product from 'ABI-L2-CMIPC' to 'ABI-L2-CMIPF', which will instead download the full GOES image beyond the CONUS. The downside is that download times will be significantly longer.

If posting a loop to Twitter, give me a tag @mark_debruin .
