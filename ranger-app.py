from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
from datetime import datetime
from osgeo import gdal, gdalnumeric, ogr, osr
from PIL import Image, ImageDraw
import zipfile 
import json
import os, glob, re
import numpy as np
from skimage.transform import resize
from google.cloud import storage
import geopandas as gpd
import pandas as pd
import fiona
from geopandas import GeoSeries, GeoDataFrame
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

try:
    import asyncio
except ImportError:
	import trollius as asyncio

def upload_file(bucket, directory, file):
	
	blob = bucket.blob(directory + file)
	blob.upload_from_filename(directory + file)
	print(file)

def imageToArray(i):

    a=gdalnumeric.fromstring(i.tobytes(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a

def world2Pixel(geoMatrix, x, y):

  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = int((x - ulX) / xDist)
  line = int((ulY - y) / xDist)
  return (pixel, line)

def get_extent(raster):

	geoTransform = raster.GetGeoTransform()
	minx = geoTransform[0]
	miny = geoTransform[3]
	maxx = minx + geoTransform[1] * raster.RasterXSize
	maxy = miny + geoTransform[5] * raster.RasterYSize

	return [minx, miny, maxx, maxy]

def save_raster(directory, file_name, image, projection, transform):
	
	driver = gdal.GetDriverByName('GTiff')

	try:
		ndvi_data = driver.Create(directory + file_name + '.tif',image.shape[1], image.shape[0], image.shape[2], gdal.GDT_Float32)
		for i in range(0, image.shape[2]):
			ndvi_data.GetRasterBand(i+1).WriteArray(image[:,:,i])
	
	except:
		ndvi_data = driver.Create(directory + file_name + '.tif',image.shape[1], image.shape[0], 1, gdal.GDT_Float32)
		ndvi_data.GetRasterBand(1).WriteArray(image)

	ndvi_data.SetGeoTransform(transform)
	ndvi_data.SetProjection(projection)
	ndvi_data.FlushCache()
	ndvi_data=None

def mask_image(clip_rgb, mask):

	final_rgb = np.zeros((clip_rgb.shape[0], clip_rgb.shape[1], 3))
	try:
		for i in range(0,clip_rgb.shape[2]):
			aux = np.multiply(clip_rgb[:,:,i], mask)
			aux[mask ==False] = np.nan
			final_rgb[:,:,i] = aux

	except:
		clip_rgb = np.multiply(clip_rgb, mask)
		clip_rgb[mask == False] = np.nan
		final_rgb = clip_rgb

	return final_rgb

def clip(ndvi, rgb, ngb, projection, geoTrans, directory_ndvi, directory_rgb, directory_ngb, file_name, codigo, bucket):

    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open('shape_file/predio.shp')
    lyr = shapef.GetLayer()
    poly = lyr.GetNextFeature()
    
    # Convert the layer extent to image pixel coordinates
    minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)

     # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    print('El predio codigo: ', codigo, ' tiene un extent de: ', minX, maxX, minY, maxY)

    clip_ndvi = ndvi[ulY:lrY, ulX:lrX]
    clip_rgb = rgb[ulY:lrY, ulX:lrX, :]
    clip_ngb = ngb[ulY:lrY, ulX:lrX, :]

    valid_pixels = np.count_nonzero(~np.isnan(clip_ndvi))
    total_pixels = pxWidth*pxHeight
    perc_pixels = valid_pixels/total_pixels
    save_bool = False

    print('El predio, codigo: ', codigo, ' tiene un porcentaje de: ', perc_pixels)

    if perc_pixels >= 0.5:

    	print('Se va a procesar, clipear y guardar.')
    	
    	# Create a new geomatrix for the image
    	geoTrans = list(geoTrans)
    	geoTrans[0] = minX
    	geoTrans[3] = maxY

    	# Map points to pixels for drawing the
    	points = []
    	pixels = []
    	geom = poly.GetGeometryRef()

    	mask = np.zeros((pxHeight, pxWidth), dtype=bool)

    	for g in range(geom.GetGeometryCount()):

        	pts = geom.GetGeometryRef(g)
        
        	if pts.GetGeometryName() == 'POLYGON':
        		pts = pts.GetGeometryRef(0)
        
        	points = []
        	pixels = []

        	for p in range(pts.GetPointCount()):
          		points.append((pts.GetX(p), pts.GetY(p)))
        
        
        	for p in points:
          		pixels.append(world2Pixel(geoTrans, p[0], p[1]))
        
        	rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
        	rasterize = ImageDraw.Draw(rasterPoly)
        	rasterize.polygon(pixels, 0)
        	aux = imageToArray(rasterPoly)
        	aux = ~np.array(aux, dtype=bool)
        	mask = mask | aux

    	mask_ndvi = mask_image(clip_ndvi, mask)
    	mask_rgb = mask_image(clip_rgb, mask)
    	mask_ngb = mask_image(clip_ngb, mask)

    	save_raster(directory_ndvi, file_name, mask_ndvi, projection, geoTrans)
    	upload_file(bucket, directory_ndvi, file_name+'.tif')
    	save_raster(directory_rgb, file_name, mask_rgb, projection, geoTrans)
    	upload_file(bucket, directory_rgb, file_name+'.tif')
    	save_raster(directory_ngb, file_name, mask_ngb, projection, geoTrans)
    	upload_file(bucket, directory_ngb, file_name+'.tif')

    
def ndvi_calculation(sentinel_dir, file, directory_ndvi, bucket):

	channel = ['B02_10m.jp2','B03_10m.jp2','B04_10m.jp2','B08_10m.jp2','SCL_20m.jp2']
	bands = []
	
	zfile = zipfile.ZipFile(sentinel_dir + file)
			
	for finfo in zfile.namelist():
		for ch in channel:
			file_name = ch
			
			if file_name in finfo:
				bands.append(finfo)
				
	bands.sort()

	blue = gdal.Open('/vsizip/' + sentinel_dir + file + '/' + str(bands[0])).ReadAsArray().astype(np.float)
	green = gdal.Open('/vsizip/' + sentinel_dir + file + '/' + str(bands[1])).ReadAsArray().astype(np.float)
	red_band = gdal.Open('/vsizip/' + sentinel_dir + file + '/' + str(bands[2]))
	nir = gdal.Open('/vsizip/' + sentinel_dir + file + '/' + str(bands[3])).ReadAsArray().astype(np.float)
	filtro = gdal.Open('/vsizip/' + sentinel_dir + file + '/' + str(bands[4])).ReadAsArray().astype(np.float)
	red = red_band.ReadAsArray().astype(np.float)

	ngb = [nir, green, blue]
	rgb = [red, green, blue]

	filtro = resize(filtro, red.shape, anti_aliasing=True)

	transform = red_band.GetGeoTransform()
	projection = red_band.GetProjection()
	
	ndvi = ((nir - red)/(nir + red))
	ngb = np.zeros((ndvi.shape[1], ndvi.shape[0],3))
	ngb[:,:,0] = nir
	ngb[:,:,1] = green
	ngb[:,:,2] = blue

	rgb = np.zeros((ndvi.shape[1], ndvi.shape[0],3))
	rgb[:,:,0] = red
	rgb[:,:,1] = green
	rgb[:,:,2] = blue

	ndvi[filtro < 2 ] = np.nan
	ndvi[filtro == 3 ] = np.nan
	ndvi[filtro > 7 ] = np.nan

	file_name = file[:-4]

	save_raster(directory_ndvi, file_name, ndvi, projection, transform)
	upload_file(bucket, directory_ndvi, file_name+'.tif')
	os.remove(directory_ndvi + file_name + '.tif')
			
	return ndvi, rgb, ngb, transform, projection

def ndvi_clipping(sentinel_dir, file_name, tile, date, directory_tiles, df_predios, directory_ndvi, directory_rgb, directory_ngb, bucket):	
	
	file_zip, file_tif = file_name + '.zip', file_name + '.tif'
	ndvi, rgb, ngb, transform, projection = ndvi_calculation(sentinel_dir, file_zip, directory_tiles, bucket)

	shp = df_predios

	for index, row in shp.iterrows(): 
		codigo = row["CODIGO"]
		aux = shp[shp['CODIGO']==codigo]
		codigo_epsg = '327'+str(tile[:2])
		aux = aux.to_crs(epsg=int(codigo_epsg))
		aux.to_file('shape_file/predio.shp')
		file_name = codigo+'_'+date
		try:
			clip(ndvi, rgb, ngb, projection, transform, directory_ndvi, directory_rgb, directory_ngb, file_name, codigo, bucket)
		except:
			pass
		#upload_file(bucket_predios, cloud_directory, directory_predios, file_name+'.tif')
			

def tiles_from_list(csv_file, shape_file):
    
    tiles_list = []
    predios_list = []

    df_predios = pd.read_csv(csv_file)

    unique_predios = df_predios['CODIGO'].unique().tolist()
    unique_predios = map(str, unique_predios)
    unique_predios = list(unique_predios)
        
    predios_chile = gpd.read_file(shape_file)
    predios_chile=predios_chile[predios_chile['CODIGO'].isin(unique_predios)]
    #print(predios_chile)
    predios_chile_coordenadas = gpd.read_file(shape_file)
    predios_chile_coordenadas=predios_chile_coordenadas[predios_chile_coordenadas['CODIGO'].isin(unique_predios)]
    
    predios_chile['geometry'] = predios_chile['geometry'].to_crs(epsg=4326)

    union_predios = predios_chile.unary_union
    envelope = union_predios.envelope
    g = GeoSeries([envelope])
    json_envelope = g.to_json()
    json_file = json.loads(json_envelope)

    return predios_chile_coordenadas, json_file

def gui_dialogo():

	root = tkinter.Tk()
	root.withdraw()
	currdir = os.getcwd()
	tempdir = filedialog.askopenfilename(parent=root, initialdir=currdir, title='Porfavor selecciona el csv con los predios')

	return tempdir

def check_folder(folder_list):
	
	for folder_path in folder_list:
	    if not os.path.exists(folder_path):
	        os.makedirs(folder_path)

def diff_ndvi_calculation(predios_list, predios_directory, mask_directory, dif_directory):

	ndvi_files = os.listdir(predios_directory)

	for predio in predios_list:
		
		images_list = []

		for file in ndvi_files:

			if file.startswith(str(predio)):
				images_list.append(file)

		
		if images_list:
			images_list.sort()

			raster_actual = gdal.Open(predios_directory + str(images_list[-1]))
			raster_pasada = gdal.Open(predios_directory + str(images_list[-2]))
			
			transform = raster_actual.GetGeoTransform()
			projection = raster_pasada.GetProjection()
		
			img_actual = raster_actual.ReadAsArray().astype(np.float)
			img_pasada = raster_pasada.ReadAsArray().astype(np.float)

			diferencia = img_actual - img_pasada
			diferencia[diferencia > 0 ] = np.nan

			limite = np.mean(diferencia)-2*np.std(diferencia)
		
			mask_diferencia = diferencia < limite

			mask_name = images_list[-1][:-4]+'_perdidas.tif'
			dif_name = images_list[-1][:-4]+'_diferencia.tif'

			save_raster(mask_directory, mask_name, mask_diferencia, projection, transform)
			save_raster(dif_directory, dif_name, diferencia, projection, transform)

def download_sentinel(current_date, past_date, json, ndvi_tiles, df_predios, ndvi_dir, rgb_dir, ngb_dir, sentinel_dir):

	# Variables datos
	geojson_file = 'Tiles-chile.geojson'
	cloud_directory = ''

	#Storage access
	client = storage.Client.from_service_account_json('ADL-forestal-segmentation-7dc429779824.json')
	bucket = client.get_bucket('ranger-app')

	# Sentinel API
	api = SentinelAPI('matias-arauco', 'arauco2019','https://scihub.copernicus.eu/apihub/')

	footprint = geojson_to_wkt(json)

	products = api.query(footprint, date=(past_date, current_date), platformname='Sentinel-2', cloudcoverpercentage=(0,30))

	products_df = api.to_dataframe(products)

	products_df_sorted = products_df.loc[products_df['tileid'].isnull()]

	products_df_sorted = products_df_sorted.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])

	print(products_df_sorted)
	
	index = products_df_sorted.index
	
	ite = 0
	
	for i in index:

		file_name = products_df_sorted['title'][ite][:] 
		year, month, day = products_df_sorted['title'][ite][11:15], products_df_sorted['title'][ite][15:17], products_df_sorted['title'][ite][17:19] 
		tile = products_df_sorted['title'][ite][39:44]
		date = year+'-'+month+'-'+day
		print('Descargando el Tile: ', tile, ', con fecha: ', date)
		api.download(i, directory_path = sentinel_dir)
		upload_file(bucket, sentinel_dir, file_name +'.zip')
		ndvi_clipping(sentinel_dir, file_name, tile, date, ndvi_tiles, df_predios, ndvi_dir, rgb_dir, ngb_dir, bucket)
		os.remove(sentinel_dir + file_name +'.zip')
		ite = ite+1

if __name__ == '__main__':

	#path_csv = gui_dialogo()
	path_csv = 'CODIGO_PLANT_2018.csv'

	ndvi_tiles = 'ndvi_tiles/'
	robos_dir = 'perdida_predios/'
	dif_dir ='diferencia_predios/'
	ndvi_dir = 'ndvi_predios/'
	rgb_dir = 'rgb_predios/'
	ngb_dir = 'ngb_predios/'
	shp_file = 'shape_file/'
	shape_path = 'predios-chile.shp'
	sentinel_dir = 'sentinel_files/'
	
	## Check if the folder exist, otherwise they are created in the current folder
	check_folder([ndvi_tiles, robos_dir, dif_dir, ndvi_dir, rgb_dir, ngb_dir, shp_file, sentinel_dir])

	df_predios, json = tiles_from_list(path_csv, shape_path)

	download_sentinel('NOW', 'NOW-10DAYS', json, ndvi_tiles, df_predios, ndvi_dir, rgb_dir, ngb_dir, sentinel_dir)
	
	scheduler = AsyncIOScheduler()
	scheduler.add_job(download_sentinel, 'interval', ['NOW', 'NOW-10DAYS', json, ndvi_tiles, df_predios, ndvi_dir, rgb_dir, ngb_dir, sentinel_dir], days=10)
	scheduler.start()
	print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

	# Execution will block here until Ctrl+C (Ctrl+Break on Windows) is pressed.
	try:
		asyncio.get_event_loop().run_forever()
	except (KeyboardInterrupt, SystemExit):
		pass
	
	#download_sentinel('NOW-8DAYS', 'NOW-15DAYS', tiles_list, predios_dir, df_predios, ndvi_dir)
	#diff_ndvi_calculation(predios_list, predios_dir, robos_dir, dif_dir)
	
