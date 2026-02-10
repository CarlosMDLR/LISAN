import os, glob
import numpy as np
import argparse
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import re 
from scipy.optimize import curve_fit
from matplotlib import rc
from matplotlib.path import Path
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
#import cblind as cb; cmap = plt.get_cmap("cb.extreme_rainbow_r")
import astropy.visualization as vis
from scipy import stats
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import sigma_clip
import itertools
from matplotlib.patches import Patch
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('xtick', labelsize=17)    # tamaño de los tick labels
plt.rc('ytick', labelsize=17)
from matplotlib.ticker import ScalarFormatter,FixedLocator

# Definición de la función de ley de potencias
def power_law(x, a, b):
    return a * (x ** b)

# Función para extender el PSF
def extend_psf_in_range(x_psf_1, y_psf_1, filter, x_min=1000, x_max=2000):
    """
    Filtra los datos de x en el rango [x_min, x_max], ajusta la ley de potencias a los últimos 1000 puntos
    y luego extrapola el ajuste hasta 6500 píxeles adicionales.
    
    Args:
        x_psf_1 (array): Array de valores de x.
        y_psf_1 (array): Array de valores de y correspondientes a x.
        filter (str): El filtro ("g" o "r") utilizado para seleccionar los datos.
        x_min (int): Límite inferior del rango de x. Default es 1000.
        x_max (int): Límite superior del rango de x. Default es 2000.
    
    Returns:
        x_cadena (array): Valores extendidos de x.
        y_cadena (array): Valores extendidos de y.
    """
    # Convertir a arrays numpy
    x_psf = np.array(x_psf_1)
    y_psf = np.array(y_psf_1)
    
    # Filtrar valores válidos (no NaN)
    valid = ~np.isnan(y_psf)
    x_psf = x_psf[valid]
    y_psf = y_psf[valid]
    
    # Filtrar por rango de x (entre 1000 y 2000)
    valid_range = (x_psf >= x_min) & (x_psf <= x_max)
    x_psf = x_psf[valid_range]
    y_psf = y_psf[valid_range]
    
    # Ajustar por ley de potencias a los últimos 1000 puntos del rango seleccionado
    params, _ = curve_fit(power_law, x_psf, y_psf, maxfev=500000)
    a1, b1 = params  # Parámetros de ajuste (a, b)
    
    # Extrapolar hasta 6500 píxeles adicionales
    r_extrapolado = np.arange(x_psf[-1] + 1, x_psf[-1] + 6500, 1)
    y_extrapolado = power_law(r_extrapolado, a1, b1)
    
    # Concatenar los valores originales con los extrapolados
    x_cadena = np.concatenate((np.array(x_psf_1), r_extrapolado))
    y_cadena = np.concatenate((np.array(y_psf_1), y_extrapolado))
    
    # Visualizar el ajuste
    plt.figure()
    plt.loglog(x_cadena, y_cadena, "k.", label=f'Ajuste: a={a1:.2f}, b={b1:.2f}')
    plt.loglog(r_extrapolado, y_extrapolado, "c-", label="Extrapolación")
    plt.loglog(x_psf[-1000:], power_law(x_psf[-1000:], a1, b1), "r--", label="Ajuste ley de potencias")
    plt.legend()
    plt.xlabel("x (píxeles)")
    plt.ylabel("y (intensidad)")
    plt.title(f"Ajuste por ley de potencias para filtro {filter}")
    plt.show()
    
    return x_cadena, y_cadena

def lorentzian(x, amplitude, center, width):
    return amplitude / (1 + ((x - center) / (0.5 * width))**2)

def ajuste_gaussiano(datos_raw):
    # Ajuste de la gaussiana a los datos
    datos_1=datos_raw
    datos_1=datos_1.flatten()
   
    datos_1 = datos_1[~np.isnan(datos_1)]
    datos = datos_1-np.min(datos_1)
    
    initial_params = [1,len(datos)/2, 50]
    x = np.arange(0,len(datos))
  
    # Ajuste de la curva
    params, covariance = curve_fit(lorentzian, x, datos,nan_policy='omit', p0=initial_params)

    # Obtén los parámetros ajustados
    amplitude, mean, stddev = params

    # Calcula la curva ajustada
    fit_curve = lorentzian(x, amplitude, mean, stddev)

    # plt.figure()
    # # Visualiza los resultados
    # plt.plot(x, datos, "ko")
    # plt.axvline(mean, ls="--",color="k")
    # plt.plot(x, fit_curve, 'r-', label='Ajuste Lorentziano')
    # plt.title('Ajuste Lorentziano al Histograma de Cuentas')
    # plt.xlabel('Pixeles',fontsize = 18)
    # plt.ylabel('Counts', fontsize = 18)

    # plt.legend()
    # plt.show()
    return(mean,stddev,fit_curve+np.min(datos_1))

def graph_interactive_center(data):
    
    # Plotting the graph
    fig, ax = plt.subplots()
    points=[]

    interval = vis.PercentileInterval(99.9)
    
    norm = vis.ImageNormalize(vmin=0, vmax=2000, stretch=vis.LogStretch(10000))
    img = ax.imshow(data, cmap="viridis", norm=norm, picker=True)  # Enable picker for the image
    ax.grid(False)
    
    line, = ax.plot([], [], marker='o', color='r', linestyle='-', linewidth=2)  # Line for selected points

    def onpick(event):
        if event.mouseevent is not None:
            xdata, ydata = event.mouseevent.xdata, event.mouseevent.ydata
            points.append((xdata, ydata))
            
            # Update the line with selected points
            line.set_xdata([point[0] for point in points])
            line.set_ydata([point[1] for point in points])
            
            # Redraw the figure
            fig.canvas.draw_idle()

            print('Selected points coordinates (x, y):', points)

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.set_size_inches(40, 40)
    plt.show()

    # Después de cerrar la figura, calcular los puntos dentro del área encerrada
    if len(points) == 1:
        try:
            center = points[0]
            
            square_center = data[(int(float(center[1]))-200):(int(float(center[1]))+201),\
                                    (int(float(center[0]))-200):(int(float(center[0]))+201)]

            # Definir parámetros para sigma clipping
            sigma = 5.0  # Número de desviaciones estándar para el clipping
            max_iter = 5  # Número máximo de iteraciones
            
            _, median_fila, std_fila = sigma_clipped_stats(square_center, sigma=sigma, axis=1, maxiters=max_iter)
            _, median_columna, std_columna = sigma_clipped_stats(square_center, sigma=sigma, axis=0, maxiters=max_iter)
            
            center_fila,std_center_fila, fit_fila = ajuste_gaussiano(median_fila)
            center_columna,std_center_columna, fit_columna = ajuste_gaussiano(median_columna)

            # Create a Figure, which doesn't have to be square.
            fig = plt.figure(layout='constrained')
            # Create the main axes, leaving 25% of the figure space at the top and on the
            # right to position marginals.
            ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
            circulo = Circle((center_columna,center_fila),50, edgecolor='b', facecolor='none')
            circulo2 = Circle((center_columna+5,center_fila),50, edgecolor='g', facecolor='none')
            ax.add_patch(circulo)
            ax.add_patch(circulo2)
            # The main axes' aspect can be fixed.
            ax.set(aspect=1)
            # Create marginal axes, which have 25% of the size of the main axes.  Note that
            # the inset axes are positioned *outside* (on the right and the top) of the
            # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
            # less than 0 would likewise specify positions on the left and the bottom of
            # the main axes.
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.25])
            ax_histy = ax.inset_axes([1.05, 0, 0.25, 1])
            # Draw the scatter plot and marginals.
            ax.imshow(square_center, cmap="viridis", norm=norm)
            ax.axhline(center_fila,ls="--",color="k") 
            ax.axvline(center_columna,ls="--",color="k")   
            ax_histx.plot(median_columna, "k-")
            ax_histx.plot(fit_columna, "r--")
            ax_histx.axvline(center_columna,ls="--",color="k")   

            ax_histy.plot(median_fila,np.arange(0,len(median_fila)), "k-")
            ax_histy.plot(fit_fila,np.arange(0,len(median_fila)), "r--")
            ax_histy.axhline(center_fila,ls="--",color="k")  
            ax_histx.set_xticks([])
            ax_histy.set_yticks([])

            ax.set_xlabel("x (px)",fontsize = 16)
            ax.set_ylabel("y (px)",fontsize = 16)

            ax_histx.set_ylabel("Column values",fontsize = 16)
            ax_histy.set_xlabel("Row values",fontsize = 16)
            plt.show()     
            
            new_center_row = float(center[1])-201+ center_fila
            new_center_colum = float(center[0])-201 +center_columna
            center_str = np.array([new_center_row,new_center_colum])

            return(center_str, np.array([float(center[1]),float(center[0])]))

        except Exception as e:
            print(e)
            center = points[0]
            return(np.array([float(center[1]),float(center[0])]), np.array([float(center[1]),float(center[0])]))
            

    else:
        print('Wrong number of points to define the center.')
        raise(Exception)

def profiles_reader(data_name):
    hdul = fits.open(data_name)
    x   = [hdul[1].data[i][0] for i in range(0, len(hdul[1].data))]
    y   = [hdul[1].data[i][1] for i in range(0, len(hdul[1].data))]
    std = [hdul[1].data[i][2] for i in range(0, len(hdul[1].data))]
    area= [hdul[1].data[i][3] for i in range(0, len(hdul[1].data))]
    rms = [hdul[1].data[i][4] for i in range(0, len(hdul[1].data))]
    return(np.array(x),np.array(y),np.array(std),np.array(area),np.array(rms))   

def save_subtracted_fits(original_fits_path, y_new):
    # Abrir el archivo FITS original
    hdul = fits.open(original_fits_path)
    
    # Modificar el valor de 'y' en el archivo
    for i in range(len(y_new)):
        hdul[1].data[i][1] = y_new[i]  # Reemplaza la columna 1 con el nuevo valor de 'y'
    
    # Crear el nuevo nombre del archivo añadiendo "substracted"
    new_fits_path = original_fits_path.replace(".fits", "_substracted.fits")
    
    # Guardar el archivo modificado
    hdul.writeto(new_fits_path, overwrite=True)
    
    # Cerrar el archivo original
    hdul.close()
    print(f"Archivo modificado guardado como: {new_fits_path}")
    return new_fits_path

def masks_mto(name,name_2,mto_path):
    print('\n'+60*'=')
    print('\n Making MTO mask of %s'%(name))
    print('\n'+60*'=')
    output = name.replace('.fits', '_mto.fits')
    par_output = name.replace('.fits', '_mto_par.csv')
    line_mto = (f'python3 {mto_path} {name} -verbosity=2 -move_factor=0.2 -par_out={par_output} -out={output}')
    os.system(line_mto)
    
    
    data = fits.open(output)
    
    objects = data[0].data
    number = stats.mode(objects[0:10,0:10],\
                                    axis=None,nan_policy='omit').mode
    objects[objects==number] = -1
    objects[objects != -1] = 1
    imagen = fits.getdata(name_2)
    imagen[objects!=-1]= np.nan
    hdu = fits.PrimaryHDU(imagen)
    hdu.writeto(name.replace(".fits","_masked.fits"), overwrite=True)
    os.system("rm "+par_output)
    #os.system("rm "+output)
    return(name.replace(".fits","_masked.fits")) 

def extended_outer_part_creator(filters,image_dir,width_crop_extended,mto_path):
    directorio_norm_radii = image_dir
    width_x = width_crop_extended.split(",")[0]
    width_y = width_crop_extended.split(",")[1]
    for filter in filters:

        archivos_fits_cut = [archivo for archivo in os.listdir(directorio_norm_radii) if re.search(f"_{filter}_"+'.*\.fits', archivo)]
        archivos_fits_cut = np.sort(archivos_fits_cut)

        for name in archivos_fits_cut:


            dir_image = os.path.join(directorio_norm_radii,name)
            hdu = fits.open(dir_image)
            image_psf_raw = hdu[1].data
            hdu.close()
            center,_=graph_interactive_center(image_psf_raw)
            center_str= str(np.round(center[1]))+","+str(np.round(center[0]))

            print("\n Star center with subpixel precision in pixels:")
            print(center_str)
            dir_crop= os.path.join(directorio_norm_radii,"Crops");os.makedirs(dir_crop, exist_ok=True)
            path_crop = os.path.join(dir_crop,name.replace(".fits", "_crop.fits"))
            os.system(f"astcrop --mode=img --section={str(int(np.round(center[1])))}:{str(int(np.round(center[1])+int(width_x)))},{str(int(np.round(center[0])))}:{str(int(np.round(center[0]))+int(width_y))} {dir_image} -o {path_crop}")

            ruta_masked=masks_mto(path_crop,path_crop,mto_path)

            hdu = fits.open(ruta_masked)
            image_masked = hdu[0].data
            hdu.close()

            o_dir_profile = ruta_masked.replace(".fits", "_radial_prof_temp.fits")
            o_dir_temp_star = ruta_masked.replace(".fits", "_star_temp.fits")
            os.system(f'astscript-radial-profile {ruta_masked} --sigmaclip=2,0.1 --measure=sigclip-mean --hdu=0 --center=0,0 --rmax=5000 -o {o_dir_profile}')
            os.system(f'asttable {o_dir_profile} --output=interval_temp.fits -c"arith RADIUS sorted-to-interval,SIGCLIP_MEAN" -o custom_table_temp.fits')
            os.system('echo "1 0 0 8 3500 0 0 1 0 1" \
                            | astmkprof --customtable=custom_table_temp.fits \
                                        --mergedsize='+str(np.shape(image_masked)[1])+","+str(np.shape(image_masked)[0])+' \
                                        --output='+o_dir_temp_star+' \
                                        --mcolnocustprof \
                                        --oversample=1 \
                                        --clearcanvas \
                                        --mode=img')
            

            o_dir_residual = path_crop.replace(".fits", "_residuals.fits")
            os.system(f'astarithmetic {path_crop} {o_dir_temp_star} - --hdu=1 --hdu=1 --output={o_dir_residual} --hdu=0')
            ruta_masked_residuals=masks_mto(o_dir_residual,path_crop,mto_path)

            os.system(f'astscript-radial-profile {ruta_masked_residuals} --quiet --center=0,0 --hdu=0 --measure=mean,std,area,semi-major --rmax=6001 -o {ruta_masked_residuals.replace(".fits","_profile.fits")}')            # breakpoint()
            os.system(f"rm custom_table_temp.fits {o_dir_profile} {o_dir_temp_star} {o_dir_residual} {ruta_masked}")

            x,y,std,area,rms= profiles_reader(ruta_masked_residuals.replace(".fits","_profile.fits")) 

            valid = ~np.isnan(y)

            # Seleccionar los últimos 50 puntos válidos (no NaN)
            y_valid = y[valid][-2000:]
            y_sky_clip = sigma_clip(y_valid, 2)
            y_sky_clip_2 = y_sky_clip.data[np.invert(y_sky_clip.mask)]
            sky = np.median(y_sky_clip_2)

            os.system(f"astarithmetic {ruta_masked_residuals} {sky} - --hdu=0 --output={ruta_masked_residuals.replace('.fits', '_sky_subtracted.fits')}")
            os.system(f"astscript-radial-profile {ruta_masked_residuals.replace('.fits', '_sky_subtracted.fits')} --quiet --center=0,0 --hdu=1 --measure=mean,std,area,semi-major --rmax=6001 -o {ruta_masked_residuals.replace('.fits','_profile_sky_subtracted.fits')}")
