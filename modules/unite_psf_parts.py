import os, glob
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import re 
import shutil

import matplotlib.ticker as ticker
#import cblind as cb; cmap = plt.get_cmap("cb.extreme_rainbow_r")
import astropy.visualization as vis
import itertools
import warnings
import subprocess
import tempfile
from astropy.io import fits

from scipy.optimize import curve_fit
from matplotlib import rc
from scipy.interpolate import interp1d
from matplotlib.path import Path
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import stats
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from matplotlib.patches import Patch
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('xtick', labelsize=17)    # tamaño de los tick labels
plt.rc('ytick', labelsize=17)
from matplotlib.ticker import ScalarFormatter,FixedLocator
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.restoration import inpaint

ast_psf_unite="./modules/psf-unite-remade.sh"

          
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
    
    x_psf_sky=x_psf;y_psf_sky = y_psf# local_sky_subs(x_psf,y_psf)
    # Filtrar valores válidos (no NaN)
    valid = ~np.isnan(y_psf_sky)
    x_psf = x_psf_sky[valid]
    y_psf = y_psf_sky[valid]
    
    # Filtrar por rango de x (entre 1000 y 2000)
    valid_range = (x_psf >= x_min) & (x_psf <= x_max)
    x_psf = x_psf[valid_range]
    y_psf = y_psf[valid_range]
    
    # Ajustar por ley de potencias a los últimos 1000 puntos del rango seleccionado
    params, _ = curve_fit(power_law, x_psf, y_psf, maxfev=500000)
    a1, b1 = params  # Parámetros de ajuste (a, b)

    # Extrapolar hasta 6500 píxeles adicionales
    r_extrapolado = np.arange(x_psf[0] + 1, x_psf[-1] + 6500, 1)
    y_extrapolado = power_law(r_extrapolado, a1, b1)
    
    # Concatenar los valores originales con los extrapolados
    x_cadena = np.concatenate((np.array(x_psf_sky[:x_min]), r_extrapolado))
    y_cadena = np.concatenate((np.array(y_psf_sky[:x_min]), y_extrapolado))
    
    plt.figure()
    plt.loglog(np.array(x_psf_1), np.array(y_psf_1), "g.", label="Datos originales")
    plt.loglog(np.array(x_psf_sky), np.array(y_psf_sky), "c.", label="Cielo sustraido")
    plt.loglog(x_cadena, y_cadena, "k.", label=f'Ajuste: a={a1:.2f}, b={b1:.2f}')
    plt.loglog(r_extrapolado, y_extrapolado, "c-", label="Extrapolación")
    plt.loglog(x_psf[-1000:], power_law(x_psf[-1000:], a1, b1), "r--", label="Ajuste ley de potencias")


    plt.legend()
    plt.xlabel("x (píxeles)")
    plt.ylabel("y (intensidad)")
    plt.title(f"Ajuste por ley de potencias para filtro {filter}")
    plt.show()
   
    return x_cadena, y_cadena

def r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,l_lim,l_max):
    ginner = np.gradient(y_inner)
    ginter = np.gradient(y_outer)

    linner = len(ginner)
    linter = len(ginter)
    lmin = np.min((linner, linter))


    sign_gradient = ginner[:lmin] * ginter[:lmin]

    dif_gradient = np.abs(ginner[:lmin] - ginter[:lmin])
    dif_gradient[sign_gradient < 0.0] = np.nan   
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        ## PROFILES SIGNAL TO NOISE
        dif_sn = np.abs((y_outer/(rms_outer/np.sqrt(area_outer))) - (y_inner/(rms_inner/np.sqrt(area_inner))))

    dif = dif_sn[:lmin] * dif_gradient[:lmin]   

    index= (np.where(dif[l_lim:l_max] == np.nanmin(dif[l_lim:l_max]))[0])[0]+l_lim
    return(index)

def fc_c_calc(index,pouter,pinner):
    
    # c1 = pinner[index-1]
    # c2 = pinner[index+1]
    # w1 = pouter[index-1]
    # w2 = pouter[index+1]
    
    #fc = (w2-w1) / (c2-c1)
    #c = w1 - c1 * fc
    c1 = pinner[index]
    w1 = pouter[index]
    fc = w1/c1
    c = w1 -c1*fc
    
    return(fc,c)

def power_law(x, a, b):
    return a*(x**b)

def profiles_reader(data_name):
    hdul = fits.open(data_name)
    x   = [hdul[1].data[i][0] for i in range(0, len(hdul[1].data))]
    y   = [hdul[1].data[i][1] for i in range(0, len(hdul[1].data))]
    std = [hdul[1].data[i][2] for i in range(0, len(hdul[1].data))]
    area= [hdul[1].data[i][3] for i in range(0, len(hdul[1].data))]
    rms = [hdul[1].data[i][4] for i in range(0, len(hdul[1].data))]
    return(np.array(x),np.array(y),np.array(std),np.array(area),np.array(rms))

def process_subdirectories(ruta_completa, subdirs):
    # Listar todos los subdirectorios en el directorio principal
    files_dict = {}
    for dir_name in os.listdir(ruta_completa):
        # Verificar si el nombre del directorio contiene uno de los subdirectorios de interés
        if any(subdir in dir_name for subdir in subdirs):
            # Crear la ruta completa al subdirectorio
            subdir_path = os.path.join(ruta_completa, dir_name)
            
            # Verificar que sea un directorio
            if os.path.isdir(subdir_path):
                # Lista los archivos dentro del subdirectorio
                files_in_subdir = os.listdir(subdir_path)
                
                # Filtrar archivos que empiezan con 'profile' o 'stack'
                profile_file_1 = next((f for f in files_in_subdir if f.startswith('profile')), None)
                stack_file_1 = next((f for f in files_in_subdir if f.startswith('stack')), None)
                
                profile_file =os.path.join(subdir_path, profile_file_1)
                stack_file = os.path.join(subdir_path, stack_file_1)

                x,y,std,area,rms=profiles_reader(profile_file)

                valid = ~np.isnan(y)

                # Seleccionar los últimos 50 puntos válidos (no NaN)
                y_valid = y[valid][-100:]
                y_sky_clip = sigma_clip(y_valid, 2)
                y_sky_clip_2 = y_sky_clip.data[np.invert(y_sky_clip.mask)]
                sky = np.median(y_sky_clip_2)
                y=y-sky
                stack_data = fits.getdata(stack_file)
                # Guardar la información en el diccionario
                files_dict[dir_name] = {
                    "x": x,
                    "y": y ,
                    "std": std,
                    "area": area ,
                    "rms": rms,
                    "stack": stack_data
                }
    return files_dict

def calculate_junction_indices(files_dict, name_subdirs, index_ranges,bol):
    index_dict = {}
    for index in range(0, len(name_subdirs) - 1):
        inner_most_name = name_subdirs[index]
        outer_most_name = name_subdirs[index + 1]

        ranges = [int(i) for i in index_ranges[index].split(",")]
        x_inner = files_dict[inner_most_name]["x"]
        x_outer = files_dict[outer_most_name]["x"]
        y_inner = files_dict[inner_most_name]["y"]
        y_outer = files_dict[outer_most_name]["y"]
        area_inner = files_dict[inner_most_name]["area"]
        area_outer = files_dict[outer_most_name]["area"]
        rms_inner = files_dict[inner_most_name]["rms"]
        rms_outer = files_dict[outer_most_name]["rms"]

        index = r_junc_calc(y_inner, y_outer, rms_inner, rms_outer, area_inner, area_outer, ranges[0], ranges[1])

        # if bol:
        #     plt.figure()
        #     plt.loglog(x_outer, y_outer, "k.",label = "Outer")
        #     plt.loglog(x_inner,y_inner, "r.", label="Inner")        
        #     plt.axvline(index)
        #     plt.show()
        inner_parts = inner_most_name.split('_')
        outer_parts = outer_most_name.split('_')

        # Crear el nuevo nombre
        index_name = '_'.join([inner_parts[0], inner_parts[1], outer_parts[1], inner_parts[2]])
        index_dict[index_name] = index
    
    return index_dict


def calibrate_layers(files_dict_ordenado, index_outer_dict, bool_outer,inner_keys,filter):
    
    if bool_outer:
        outer_keys = list(files_dict_ordenado.keys())

        results = {}

        y_calibrated = None
        for i in range(len(outer_keys) - 1, 0, -1):
            outer_key_current = outer_keys[i]  
            outer_key_previous = outer_keys[i - 1]  

            if len(list(index_outer_dict.keys()))==1:
                index_key = list(index_outer_dict.keys())[0]
            else:
                index_key = f"Outer_{i-1}_{i}_{filter}"  

            if y_calibrated is None:
                x_current = files_dict_ordenado[outer_key_current]["x"]
                y_current = files_dict_ordenado[outer_key_current]["y"]
                stack_current = files_dict_ordenado[outer_key_current]["stack"]
                results[outer_key_current] = {
                    'x':x_current,
                    'y_calibrated': y_current,
                    'stack_calibrated': stack_current
                }
            else:
                y_current = y_calibrated  # En iteraciones posteriores usamos el valor calibrado anterior

            x_previous = files_dict_ordenado[outer_key_previous]["x"]
            y_previous = files_dict_ordenado[outer_key_previous]["y"]
            stack_previous = files_dict_ordenado[outer_key_previous]["stack"]
            
            fc_outer, c_outer = fc_c_calc(index_outer_dict[index_key], y_current, y_previous)
            
            y_calibrated = (y_previous * fc_outer) + c_outer
            stack_calibrated =(stack_previous*fc_outer) + c_outer
            results[outer_key_previous] = {
                'x':x_previous,
                'y_calibrated': y_calibrated,
                'stack_calibrated': stack_calibrated
            }
        return results
    else:
        outer_keys = list(files_dict_ordenado.keys())
        name_gal = outer_keys[0].split("_")[0]
        results = {}

        y_calibrated = None
        for i in range(len(outer_keys) - 1, 0, -1):
            outer_key_current = outer_keys[i]  
            outer_key_previous = outer_keys[i - 1]  

            index_key = f"{name_gal}_{inner_keys[i-1]}_{inner_keys[i]}_{filter}"  

            if y_calibrated is None:
                x_current = files_dict_ordenado[outer_key_current]["x"]
                y_current = files_dict_ordenado[outer_key_current]["y"]
                stack_current = files_dict_ordenado[outer_key_current]["stack"]
                results[outer_key_current] = {
                    'x':x_current,
                    'y_calibrated': y_current,
                    'stack_calibrated': stack_current
                }
            else:
                y_current = y_calibrated  # En iteraciones posteriores usamos el valor calibrado anterior
            x_previous = files_dict_ordenado[outer_key_previous]["x"]
            y_previous = files_dict_ordenado[outer_key_previous]["y"]
            stack_previous = files_dict_ordenado[outer_key_previous]["stack"]
            
            fc_outer, c_outer = fc_c_calc(index_outer_dict[index_key], y_current, y_previous)

            y_calibrated = (y_previous * fc_outer) + c_outer
            stack_calibrated =(stack_previous*fc_outer) + c_outer
            results[outer_key_previous] = {
                'x':x_previous,
                'y_calibrated': y_calibrated,
                'stack_calibrated': stack_calibrated
            }
        return results        

def unite_psf_with_gnuastro(results, filter,index_outer_dict, bool_outer,inner_keys,name_gal,output_filename="psf.fits",inner_dir=None, scale=1):

    if bool_outer:
        # Crear una lista para los archivos FITS temporales
        fits_files = []

        try:
            # Crear archivos FITS temporales para cada stack_calibrated
            for key, data in results.items():
                # Crear un archivo FITS temporal
                temp_fits = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                fits_files.append(temp_fits.name)
                
                # Guardar la matriz 'stack_calibrated' en un archivo FITS temporal
                hdu = fits.PrimaryHDU(data['stack_calibrated'])
                hdu.writeto(temp_fits.name, overwrite=True)
            
            # Recorrer los índices de las capas y unirlas usando 'astscript-psf-unite'
            for i in range(len(fits_files) - 1, 0, -1):
                if i ==(len(fits_files)-1): outer_fits = fits_files[i]
                inner_fits = fits_files[i - 1]
                if len(list(index_outer_dict.keys()))==1:
                    index_key = list(index_outer_dict.keys())[0]
                else:
                    index_key = f"Outer_{i-1}_{i}_{filter}"  

                # Obtener el radio desde el diccionario de índices
                radius = index_outer_dict.get(index_key, 12)  # Si no está el índice, por defecto 12

                # Definir el archivo de salida temporal para esta iteración
                output_temp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                output_temp.close()
                # Ejecutar el comando 'astscript-psf-unite' para combinar las dos capas
                if i ==(len(fits_files)-1):
                    command = [
                        f"{ast_psf_unite}", outer_fits,
                        f"--inner={inner_fits}",
                        f"--radius={radius}",
                        f"--scale={scale}",
                        f"--hdu=0",
                        f"--innerhdu=0",
                        f"--quiet",
                        f"--output={output_temp.name}"
                    ]
                else:
                    command = [
                        f"{ast_psf_unite}", outer_fits,
                        f"--inner={inner_fits}",
                        f"--radius={radius}",
                        f"--scale={scale}",
                        f"--hdu=1",
                        f"--innerhdu=0",
                        f"--quiet",
                        f"--output={output_temp.name}"
                    ]
                print(f"Ejecutando: {' '.join(command)}")
                subprocess.run(command, check=True)
            
                # Actualizar el archivo outer con el resultado para la próxima iteración
                outer_fits = output_temp.name

            # Al final del bucle, el archivo 'outer_fits' será el archivo PSF final
            # Renombrar este archivo al archivo de salida final
            puf=shutil.copy(outer_fits, inner_dir)
            os.rename(puf, output_filename)
            print(f"PSF combinado guardada en {output_filename}")
            
            # # Abrir el PSF final y añadir los anillos en los radios de unión
            # with fits.open(output_filename, mode='update') as hdul:
            #     psf_image = hdul[1].data
            #     center = ((psf_image.shape[1] // 2)+1, (psf_image.shape[0] // 2)+1)  # Asumimos que el centro es la mitad de la imagen
                
            #     # Añadir un anillo en cada radio dado por index_outer_dict
            #     for radius in index_outer_dict.values():
            #         psf_image = create_annulus(psf_image, center, radius, thickness=5)
            #         #psf_image = smooth_interpolation(psf_image, order=3)
            #     # Guardar el PSF con los anillos añadidos
            #     hdul.close()
            #     print(f"PSF con anillos guardada en {output_filename}")
            # with fits.open(output_filename) as hduli:
            #     psf_image = hduli[1].data
                
            #     psf_image = smooth_interpolation_with_inpaint(psf_image)
            #     hduli[1].data=psf_image
            #     # Guardar el PSF con los anillos añadidos
            #     hduli.writeto(output_filename, overwrite=True)
            #     print(f"PSF interpolada guardada en {output_filename}")
                        # Abrir el PSF final y añadir los anillos en los radios de unión
            hdul=fits.open(output_filename)
            psf_image = hdul[1].data
        finally:
            # Limpiar todos los archivos temporales
            for file in fits_files:
                if os.path.exists(file):
                    os.remove(file)
            return(psf_image,output_filename)
    else:
        # Crear una lista para los archivos FITS temporales
        fits_files = []

        try:
            # Crear archivos FITS temporales para cada stack_calibrated
            for key, data in results.items():
                # Crear un archivo FITS temporal
                temp_fits = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                fits_files.append(temp_fits.name)
                
                # Guardar la matriz 'stack_calibrated' en un archivo FITS temporal
                hdu = fits.PrimaryHDU(data['stack_calibrated'])
                hdu.writeto(temp_fits.name, overwrite=True)

            # Recorrer los índices de las capas y unirlas usando 'astscript-psf-unite'
            for i in range(len(fits_files) - 1, 0, -1):
                if i ==(len(fits_files)-1): outer_fits = fits_files[i]
                inner_fits = fits_files[i - 1]
                index_key = f"{name_gal}_{inner_keys[i-1]}_{inner_keys[i]}_{filter}"  # Generar la clave del índice

                # Obtener el radio desde el diccionario de índices
                radius = index_outer_dict.get(index_key, 12)  # Si no está el índice, por defecto 12

                # Definir el archivo de salida temporal para esta iteración
                output_temp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                output_temp.close()
                
                # Ejecutar el comando 'astscript-psf-unite' para combinar las dos capas
                if i ==(len(fits_files)-1):
                    command = [
                        f"{ast_psf_unite}", outer_fits,
                        f"--inner={inner_fits}",
                        f"--radius={radius}",
                        f"--scale={scale}",
                        f"--hdu=0",
                        f"--innerhdu=0",
                        f"--quiet",
                        f"--output={output_temp.name}"
                    ]
                else:
                    command = [
                        f"{ast_psf_unite}", outer_fits,
                        f"--inner={inner_fits}",
                        f"--radius={radius}",
                        f"--scale={scale}",
                        f"--hdu=1",
                        f"--innerhdu=0",
                        f"--quiet",
                        f"--output={output_temp.name}"
                    ]
                
                print(f"Ejecutando: {' '.join(command)}")
                subprocess.run(command, check=True)
                # Actualizar el archivo outer con el resultado para la próxima iteración
                outer_fits = output_temp.name
                
            # Al final del bucle, el archivo 'outer_fits' será el archivo PSF final
            # Renombrar este archivo al archivo de salida final
            puf=shutil.copy(outer_fits, inner_dir)
            os.rename(puf, output_filename)
            print(f"PSF combinado guardada en {output_filename}")
            
            # Abrir el PSF final y añadir los anillos en los radios de unión
            hdul=fits.open(output_filename)
            psf_image = hdul[1].data
        finally:
            # Limpiar todos los archivos temporales
            for file in fits_files:
                if os.path.exists(file):
                    os.remove(file)
            return(psf_image,output_filename)

def build_profile(results, index_outer_dict, inner_keys):
    inner_key, outer_key = inner_keys[0], inner_keys[1]
    
    x_outer = results[outer_key]['x']
    y_outer = results[outer_key]['y_calibrated']
    
    x_inner = results[inner_key]['x']
    y_inner = results[inner_key]['y_calibrated']
    
    index_key = list(index_outer_dict.keys())[0]  
    index = index_outer_dict[index_key]
    
    x_psf_final = np.concatenate((x_inner[:index],x_outer[index:]))
    y_psf_final = np.concatenate(( y_inner[:index],y_outer[index:]))

    return x_psf_final, y_psf_final


class PSF_joint:
    """
    For joint different parts of the PSF
    """
    def __init__(self,filters,sample_dir,directorio_inner,directorio_outers,parts,index_ranges,outer_part_sample,outer_index_ranges,\
                 use_outer_extended,dir_outer_extended,range_outer_extended,name_parallel,outer_parts_to_build):
        self.filters = filters ; self.directorio_inner = directorio_inner
        self.parts = parts     ; self.sample_dir=sample_dir
        self.index_ranges=index_ranges; self.directorio_outers=directorio_outers
        self.outer_part_sample=outer_part_sample; self.outer_index_ranges=outer_index_ranges
        self.psfs_dir= "./PSF_files"; self.use_outer_extended=use_outer_extended; self.dir_outer_extended=dir_outer_extended
        self.range_outer_extended=range_outer_extended.split(",");self.name_parallel=name_parallel
        self.outer_parts_to_build=outer_parts_to_build

    def check_outer_directory(self, directory,name,filter,subdirs_outer):
        # Lista los subdirectorios esperados
        
        expected_directories = [f"{directory}/{name}/{part}" for part in subdirs_outer]
        
        # Expresiones regulares para los nombres de archivos
        file_patterns = {
            'stack': re.compile(f"stack_*_{filter}_*_.fits"),
            'profile': re.compile(f"profile_*_{filter}_*_.fits")
        }
        # Cargar los datos desde el archivo usando np.loadtxt
        data = np.loadtxt('./surface_brightness_limits.txt', delimiter=':', dtype=[('name', 'U50'), ('value', 'f8')])

        filter_data = [(name, value) for name, value in data if name.endswith(f'_{filter}')] 
        
        names = [filter_data[i][0] for i in range(len(filter_data))]
        values= [filter_data[i][1] for i in range(len(filter_data))] 
        limits_dict =dict(zip(names, values))
        
        for subdir_path in expected_directories:

            # Lista los archivos en el subdirectorio
            files_in_subdir = os.listdir(subdir_path)
            
            found_stack_file = False
            found_profile_file = False
            
            # Verifica los archivos en el subdirectorio
            for file_name in files_in_subdir:
                if file_name.startswith('stack_'):
                    found_stack_file = True
                if file_name.startswith('profile_'):
                    found_profile_file = True
            
            # Reportar resultados
            if not found_stack_file and not found_profile_file:
                print(f"Missign 'stack' and 'profile' files at {subdir_path}")
                
                sf_val = limits_dict[f"{name}"]
                
                reference_value = sf_val
                
                # Ordenar el diccionario según la cercanía a reference_value
                sorted_data = sorted(limits_dict.items(), key=lambda item: abs(item[1] - reference_value))
                names_sorted = [sorted_data[i][0] for i in range(len(sorted_data))]
                part_to_add = subdir_path.split("/")[-1]
                
                for gal in names_sorted:
                    dir_to_add = f"{directory}/{gal}/{part_to_add}"
                    
                    # Lista los archivos en el subdirectorio
                    try:
                        files_in_dir_to_add = os.listdir(dir_to_add)
                        
                        found_stack_add = False
                        found_profile_add = False
                        
                        # Verifica los archivos en el subdirectorio
                        for file_name in files_in_dir_to_add:
                            if file_name.startswith('stack_'):
                                found_stack_add = True
                                path_stack= os.path.join(dir_to_add,file_name)
                                os.system(f"cp -r {path_stack} {subdir_path}")
                            if file_name.startswith('profile_'):
                                found_profile_add = True
                                path_stack= os.path.join(dir_to_add,file_name)
                                os.system(f"cp -r {path_stack} {subdir_path}")
                        if found_stack_add and found_profile_add:
                            break
                    except:continue
        return(None)
        
    def check_directory(self, sample_dir, directory,name,filter):
        # Lista los subdirectorios esperados
        expected_directories = [f"{directory}/{name}_{part}_{filter}" for part in self.parts]
        
        archivo_fits_cut=[archivo for archivo in os.listdir(sample_dir) if re.search(f"{name}_{filter}"+'.*\.fits', archivo)]

        # Expresiones regulares para los nombres de archivos
        file_patterns = {
            'stack': re.compile(f"stack_*_{filter}_*_.fits"),
            'profile': re.compile(f"profile_*_{filter}_*_.fits")
        }
        # Cargar los datos desde el archivo usando np.loadtxt
        data = np.loadtxt('./surface_brightness_limits.txt', delimiter=':', dtype=[('name', 'U50'), ('value', 'f8')])

        filter_data = [(name, value) for name, value in data if name.endswith(f'_{filter}')] 
        
        names = [filter_data[i][0] for i in range(len(filter_data))]
        values= [filter_data[i][1] for i in range(len(filter_data))] 
        limits_dict =dict(zip(names, values))
        for subdir_path in expected_directories:
            if not os.path.isdir(subdir_path) and len(archivo_fits_cut)>0:
                print(f"Missing directory: {subdir_path}")
                os.makedirs(subdir_path)
            elif not os.path.isdir(subdir_path) and len(archivo_fits_cut)==0:
                print(f"Missing filter file: {name}_{filter}")
                return(f"{name}_{filter}")
                #break

            # Lista los archivos en el subdirectorio
            files_in_subdir = os.listdir(subdir_path)
            
            found_stack_file = False
            found_profile_file = False
            
            # Verifica los archivos en el subdirectorio
            for file_name in files_in_subdir:
                if file_name.startswith('stack_'):
                    found_stack_file = True
                if file_name.startswith('profile_'):
                    found_profile_file = True

            # Reportar resultados
            if not found_stack_file and not found_profile_file:
                print(f"Missign 'stack' and 'profile' files at {subdir_path}")
                sf_val = limits_dict[f"{name}_{filter}"]
                reference_value = sf_val

                # Ordenar el diccionario según la cercanía a reference_value
                sorted_data = sorted(limits_dict.items(), key=lambda item: abs(item[1] - reference_value))
                names_sorted = [sorted_data[i][0] for i in range(len(sorted_data))]
                part_to_add = subdir_path.split("_")[-2]
                dir_prin= "/".join(directory.split("/")[:-1])
                
                for gal in names_sorted:
                    n=gal.split("_")[0]
                    dir_to_add = f"{dir_prin}/{n}/{n}_{part_to_add}_{filter}"
                    # Lista los archivos en el subdirectorio
                    try:
                        files_in_dir_to_add = os.listdir(dir_to_add)
                        
                        found_stack_add = False
                        found_profile_add = False
                        
                        # Verifica los archivos en el subdirectorio
                        for file_name in files_in_dir_to_add:
                            if file_name.startswith('stack_'):
                                found_stack_add = True
                                path_stack= os.path.join(dir_to_add,file_name)
                                os.system(f"cp -r {path_stack} {subdir_path}")
                            if file_name.startswith('profile_'):
                                found_profile_add = True
                                path_stack= os.path.join(dir_to_add,file_name)
                                os.system(f"cp -r {path_stack} {subdir_path}")
                        if found_stack_add and found_profile_add:
                            break
                    except:continue
        return(None)
    def junction_parts(self):
        for filter in self.filters:
            
            ##############
            # Check directories and files in the sample directory 
            ##############
            missing_files = []
            for nombre in os.listdir(self.directorio_inner):
                ruta_completa = os.path.join(self.directorio_inner, nombre)
                add=self.check_directory(self.sample_dir,ruta_completa,nombre,filter)
                missing_files.append(add)
            missing_files=np.array([d for d in missing_files if d is not None])

            ##############
            # Outer parts
            ##############
            filtered_directories = [archivo for archivo in os.listdir(self.directorio_outers) if re.search('_'+filter+'_'+'.*'+self.outer_part_sample[0]+'_'+self.outer_part_sample[1], archivo)]
            sorted_directories = sorted(filtered_directories, key=lambda x: int(x.split('_')[1]))

            files_dict=process_subdirectories(self.directorio_outers,sorted_directories)
            files_dict_ordenado = dict(sorted(files_dict.items(), key=lambda x: int(x[0].split('_')[1])))
            
            dir_s = os.path.join(self.directorio_outers,sorted_directories[0])

            files = os.listdir(dir_s)
            
            # Verifica los archivos en el subdirectorio
            for file_name in files:
                if file_name.startswith('stack_'):
                    path_stack= os.path.join(dir_s,file_name)
                    outer_dir = os.path.join(self.psfs_dir, "PSF_outer_final");os.makedirs(outer_dir, exist_ok=True)
                    output_outer_dir =os.path.join(outer_dir, f"psf_outer_{filter}.fits")
                    puf=shutil.copy(path_stack, outer_dir)
                    os.rename(puf, output_outer_dir)
                if file_name.startswith('profile_'):
                    path_stack= os.path.join(dir_s,file_name)
                    outer_dir = os.path.join(self.psfs_dir, "PSF_outer_final");os.makedirs(outer_dir, exist_ok=True)
                    output_profile_outer_dir =os.path.join(outer_dir, f"psf_outer_profile_{filter}.fits")
                    puf=shutil.copy(path_stack, outer_dir)
                    os.rename(puf, output_profile_outer_dir)

            ##############
            # Inner parts
            ##############
            sample_archivos = [archivo.split("_")[0] for archivo in os.listdir(self.sample_dir) if re.search(filter+'.*\.fits', archivo)]
            for nombre in sample_archivos:
                try:
                    print('\n'+40*'=')
                    print('\n Galaxy, filter: %s %s'%(nombre,filter))
                    print('\n'+40*'=')    
                    ruta_completa = os.path.join(self.directorio_inner, nombre)
                    name_control=f"{nombre}_{filter}"
                    
                    ruta_completa_outers = os.path.join(self.directorio_outers,name_control)

                    if name_control in missing_files: 
                        pass
                    else:
                        ##################
                        # Inner PSF data
                        ##################
                        #Read internal PSF data
                        subdirs = [subdir + f"_{filter}" for subdir in self.parts]

                        # Diccionario para almacenar los resultados

                        files_dict=process_subdirectories(ruta_completa,subdirs)
                        
                        name_subdirs = [nombre+"_"+subdir + f"_{filter}" for subdir in self.parts]

                        index_inner_dict=calculate_junction_indices(files_dict, name_subdirs, self.index_ranges,False)
                        
                        files_dict_ordenado = {key: files_dict[key] for key in name_subdirs if key in files_dict}
                        
                        results_inner=calibrate_layers(files_dict_ordenado, index_inner_dict,False,self.parts,filter)
                        results_inner_ordenado = {key: results_inner[key] for key in name_subdirs if key in results_inner}
                        
                        inner_dir = os.path.join(self.psfs_dir, f"PSFs_inner_final/{nombre}_{filter}");os.makedirs(inner_dir, exist_ok=True)
                        output_inner_dir =os.path.join(inner_dir, f"psf_inner_{nombre}_{filter}.fits")
                        stack_inner,stack_inner_filename= unite_psf_with_gnuastro(results_inner_ordenado,filter, index_inner_dict,False,self.parts,\
                                                                                nombre, output_filename=output_inner_dir, inner_dir=inner_dir,scale=1)


                        ##################
                        # Outers PSF data
                        ##################
                        
                        subdirs_outer = [os.listdir(ruta_completa_outers)[0]] #os.listdir(ruta_completa_outers)
                        
                        self.check_outer_directory(self.directorio_outers,name_control,filter,subdirs_outer)
                        
                        #second_mask_outlayers(ruta_completa_outers,subdirs_outer)
                    
                        files_dict_outer=process_subdirectories(ruta_completa_outers,subdirs_outer)
                    
                        ext_profile_dir = os.path.join(self.directorio_outers,f"Outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}/profile_psf_outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}.fits")
                        ext_stack_dir = os.path.join(self.directorio_outers,f"Outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}/stack_outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}.fits")

                        x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(ext_profile_dir)

                        valid = ~np.isnan(y_outer)

                        # Seleccionar los últimos 50 puntos válidos (no NaN)
                        y_valid = y_outer[valid][-100:]
                        y_sky_clip = sigma_clip(y_valid, 2)
                        y_sky_clip_2 = y_sky_clip.data[np.invert(y_sky_clip.mask)]
                        sky = np.median(y_sky_clip_2)

                        y_outer=y_outer-sky
                        
                        ##############
                        #Combine inner and outer parts
                        ##############
                        output_profile_inner_dir =os.path.join(inner_dir, f"psf_profile_inner_{nombre}_{filter}.fits")
                        os.system(f'astscript-radial-profile {stack_inner_filename} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=2500 -o {output_profile_inner_dir}')

                        x_inner,y_inner,std_inner,area_inner,rms_inner =profiles_reader(f"{output_profile_inner_dir}")
                    

                        
                        ################
                        # Extend the PSF
                        ################
                        if self.use_outer_extended:
                            
                            profile_outer_extended=[archivo for archivo in os.listdir(self.dir_outer_extended) if re.search('_'+filter+'_'+'.*profile', archivo)]
                            dir_outer_extended = os.path.join(self.dir_outer_extended, profile_outer_extended[0])
                            
                            # x_complete_1,y_complete_1,std_complete,area_complete,rms_complete =profiles_reader(f"{output_profile_outer_dir}")
                            # x_complete,y_complete = local_sky_subs(x_complete_1,y_complete_1)

                            x_extended,y_extended,std_extended,area_extended,rms_extended =profiles_reader(f"{dir_outer_extended}")
                            shape_index=min(x_outer.shape[0],x_extended.shape[0])
                            x_outer,y_outer=x_outer[:shape_index],y_outer[:shape_index]
                            x_extended,y_extended=x_extended[:shape_index],y_extended[:shape_index]
                            rms_outer,rms_extended=rms_outer[:shape_index],rms_extended[:shape_index]
                            area_outer,area_extended=area_outer[:shape_index],area_extended[:shape_index]
                            

                            if filter=="g":
                                index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,60,80)
                                
                                fc_union_outer_part,c_union_outer_part=fc_c_calc(index_union_extended,y_extended,y_outer)
                                y_union_outer_part=(y_outer*fc_union_outer_part)+c_union_outer_part

                                y_outer_to_extend=np.concatenate((y_union_outer_part[:index_union_extended],y_extended[index_union_extended:]))
                                x_to_circ,y_to_circ=extend_psf_in_range(x_outer,y_outer_to_extend,filter,900,1400)
                            elif filter=="i": 
                                #index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,int(self.range_outer_extended[0]),int(self.range_outer_extended[1]))
                                
                                index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,130,180)
                                fc_union_outer_part,c_union_outer_part=fc_c_calc(index_union_extended,y_extended,y_outer)
                                y_union_outer_part=(y_outer*fc_union_outer_part)+c_union_outer_part
                                
                                y_outer_to_extend=np.concatenate((y_union_outer_part[:index_union_extended],y_extended[index_union_extended:]))
                                x_to_circ,y_to_circ=extend_psf_in_range(x_outer,y_outer_to_extend,filter,1210,1500)
                            else: 
                                #index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,int(self.range_outer_extended[0]),int(self.range_outer_extended[1]))
                                
                                index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,50,70)
                                fc_union_outer_part,c_union_outer_part=fc_c_calc(index_union_extended,y_extended,y_outer)
                                y_union_outer_part=(y_outer*fc_union_outer_part)+c_union_outer_part
                                
                                y_outer_to_extend=np.concatenate((y_union_outer_part[:index_union_extended],y_extended[index_union_extended:]))
                                x_to_circ,y_to_circ=extend_psf_in_range(x_outer,y_outer_to_extend,filter,1210,1500)
                                    
                            
                        ################
                        # Join internal part with outer part
                        ################

                        if nombre=="NGC0521":
                            # os.system(f'astscript-radial-profile {ext_stack_dir} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=6001 -o {output_profile_outer_dir.replace(".fits","_temp.fits")}')
                            # x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(output_profile_outer_dir.replace(".fits","_temp.fits"))
                            index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,40,50)
                        elif nombre=="IC1101":
                            # os.system(f'astscript-radial-profile {ext_stack_dir} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=6001 -o {output_profile_outer_dir.replace(".fits","_temp.fits")}')
                            # x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(output_profile_outer_dir.replace(".fits","_temp.fits"))
                            if filter=="r": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,5,15)
                            elif filter=="g": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,15,25)
                            elif filter=="i": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,15,20)
                        else:
                            #x_outer,y_outer,std_outer,area_outer,rms_outer =profiles_reader(f"{output_profile_outer_dir}")
                            index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,20,30)

                        
                        fc_union,c_union=fc_c_calc(index_union,y_to_circ,y_inner)
                        y_union=(y_inner*fc_union)+c_union
                        
                        x_psf_final = np.concatenate((x_inner[:index_union],x_to_circ[index_union:]))
                        y_psf_final = np.concatenate(( y_union[:index_union],y_to_circ[index_union:]))
                        
                        plt.figure()
                        plt.loglog(x_psf_final,y_psf_final,"k.")
                        plt.show()
                        final_dir = os.path.join(self.psfs_dir, f"PSFs_complete/{nombre}_{filter}");os.makedirs(final_dir, exist_ok=True)
                        output_final_dir =os.path.join(final_dir, f"psf_complete_profile_temp_{nombre}_{filter}.fits")

                        radios_intensidades = [(r, i) for r, i in zip(x_psf_final, y_psf_final) if i != 0]
                        radios_ordenados, intensidades_ordenadas = zip(*sorted(radios_intensidades))
                        
                        hdul = fits.open(f"{dir_outer_extended}")
                        #Save profile to circularize
                        data_combined = np.zeros(len(radios_ordenados), dtype=[('RADIUS', '>f4'), ('MEAN', '>f4')])
                        data_combined = np.zeros(len(intensidades_ordenadas), dtype=[('RADIUS', '>f4'), ('MEAN', '>f4')])
                        data_combined['RADIUS'] =radios_ordenados
                        data_combined['MEAN'] = intensidades_ordenadas
                        hdul[1].header["NAXIS2"]=int(len(intensidades_ordenadas))

                        new_hdul = fits.HDUList([
                            fits.PrimaryHDU(header=hdul[0].header),  # Conserva la extensión primaria
                            fits.BinTableHDU(data_combined, header=hdul[1].header)  # Crea una nueva tabla binaria con los nuevos datos
                        ])

                        new_hdul.writeto(f"{output_final_dir}", overwrite=True)   

                        
                        ##################
                        # Create custom tables
                        ##################    
                        interval_dir=os.path.join(final_dir, f"Intervals")
                        if not os.path.isdir(interval_dir): os.makedirs(interval_dir)
                        custom_table_dir=os.path.join(final_dir, f"Custom_tables")
                        if not os.path.isdir(custom_table_dir): os.makedirs(custom_table_dir)
                        
                        interval_path=os.path.join(interval_dir,f"{nombre}_{filter}_interval_tmp.fits")
                        custom_table_path=os.path.join(custom_table_dir,f"{nombre}_{filter}_custom_table.fits")
                        os.system(f'asttable {output_final_dir} \
                                    --output={interval_path} -c"arith RADIUS sorted-to-interval,MEAN" -o {custom_table_path}')

                        name_control=f'{nombre}_{filter}'        
                        final_dir = os.path.join(self.psfs_dir, f"PSFs_complete/{name_control}")
                        custom_table_dir=os.path.join(final_dir, f"Custom_tables")
                        ruta_custom_table = os.path.join(custom_table_dir, f"{nombre}_{filter}_custom_table.fits")
                        ruta_dir = os.path.join(final_dir, f"Circular_profiles")
                        if not os.path.isdir(ruta_dir): os.makedirs(ruta_dir)

                        path_circular= os.path.join(final_dir, f"psf_{name_control}.fits")
                        os.system(f'echo "1 6801 6801 8 13601 0 0 1 0 1" \
                                        | astmkprof --customtable={ruta_custom_table} \
                                                    --mergedsize=13601,13601 \
                                                    --output={path_circular} \
                                                    --mcolnocustprof \
                                                    --oversample=1 \
                                                    --clearcanvas \
                                                    --mode=img')

                        imagen_2d = fits.getdata(path_circular)
                        
                        imagen_2d /= np.nansum(imagen_2d)
                        hdu = fits.PrimaryHDU(imagen_2d);hdul = fits.HDUList([hdu])
                        hdul.writeto(path_circular, overwrite=True)
                        hdul.close()

                        os.system(f"astscript-radial-profile {path_circular} --quiet --hdu=0\
                                    --measure=mean,std,area,semi-major --rmax=6801 -o {os.path.join(final_dir, f'psf_profile_{name_control}.fits')}")
                except:continue



    def junction_parts_parallel(self):

        nombre=self.name_parallel.split("_")[0]
        filter=self.name_parallel.split("_")[1]
            
        ##############
        # Check directories and files in the sample directory 
        ##############
        missing_files = []
        for diri in os.listdir(self.directorio_inner):
            ruta_completa = os.path.join(self.directorio_inner, diri)
            add=self.check_directory(self.sample_dir,ruta_completa,diri,filter)
            missing_files.append(add)
        missing_files=np.array([d for d in missing_files if d is not None])

        ##############
        # Outer parts
        ##############
        
        filtered_directories = [archivo for archivo in os.listdir(self.directorio_outers) if re.search('_'+filter+'_'+'.*'+self.outer_part_sample[0]+'_'+self.outer_part_sample[1], archivo)]
        sorted_directories = sorted(filtered_directories, key=lambda x: int(x.split('_')[1]))
        files_dict=process_subdirectories(self.directorio_outers,sorted_directories)
        files_dict_ordenado = dict(sorted(files_dict.items(), key=lambda x: int(x[0].split('_')[1])))
        
        dir_s = os.path.join(self.directorio_outers,sorted_directories[0])

        files = os.listdir(dir_s)

        # Verifica los archivos en el subdirectorio
        for file_name in files:
            if file_name.startswith('stack_'):
                path_stack= os.path.join(dir_s,file_name)
                outer_dir = os.path.join(self.psfs_dir, "PSF_outer_final");os.makedirs(outer_dir, exist_ok=True)
                output_outer_dir =os.path.join(outer_dir, f"psf_outer_{filter}.fits")
                puf=shutil.copy(path_stack, outer_dir)
                os.rename(puf, output_outer_dir)
            if file_name.startswith('profile_'):
                path_stack= os.path.join(dir_s,file_name)
                outer_dir = os.path.join(self.psfs_dir, "PSF_outer_final");os.makedirs(outer_dir, exist_ok=True)
                output_profile_outer_dir =os.path.join(outer_dir, f"psf_outer_profile_{filter}.fits")
                puf=shutil.copy(path_stack, outer_dir)
                os.rename(puf, output_profile_outer_dir)


        ##############
        # Inner parts
        ##############
    
        print('\n'+40*'=')
        print('\n Galaxy, filter: %s %s'%(nombre,filter))
        print('\n'+40*'=')    
        ruta_completa = os.path.join(self.directorio_inner, nombre)
        name_control=f"{nombre}_{filter}"
        
        ruta_completa_outers = os.path.join(self.directorio_outers,name_control)

        if name_control in missing_files: 
            pass
        else:
            ##################
            # Inner PSF data
            ##################
            #Read internal PSF data
            subdirs = [subdir + f"_{filter}" for subdir in self.parts]

            # Diccionario para almacenar los resultados

            files_dict=process_subdirectories(ruta_completa,subdirs)
            
            name_subdirs = [nombre+"_"+subdir + f"_{filter}" for subdir in self.parts]

            index_inner_dict=calculate_junction_indices(files_dict, name_subdirs, self.index_ranges,False)
            
            files_dict_ordenado = {key: files_dict[key] for key in name_subdirs if key in files_dict}
            
            results_inner=calibrate_layers(files_dict_ordenado, index_inner_dict,False,self.parts,filter)
            results_inner_ordenado = {key: results_inner[key] for key in name_subdirs if key in results_inner}
            
            inner_dir = os.path.join(self.psfs_dir, f"PSFs_inner_final/{nombre}_{filter}");os.makedirs(inner_dir, exist_ok=True)
            output_inner_dir =os.path.join(inner_dir, f"psf_inner_{nombre}_{filter}.fits")
            stack_inner,stack_inner_filename= unite_psf_with_gnuastro(results_inner_ordenado,filter, index_inner_dict,False,self.parts,\
                                                                        nombre, output_filename=output_inner_dir, inner_dir=inner_dir,scale=1)


            ##################
            # Outers PSF data
            ##################
            
            subdirs_outer = [os.listdir(ruta_completa_outers)[0]] #os.listdir(ruta_completa_outers)
            
            self.check_outer_directory(self.directorio_outers,name_control,filter,subdirs_outer)
            
            #second_mask_outlayers(ruta_completa_outers,subdirs_outer)
            
            files_dict_outer=process_subdirectories(ruta_completa_outers,subdirs_outer)
            
            ext_profile_dir = os.path.join(self.directorio_outers,f"Outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}/profile_psf_outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}.fits")
            ext_stack_dir = os.path.join(self.directorio_outers,f"Outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}/stack_outer_4_{filter}_suf_mag_{self.outer_part_sample[0]}_{self.outer_part_sample[1]}.fits")

            x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(ext_profile_dir)

            valid = ~np.isnan(y_outer)

            # Seleccionar los últimos 50 puntos válidos (no NaN)
            y_valid = y_outer[valid][-100:]
            y_sky_clip = sigma_clip(y_valid, 2)
            y_sky_clip_2 = y_sky_clip.data[np.invert(y_sky_clip.mask)]
            sky = np.median(y_sky_clip_2)

            y_outer=y_outer-sky
            
            ##############
            #Combine inner and outer parts
            ##############
            output_profile_inner_dir =os.path.join(inner_dir, f"psf_profile_inner_{nombre}_{filter}.fits")
            os.system(f'astscript-radial-profile {stack_inner_filename} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=2500 -o {output_profile_inner_dir}')

            x_inner,y_inner,std_inner,area_inner,rms_inner =profiles_reader(f"{output_profile_inner_dir}")
            
            if nombre=="NGC0521":
                # os.system(f'astscript-radial-profile {ext_stack_dir} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=6001 -o {output_profile_outer_dir.replace(".fits","_temp.fits")}')
                # x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(output_profile_outer_dir.replace(".fits","_temp.fits"))
                index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,40,50)
            elif nombre=="IC1101":
                # os.system(f'astscript-radial-profile {ext_stack_dir} --quiet --hdu=1 --measure=mean,std,area,semi-major --rmax=6001 -o {output_profile_outer_dir.replace(".fits","_temp.fits")}')
                # x_outer,y_outer,std_outer,area_outer,rms_outer=profiles_reader(output_profile_outer_dir.replace(".fits","_temp.fits"))
                if filter=="r": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,5,15)
                elif filter=="g": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,35,45)
                elif filter=="i": index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,35,45)
            else:
                #x_outer,y_outer,std_outer,area_outer,rms_outer =profiles_reader(f"{output_profile_outer_dir}")
                index_union=r_junc_calc(y_inner,y_outer,rms_inner,rms_outer,area_inner,area_outer,20,30)

            
            fc_union,c_union=fc_c_calc(index_union,y_outer,y_inner)
            y_union=(y_inner*fc_union)+c_union

            ################
            # Extend the PSF
            ################
            if self.use_outer_extended:
                
                profile_outer_extended=[archivo for archivo in os.listdir(self.dir_outer_extended) if re.search('_'+filter+'_'+'.*profile', archivo)]
                dir_outer_extended = os.path.join(self.dir_outer_extended, profile_outer_extended[0])
                
                # x_complete_1,y_complete_1,std_complete,area_complete,rms_complete =profiles_reader(f"{output_profile_outer_dir}")
                # x_complete,y_complete = local_sky_subs(x_complete_1,y_complete_1)

                x_extended,y_extended,std_extended,area_extended,rms_extended =profiles_reader(f"{dir_outer_extended}")
                shape_index=min(x_outer.shape[0],x_extended.shape[0])
                x_outer,y_outer=x_outer[:shape_index],y_outer[:shape_index]
                x_extended,y_extended=x_extended[:shape_index],y_extended[:shape_index]
                rms_outer,rms_extended=rms_outer[:shape_index],rms_extended[:shape_index]
                area_outer,area_extended=area_outer[:shape_index],area_extended[:shape_index]
                
                index_union_extended=r_junc_calc(y_outer,y_extended,rms_outer,rms_extended,area_outer,area_extended,int(self.range_outer_extended[0]),int(self.range_outer_extended[1]))
                if filter=="g": x_to_circ,y_to_circ=extend_psf_in_range(x_outer,y_outer,filter,370,770)
                else: x_to_circ,y_to_circ=extend_psf_in_range(x_outer,y_outer,filter,index_union_extended+30,300)


            x_psf_final = np.concatenate((x_inner[:index_union],x_to_circ[index_union:]))
            y_psf_final = np.concatenate(( y_union[:index_union],y_to_circ[index_union:]))
            
            final_dir = os.path.join(self.psfs_dir, f"PSFs_complete/{nombre}_{filter}");os.makedirs(final_dir, exist_ok=True)
            output_final_dir =os.path.join(final_dir, f"psf_complete_profile_temp_{nombre}_{filter}.fits")

            radios_intensidades = [(r, i) for r, i in zip(x_psf_final, y_psf_final) if i != 0]
            radios_ordenados, intensidades_ordenadas = zip(*sorted(radios_intensidades))
            
            hdul = fits.open(f"{dir_outer_extended}")
            #Save profile to circularize
            data_combined = np.zeros(len(radios_ordenados), dtype=[('RADIUS', '>f4'), ('MEAN', '>f4')])
            data_combined = np.zeros(len(intensidades_ordenadas), dtype=[('RADIUS', '>f4'), ('MEAN', '>f4')])
            data_combined['RADIUS'] =radios_ordenados
            data_combined['MEAN'] = intensidades_ordenadas
            hdul[1].header["NAXIS2"]=int(len(intensidades_ordenadas))

            new_hdul = fits.HDUList([
                fits.PrimaryHDU(header=hdul[0].header),  # Conserva la extensión primaria
                fits.BinTableHDU(data_combined, header=hdul[1].header)  # Crea una nueva tabla binaria con los nuevos datos
            ])

            new_hdul.writeto(f"{output_final_dir}", overwrite=True)   

            
            ##################
            # Create custom tables
            ##################    
            interval_dir=os.path.join(final_dir, f"Intervals")
            if not os.path.isdir(interval_dir): os.makedirs(interval_dir)
            custom_table_dir=os.path.join(final_dir, f"Custom_tables")
            if not os.path.isdir(custom_table_dir): os.makedirs(custom_table_dir)
            
            interval_path=os.path.join(interval_dir,f"{nombre}_{filter}_interval_tmp.fits")
            custom_table_path=os.path.join(custom_table_dir,f"{nombre}_{filter}_custom_table.fits")
            os.system(f'asttable {output_final_dir} \
                        --output={interval_path} -c"arith RADIUS sorted-to-interval,MEAN" -o {custom_table_path}')

            name_control=f'{nombre}_{filter}'        
            final_dir = os.path.join(self.psfs_dir, f"PSFs_complete/{name_control}")
            custom_table_dir=os.path.join(final_dir, f"Custom_tables")
            ruta_custom_table = os.path.join(custom_table_dir, f"{nombre}_{filter}_custom_table.fits")
            ruta_dir = os.path.join(final_dir, f"Circular_profiles")
            if not os.path.isdir(ruta_dir): os.makedirs(ruta_dir)

            path_circular= os.path.join(final_dir, f"psf_{name_control}.fits")
            os.system(f'echo "1 6801 6801 8 13601 0 0 1 0 1" \
                            | astmkprof --customtable={ruta_custom_table} \
                                        --mergedsize=13601,13601 \
                                        --output={path_circular} \
                                        --mcolnocustprof \
                                        --oversample=1 \
                                        --clearcanvas \
                                        --mode=img')

            imagen_2d = fits.getdata(path_circular)
            
            imagen_2d /= np.nansum(imagen_2d)
            hdu = fits.PrimaryHDU(imagen_2d);hdul = fits.HDUList([hdu])
            hdul.writeto(path_circular, overwrite=True)
            hdul.close()

            os.system(f"astscript-radial-profile {path_circular} --quiet --hdu=0\
                        --measure=mean,std,area,semi-major --rmax=6801 -o {os.path.join(final_dir, f'psf_profile_{name_control}.fits')}")