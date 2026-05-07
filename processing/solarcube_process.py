"""Contains functions for processing raw data for SolarCube.

Example usage:
--------
    
    $ python solarcube_process.py
    
"""
import os
import re
import numpy as np
import pandas as pd
import h5py

# set paths to sources and targets
path_root_target = <set_target_dir_here>
path_root_source = <set_source_dir_here>

def main():
    # get content in raw data folder
    file_list = os.listdir(path_root_source)

    # iterate over directory content
    for content_name in file_list:
        # set path to file
        path_content = os.path.join(path_root_source, content_name)

        # check if hierarchical data format
        if content_name.endswith('.hdf'):

            # find station number
            station_number = re.findall(r'_(\d+)', content_name)[1]
            
            # set target name
            target_name = 'station_' + station_number

            # set path to target
            path_target = os.path.join(path_root_target, target_name)

            # create target folder
            os.makedirs(path_target, exist_ok=True)

            # set entryname
            if 'ir133' in content_name:
                entryname = 'ir133'
                savename = 'infrared_band_133'
            elif 'vis047' in content_name:
                entryname = 'vis047'
                savename = 'visual_band_47'
            elif 'vis086' in content_name:
                entryname = 'vis086'
                savename = 'visual_band_86'
            elif 'sza' in content_name:
                entryname = 'sza'
                savename = 'solar_zenith_angle'
            elif 'ssr' in content_name:
                entryname = 'ssr'
                savename = 'satellite_solar_radiation'
            elif 'cm' in content_name:
                entryname = 'cm'
                savename = 'cloud_mask'
            
            # load data
            data = h5py.File(path_content, 'r').get(entryname)[:]

            # create saving path
            path_save = os.path.join(path_target, savename + '.h5')

            # save unchanged data 
            with h5py.File(path_save, 'w') as hf:
                hf.create_dataset(savename, data=data)

        elif content_name == 'SolarCube_insitu.csv':
            # load data
            data = pd.read_csv(path_content) 

            # average in 15-row steps.
            data = data.groupby(np.arange(len(data)) // 15).mean()

            # set path for saving
            path_save = os.path.join(path_root_target, 'ground_radiation.csv')

            # save to csv
            data.to_csv(path_save, index=False)
            
        elif content_name == 'SolarCube_sitelist.csv':
            # load data
            data = pd.read_csv(path_content)

            # set path for saving
            path_save = os.path.join(path_root_target, 'station_features.csv')

            # save unchanged at target as csv file
            data.to_csv(path_save, index=False)  

        elif content_name == 'index':

            # set path to index folder
            path_target = os.path.join(path_root_target, 'availability_IDs')

            # create target folder
            os.makedirs(path_target, exist_ok=True)

            # list content of index folder
            index_content_list = os.listdir(path_content)

            # iterate over index folder
            for index_content in index_content_list:
                if index_content.endswith('.csv'):
                    # set path to file for loading
                    path_load = os.path.join(path_content, index_content)

                    # load data
                    data = pd.read_csv(path_load)

                    # find station number
                    station_number = re.findall(r'(\d+)_', index_content)[0]

                    # set saving name
                    save_name = 'station_' + station_number + '_timestamps.csv'

                    # path save
                    path_save = os.path.join(path_target, save_name)

                    # save unchanged at target as csv file
                    data.to_csv(path_save, index=False)

                else:
                    print('Skipping unhandeled index content:', index_content)
                    
        else:
            print('Skipping unhandled directory content:', content_name)


if __name__ == '__main__':
    main()
    print("Successfully processed SolarCube data.")