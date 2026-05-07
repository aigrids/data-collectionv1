"""Contains functions for processing raw data for BuildingElectricity

Example usage:
--------
    
    $ python buildingelectricity_process.py
    
"""
import os
import pandas as pd
from PIL import Image
import shutil

# set paths to sources and targets
PATH_ROOT_TARGET = <set_target_dir_here>
PATH_ROOT_SOURCE = <set_source_dir_here>

PROFILES_LIST = [
    'profiles_92',
    'profiles_451'
]

SAT_IMAGE_TYPE_LIST = [
    'aspect',
    'ortho',
    'relief',
    'slope'
]

ZOOM_LEVEL_LIST = [
    'zoom1',
    'zoom2',
    'zoom3'
]

PROFILES_FILENAME = '2014 building-year profiles.csv'

def process_load_profiles():
    """
    Processes load profile data and save at target.

    """
    profile_mappings = {}
    # iterate over profiles
    for profile in PROFILES_LIST:
        # set path to source profile data
        path_profile_source = os.path.join(
            PATH_ROOT_SOURCE, profile, 'building-year profiles', 'original',
            PROFILES_FILENAME
        )

        # load data
        df_profiles = pd.read_csv(path_profile_source)

        # drop year entry. This is row 1.
        df_profiles.drop(labels=1, axis='index', inplace=True)

        # Drop 'building ID' from columns
        numeric_cols = df_profiles.columns.drop('building ID')

        # Convert those column labels to integers, then sort
        numeric_cols_sorted = sorted(numeric_cols, key=int)

        # Create the new sequential labels
        mapping = {old_col: str(i+1) 
                for i, old_col in enumerate(numeric_cols_sorted)}

        # Reindex the DataFrame in sorted order, then rename
        df_reordered = df_profiles[numeric_cols_sorted].rename(columns=mapping)

        # insert building ID column back in.
        df_reordered.insert(0, 'building ID', df_profiles['building ID'])

        # set path to target profile data
        path_subtask_target = os.path.join(
            PATH_ROOT_TARGET, profile
        )

        # make sure path exists
        os.makedirs(path_subtask_target, exist_ok=True)

        # path to target
        target_path = os.path.join(path_subtask_target, 'load_profiles.csv')

        # save df to csv
        df_reordered.to_csv(target_path, index=False)

        profile_mappings[profile] = mapping

    return profile_mappings


def process_meteo_data():
    """
    Processes meteo data and save at target.

    """
    # iterate over profiles
    for profile in PROFILES_LIST:
        # set path to source meteo data
        path_meteo_source = os.path.join(
            PATH_ROOT_SOURCE, profile, 'meteo data'
        )

        # set path to target meteo data
        path_meteo_target = os.path.join(
            PATH_ROOT_TARGET, profile, 'meteo_data'
        )

        # make sure path exists
        os.makedirs(path_meteo_target, exist_ok=True)

        # list meteo data
        meteo_list = os.listdir(path_meteo_source)

        # iterate over all meteo data files
        for meteo_file in meteo_list:
            # set path to meteo file
            path_meteo_file = os.path.join(path_meteo_source, meteo_file)

            # load file
            df_meteo = pd.read_csv(path_meteo_file)

            # get cluster id from filename. thats the second after split by "_"
            cluster_id = meteo_file.split("_")[1]

            # create filename
            filename =  f'cluster_{cluster_id}.csv'

            # create saving path
            path_save = os.path.join(path_meteo_target, filename)
            
            # save df to csv
            df_meteo.to_csv(path_save, index=False)



def process_building_images(profile_mappings: dict):
    """
    Processes building images and saves them in a target location
    with a new ID defined in profile_mappings.
    
    """
    # iterate over profiles
    for profile in PROFILES_LIST:

        # set mapping profile
        mapping_profile = profile_mappings[profile]

        ### Process padded images

        # set path to source image data
        path_buildimage_source = os.path.join(
            PATH_ROOT_SOURCE, profile, 'building imagery', 'padded'
        )

        # set path to target image data
        path_buildimage_target = os.path.join(
            PATH_ROOT_TARGET, profile, 'building_images', 'padded'
        )

        # make sure path exists
        os.makedirs(path_buildimage_target, exist_ok=True)

        # read content of building images directory
        images_list = os.listdir(path_buildimage_source)

        # iterate over building images list
        for image_filename in images_list:
            # Only proceed if it's a PNG
            if not image_filename.lower().endswith('.png'):
                continue

            # set image ID
            image_id_str = image_filename.split(' ')[1].replace('.png', '')

            # set new image ID
            new_image_id = mapping_profile[image_id_str]

            # build the source and target file paths
            source_path = os.path.join(path_buildimage_source, image_filename)
            new_filename = f"building_{new_image_id}.png"
            target_path = os.path.join(path_buildimage_target, new_filename)

            # open and save the image with the new ID in its name
            with Image.open(source_path) as img:
                img.save(target_path)


        ### Process raw images

        # set path to source image data
        path_buildimage_source = os.path.join(
            PATH_ROOT_SOURCE, profile, 'building imagery', 'raw'
        )

        # set path to target image data
        path_buildimage_target = os.path.join(
            PATH_ROOT_TARGET, profile, 'building_images', 'raw'
        )

        # make sure path exists
        os.makedirs(path_buildimage_target, exist_ok=True)

        # read content of building images directory
        images_list = os.listdir(path_buildimage_source)

        # iterate over building images list
        for image_filename in images_list:
            # Proceed only if file is a .tif (or .tiff)
            if not (image_filename.lower().endswith('.tif') or 
                    image_filename.lower().endswith('.tiff')):
                continue

            # set image ID
            image_id_str = image_filename.split(' ')[1].replace('.tif','').replace('.tiff','')

            # find the new image ID from the mapping
            new_image_id = mapping_profile[image_id_str]

            # build the source path
            source_path = os.path.join(path_buildimage_source, image_filename)

            # construct new filename with .png extension
            new_filename = f"building_{new_image_id}.png"

            # build the target path
            target_path = os.path.join(path_buildimage_target, new_filename)

            # open the TIFF file and save as PNG
            with Image.open(source_path) as img:
                img.save(target_path, format="PNG")


def process_cluster_images():
    """
    Copy .png satellite images from each profile's source directory,
    rename them as cluster_<id>.png, and save them in the target directory.
    
    """
    for profile in PROFILES_LIST:
        # Define source (where your raw .png satellite images are)
        path_sat_source = os.path.join(
            PATH_ROOT_SOURCE, profile, 'cluster imagery'
        )

        # Define target (where you want to copy renamed images)
        path_sat_target = os.path.join(
            PATH_ROOT_TARGET, profile, 'cluster_images'
        )

        # iterate over zoom level data
        for zoom_level in ZOOM_LEVEL_LIST:
            
            # set path to zoom level
            path_zoom_source = os.path.join(path_sat_source, zoom_level)

            for image_type in SAT_IMAGE_TYPE_LIST:
                # set path to image
                path_image_type = os.path.join(path_zoom_source, image_type)

                # image list
                image_list = os.listdir(path_image_type)

                # iterate over all image types
                for sat_file in image_list:

                    # We only care about .png files
                    if not sat_file.lower().endswith('.png'):
                        continue

                    # set cluster ID
                    cluster_id = sat_file.split("_")[2].replace('.png', '')

                    # Define the new name as "cluster_<id>.png"
                    new_filename = f"cluster_{cluster_id}.png"

                    # Make sure the target directory exists
                    path_image_type_target = os.path.join(path_sat_target, 
                        zoom_level, image_type)
                    os.makedirs(path_image_type_target, exist_ok=True)

                    # Build full source and target paths
                    source_path = os.path.join(path_image_type, sat_file)
                    target_path = os.path.join(path_image_type_target, 
                        new_filename)

                    # Copy the file to the new location with the new name
                    shutil.copy2(source_path, target_path)


if __name__ == '__main__':
    # process cluster images
    process_cluster_images()
    
    # process meteo data
    process_meteo_data()

    # process load profiles and get mapping to new IDs
    profile_mappings = process_load_profiles()

    # use mapping to new building IDs for processing building images
    process_building_images(profile_mappings)

    print("Successfully processed BuildingElectricity data.")