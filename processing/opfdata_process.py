"""Contains functions for processing raw data for OPFData.

Example usage:
--------
    
    $ python opfdata_process.py pglib_opf_case14_ieee regular
    
"""
from multiprocessing import Process
import argparse
import json
import os
import math
import sys

# Set up base paths
path_root = <set_root_dir_here>

def multithread_merge_gridopt(
    gridname: str, 
    perturbation: str, 
    group: str,
    dpoints_per_file: 
    int, 
    save=True
):
    """Merges all data files"""

    # tell us what's going on
    print(f"Iterating {group}")

    # create path to group
    path_group = os.path.join(path_gridtype, group)

    # read file list in iterated group folder
    file_list = os.listdir(path_group)

    # compute number of files
    n_files_group = len(file_list)

    # tell us the stats
    print(f"The group folder has {n_files_group} files.")

    # read a sample file
    path_sample_json = os.path.join(path_group, file_list[0])
    if not path_sample_json.endswith(".json"):
        raise Error("Sample file does not end in .json. Check your code!")

    # compute how many files merging will result in
    n_merged_files = math.ceil(n_files_group/dpoints_per_file)

    # tell us what's going on
    print(f"Merging {dpoints_per_file} samples into one file.",
        f"This will result in {n_merged_files} files for this group.")

    # iterate over all files in target_sample_size steps
    fload_counter = 0
    fsave_counter = 0
    dict_size_counter = 0
    save_dict = {}
    while fload_counter < len(file_list):
        # set file name for load
        fname_load = file_list[fload_counter]

        # increment fload_counter
        fload_counter += 1

        # set path for loading file name
        path_load = os.path.join(path_group, fname_load)

        # read the json file
        with open(path_load) as json_file:
            data_dict = json.load(json_file)

        # split the file name 
        fname_dict = fname_load.split('_')[1].split('.')[0]

        # save loaded json to expanding dictionary
        save_dict[fname_dict] = data_dict

        # increment dictionary size counter
        dict_size_counter += 1

        # check if reached target sample size
        if (
            dict_size_counter == dpoints_per_file 
            or fload_counter == len(file_list)
        ):
            print(f"Starting to save file number {fsave_counter+1}",
                 f"with {dict_size_counter} samples")

            # save merged files
            if save:
                # set filename
                save_fname = f"merged_{fsave_counter+1}.json"

                # set path to saving
                save_path_group = os.path.join(path_root, perturbation, 
                    gridname, 'merged', group)

                # create results path if not exists
                if not os.path.exists(save_path_group):
                    os.makedirs(save_path_group)

                # set path
                save_path_file = os.path.join(save_path_group, save_fname)

                # save save_dict as json file
                with open(save_path_file, 'w') as json_save_file:
                    json.dump(save_dict, json_save_file)
                    
                # tell us what happened
                print(f"Successfuly saved {save_path_file}")
                
            # increment fname_counter
            fsave_counter += 1

            # reset dict size counter and save_dict
            dict_size_counter = 0
            save_dict = {}

if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()

    # define arguments
    parser.add_argument('gridname', type=str,
        help='The grid type to process. Example: "pglib_opf_case2000_goc"')
    parser.add_argument('perturbation', type=str, 
        help='The gridopt dataset release. Choose "n-1" or "regular"')
    parser.add_argument('dpoints_per_file', type=int, nargs='?', default=150,
        help='The expected number of data points merged per file. Example: 150')

    # parse arguments
    args = parser.parse_args()

    # set value for release type
    if args.perturbation == 'n-1':
        perturbation = 'dataset_release_1_nminusone'
        
    elif args.perturbation == 'regular':
        perturbation = 'dataset_release_1'

    # set value for grid type
    gridname = args.gridname

    # set value for target file size in MB
    dpoints_per_file = args.dpoints_per_file

    # set path to current grid type
    path_gridtype = os.path.join(path_root, perturbation, gridname, 'raw', 
        'gridopt-dataset-tmp', perturbation, gridname)

    # read folders in grid type
    group_list = os.listdir(path_gridtype)

    # create thread list we want to fill and start
    process_list = []

    # iterate over elements in group list
    for group in group_list:
        process_list.append(
            Process(
                target=multithread_merge_gridopt, 
                args=(
                    gridname, perturbation, group, dpoints_per_file
                )
            )
        )

    # start all threads
    for process in process_list:
        process.start()

    # tell us what's going on
    print(f"\nSuccessfully executed merge_opf.py")
    
else:
    raise Error("Please execute code with\n,"
        "$ python opfdata_process.py <gridname> <perturbation>")

    
    
