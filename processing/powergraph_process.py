"""Contains functions for processing raw data for PowerGraph.

Example usage:
--------
    
    $ python powergraph_process.py
    
"""
import os
import json
import mat73
from concurrent.futures import ThreadPoolExecutor, as_completed

path_root_target = <set_target_dir_here>
path_root_source = <set_source_dir_here>
path_uk = os.path.join(path_root_source, 'dataset_cascades/uk/uk/raw')
path_ieee24 = os.path.join(path_root_source, 'dataset_cascades/ieee24/ieee24/raw')
path_ieee39 = os.path.join(path_root_source, 'dataset_cascades/ieee39/ieee39/raw')
path_ieee118 = os.path.join(path_root_source, 'dataset_cascades/ieee118/ieee118/raw')

max_workers = 1024
n_file = 100

def main():
    path_list = [
        path_uk,
        path_ieee24,
        path_ieee39,
        path_ieee118
    ]
    name_list = [
        'uk',
        'ieee24',
        'ieee39',
        'ieee118'
    ]

    for id_case, data_path in enumerate(path_list):
        casename = name_list[id_case]
        path_target = os.path.join(path_root_target, casename)
        os.makedirs(path_target, exist_ok=True)

        # Load data
        Bf = mat73.loadmat(os.path.join(data_path, 'Bf.mat'))['B_f_tot']
        blist = mat73.loadmat(os.path.join(data_path, 'blist.mat'))['bList']
        Ef_nc = mat73.loadmat(os.path.join(data_path, 'Ef_nc.mat'))['E_f_kenza']
        Ef = mat73.loadmat(os.path.join(data_path, 'Ef.mat'))['E_f_post']
        exp = mat73.loadmat(os.path.join(data_path, 'exp.mat'))['explainations']
        of_bi = mat73.loadmat(os.path.join(data_path, 'of_bi.mat'))['output_features']
        of_mc = mat73.loadmat(os.path.join(data_path, 'of_mc.mat'))['category']
        of_reg = mat73.loadmat(os.path.join(data_path, 'of_reg.mat'))['dns_MW']

        data = {
            "Bf": Bf,
            "blist": blist,
            "Ef_nc": Ef_nc,
            "Ef": Ef,
            "exp": exp,
            "of_bi": of_bi,
            "of_mc": of_mc,
            "of_reg": of_reg
        }

        n = len(Bf)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i_file, i in enumerate(range(0, n, n_file)):
                future = executor.submit(
                    _process_data, data, i, i_file+1, path_target
                )
                futures.append(future)

            for f in as_completed(futures):
                status = f.result()   # the string returned by _process_data
                print(status)

def _process_data(data, i, i_file, path_target):
    """Process a chunk of the data and save as JSON."""
    print(f'Processing file number {i_file}')

    Bf = data['Bf']
    blist = data['blist']
    Ef_nc = data['Ef_nc']
    Ef = data['Ef']
    exp = data['exp']
    of_bi = data['of_bi']
    of_mc = data['of_mc']
    of_reg = data['of_reg']

    n_file = 100
    data_dict = {}
    n = len(Bf)

    for j in range(i, min(i + n_file, n)):
        node_features = {
            1: Bf[j][0][:, 0].tolist(),
            2: Bf[j][0][:, 1].tolist(),
            3: Bf[j][0][:, 2].tolist()
        }
        edges_features = {
            1: Ef_nc[j][0][:, 0].tolist(),
            2: Ef_nc[j][0][:, 1].tolist(),
            3: Ef_nc[j][0][:, 2].tolist(),
            4: Ef_nc[j][0][:, 3].tolist(),
            5: Ef[j][0][:, 0].tolist(),
            6: Ef[j][0][:, 1].tolist(),
            7: Ef[j][0][:, 2].tolist(),
            8: Ef[j][0][:, 3].tolist()
        }
        labels = {
            1: of_bi[j][0].item(),
            2: of_mc[j][0].tolist(),
            3: of_reg[j],
            4: (exp[j][0] if exp[j][0] is None else exp[j][0].tolist())
        }
        edge_index = {
            1: blist[:, 0].tolist(),
            2: blist[:, 1].tolist()
        }

        data_dict[j] = {
            'nodes': node_features,
            'edges': edges_features,
            'edge_index': edge_index,
            'labels': labels
        }
    
    filename = f'merged_{i_file}.json'
    path_save = os.path.join(path_target, filename)
    with open(path_save, 'w') as json_outfile:
        json.dump(data_dict, json_outfile)

    return f'processed {filename}'


if __name__ == '__main__':
    main()
    print("Successfully processed Power Graph data.")
