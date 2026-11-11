"""Contains functions for processing raw data for SystemImbalance.

All data is published by Bundesnetzagentur | SMARD.de under a CC BY 4.0 licence.
The raw series as downloaded from SMARD in October 2026 are provided at
https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance_raw, with one
CSV file per series and region. Download and unpack them with

    $ wget https://huggingface.co/datasets/nikakhosravi-acs/SystemImbalance_raw/resolve/main/raw.tar.gz
    $ tar -xzf raw.tar.gz

and set path_root_source to the unpacked folder raw.

Example usage:
--------

    $ python systemimbalance_process.py

"""
import os

import pandas as pd

# set paths to sources and targets
path_root_target = <set_target_dir_here>
path_root_source = <set_source_dir_here>

TIMEZONE = 'Europe/Berlin'

# Control zone data of balancing volumes starts in June 2022. The national
# series ends in September 2022 and does not match the sum of control zones
# point by point, so that volumes are only taken from control zones.
T_START_ZONES = pd.Timestamp('2022-06-22', tz=TIMEZONE)
T_START_NATIONAL = pd.Timestamp('2018-10-01', tz=TIMEZONE)

# Fixed end of all series (exclusive). Balancing energy prices are published
# up to 20 working days after the delivery month and can be revised until then.
T_END = pd.Timestamp('2026-07-01', tz=TIMEZONE)

ZONES = ['50Hertz', 'Amprion', 'TenneT', 'TransnetBW']

# Series names with their SMARD data IDs
SERIES_LABELS_ZONES = {
    'vol_pos': 4383,        # balancing volume at undersupply [MWh]
    'vol_neg': 4384         # balancing volume at oversupply [MWh], negative
}
SERIES_FEATURES = {
    'load_fc': 411,         # day-ahead forecast of grid load [MWh]
    'gen_fc': 122,          # day-ahead forecast of total generation [MWh]
    'wind_on_fc': 123,      # day-ahead forecast of wind onshore [MWh]
    'wind_off_fc': 3791,    # day-ahead forecast of wind offshore [MWh]
    'pv_fc': 125            # day-ahead forecast of photovoltaics [MWh]
}
# name -> (data ID, region)
SERIES_NATIONAL = {
    'price_imb': (4382, 'DE'),          # balancing energy price [EUR/MWh]
    'price_da': (4169, 'DE-LU'),        # day-ahead wholesale price [EUR/MWh]
    'net_export_sched': (4629, 'DE')    # scheduled commercial net export [MWh]
}


def main():
    # national features and labels
    df_national = pd.DataFrame()
    for name in SERIES_FEATURES:
        df_national[name] = _read_series(name, 'DE', T_START_NATIONAL)
    for name, (_, region) in SERIES_NATIONAL.items():
        df_national[name] = _read_series(name, region, T_START_NATIONAL)
    _save(
        df_national,
        os.path.join(path_root_target, 'systemimbalance_national.csv')
    )

    # control zone features and labels
    df_zones_list = []
    for zone in ZONES:
        df_zone = pd.DataFrame()
        for name in {**SERIES_LABELS_ZONES, **SERIES_FEATURES}:
            df_zone[name] = _read_series(name, zone, T_START_ZONES)

        # signed system imbalance. Volumes at oversupply are already negative.
        df_zone['imbalance'] = df_zone[['vol_pos', 'vol_neg']].sum(
            axis=1, min_count=1
        )

        # day-ahead price is the same for all control zones
        df_zone['price_da'] = df_national['price_da']
        df_zone.insert(0, 'zone', zone)
        df_zones_list.append(df_zone)

    df_zones = pd.concat(df_zones_list)
    _save(df_zones, os.path.join(path_root_target, 'systemimbalance_zones.csv'))


def _read_series(name, region, t_start):
    """Read one raw series in quarter-hourly resolution."""
    index = pd.date_range(
        t_start, T_END, freq='15min', inclusive='left'
    ).tz_convert('UTC')

    path_raw = os.path.join(path_root_source, f'{name}_{region}.csv')
    if not os.path.exists(path_raw):
        # series not available for region, e.g. no wind offshore in zone
        return pd.Series(float('nan'), index=index)

    series = pd.read_csv(path_raw, index_col='Tmstamp')['value']
    series.index = pd.to_datetime(series.index, utc=True)

    return series.reindex(index)


def _save(df, path_save):
    """Save merged dataframe with UTC time stamps."""
    df.rename_axis('Tmstamp').to_csv(
        path_save, date_format='%Y-%m-%dT%H:%M:%SZ'
    )
    print(f'saved {path_save}')


if __name__ == '__main__':
    main()
    print("Successfully processed SystemImbalance data.")
