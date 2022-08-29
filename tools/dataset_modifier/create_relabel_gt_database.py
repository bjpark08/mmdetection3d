import pickle
from tools.data_converter.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from pathlib import Path

filename = 'rf2021_infos_train.pkl'
info_prefix = 'rf2021'
root_path = 'data/rf2021'
root_path = Path(root_path)

create_groundtruth_database('Custom3DDataset', root_path, info_prefix,
                                root_path / f'{info_prefix}_infos_train.pkl')
