_base_ = [
    '../_base_/models/centerpoint_02voxel_second_secfpn_rf_2021.py',
    '../_base_/datasets/rf2021-3d-2class.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py',
]