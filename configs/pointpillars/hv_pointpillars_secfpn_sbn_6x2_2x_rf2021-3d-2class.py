_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_rf2021_mvxfasterrcnn.py',    
    '../_base_/datasets/rf2021-3d-2class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py',
]