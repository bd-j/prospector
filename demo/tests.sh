python prospectr.py --param_file=demo_mock_params.py
mpirun -np 4 python propspectr.py --param_file=demo_mock_params.py
python prospectr.py --param_file=demo_cluster_params.json
python prospectr.py --param_file=demo_cluster_params.py
python prospectr.py --param_file=demo_csphot_params.py --sps=fsps --zcontinuous=true --custom_filter_keys=filter_keys_threedhst.txt
mpirun -np 4 python prospectr.py --param_file=demo_csphot_params.py --sps=fsps --zcontinuous=true --custom_filter_keys=filter_keys_threedhst.txt
