#Infer parameters of a mock spectrum
python prospectr.py --param_file=demo_mock_params.py
#Parallel version of the above
mpirun -np 4 python prospectr.py --param_file=demo_mock_params.py
#Infer SSP parameters from real data using a json model descriptor
python prospectr.py --param_file=demo_cluster_params.json
#Infer SSP parameters from real data using a script model descriptor
python prospectr.py --param_file=demo_cluster_params.py
#Infer CSP parameters from real (photometric) data using a script model descriptor
python prospectr.py --param_file=demo_csphot_params.py --sptype=fsps --zcontinuous=true --custom_filter_keys=filter_keys_threedhst.txt
#Parallel version of above
mpirun -np 4 python prospectr.py --param_file=demo_csphot_params.py --sptype=fsps --zcontinuous=true --custom_filter_keys=filter_keys_threedhst.txt
