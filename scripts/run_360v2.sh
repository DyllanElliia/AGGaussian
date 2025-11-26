gpu_id=1
# source_path output_path lambda_smooth lambda_contrast graph_tau gpu_id
sh scripts/run_one_4.sh ./datasets/360_v2/bicycle ./outputs/360_v2/bicycle 2.5 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/bicycle --scene_name bicycle

sh scripts/run_one_4.sh ./datasets/360_v2/kitchen ./outputs/360_v2/kitchen 3.0 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/kitchen --scene_name kitchen

sh scripts/run_one_4.sh ./datasets/360_v2/counter ./outputs/360_v2/counter 3.0 0.5 0.1 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/counter --scene_name counter

sh scripts/run_one_4.sh ./datasets/360_v2/bonsai ./outputs/360_v2/bonsai 3.0 0.5 0.1 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/bonsai --scene_name bonsai

sh scripts/run_one_4.sh ./datasets/360_v2/garden ./outputs/360_v2/garden 3.0 0.5 0.1 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/garden --scene_name garden

sh scripts/run_one_4.sh ./datasets/360_v2/room ./outputs/360_v2/room 3.0 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/360_v2/room --scene_name room
