gpu_id=2

# source_path output_path lambda_smooth lambda_contrast graph_tau gpu_id
sh scripts/run_one.sh ./datasets/lerf/figurines ./outputs/lerf/figurines 3.0 0.3 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/lerf/figurines --scene_name figurines

sh scripts/run_one.sh ./datasets/lerf/teatime ./outputs/lerf/teatime 3.0 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/lerf/teatime --scene_name teatime

sh scripts/run_one.sh ./datasets/lerf/waldo_kitchen ./outputs/lerf/waldo_kitchen 3.0 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/lerf/waldo_kitchen --scene_name waldo_kitchen

# check
sh scripts/run_one.sh ./datasets/lerf/ramen ./outputs/lerf/ramen 3.0 0.5 0.05 ${gpu_id}
CUDA_VISIBLE_DEVICES=${gpu_id} python render_text_mask.py -m ./outputs/lerf/ramen --scene_name ramen