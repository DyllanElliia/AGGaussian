source_path=$1
output_path=$2
lambda_smooth=$3
lambda_contrast=$4
graph_tau=$5
gpu_id=$6
echo "Input path: ${source_path}"
echo "Output path: ${output_path}"
echo "GPU ID: ${gpu_id}"


CUDA_VISIBLE_DEVICES=${gpu_id} python train_ag2.py -s ${source_path} -m ${output_path} --port 808${gpu_id} --graph_tau ${graph_tau} --lambda_m_smooth ${lambda_smooth} --lambda_m_contrast ${lambda_contrast} --use_free_scale 1
CUDA_VISIBLE_DEVICES=${gpu_id} python render.py -m ${output_path} --iteration 35000