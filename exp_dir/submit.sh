BASE_DIR=/lustre/orion/stf219/scratch/souzar/cluster_experiment_utils
PARAMS_FILE=$BASE_DIR/exp_dir/conf/llm_exp_params.yaml
which python
#echo "Remember: to use submit.sh, you cannot have an env activated. Use the base env"
echo "If you want to reuse a mongo instance, remember to run start_mongo.sh if mongo is not up yet."
python $BASE_DIR/executors/submit_batch_job.py --conf $PARAMS_FILE
#sleep 10
#watch bjobs
watch squeue -u souzar
