BASE_DIR=/lustre/orion/stf219/scratch/souzar/cluster_experiment_utils
PARAMS_FILE=$BASE_DIR/exp_dir/conf/llm_exp_params.yaml
FLOWCEPT_CONF=$BASE_DIR/exp_dir/conf/flowcept_settings_test.yaml

python $BASE_DIR/executors/start_local_mongo.py --exp_conf $PARAMS_FILE --flowcept_conf $FLOWCEPT_CONF
