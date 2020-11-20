# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--data_dir="G:\Work Related\Nlc2cmd_t5\seq2seq\Bash_Data\T5_Data" \
--learning_rate=3e-5 \
--train_batch_size=$BS \
--eval_batch_size=$BS \
--task="translation"\
--output_dir="G:\Work Related\Nlc2cmd_t5\seq2seq\Bash_Data\Output" \
--max_source_length=512 \
--val_metric="bleu"\
--max_target_length=256 \
--val_check_interval=0.1 --n_val=200 \
--save_top_k = 2\
--do_train --do_predict \
 "$@"
