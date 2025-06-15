from argparse import ArgumentParser
import deepspeed

def get_grpo_args():
	parser = ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--model_path', type=str, default=None)
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--lr', type=float, default=3e-6)
	parser.add_argument('--warmup_steps', type=int, default=100)
	parser.add_argument('--total_steps', type=int, default=3000)
	parser.add_argument('--data_path', type=str, default=None)
	parser.add_argument('--data_path_prefix', type=str, default='data/train/math_part_')
	parser.add_argument('--test_path', type=str, default=None)
	parser.add_argument('--per_device_train_batch_size', type=int, default=2)
	parser.add_argument('--eval_batch_size', type=int, default=16)
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
	parser.add_argument('--save_steps', type=int, default=10000)
	parser.add_argument('--eval_steps', type=int, default=50000)
	parser.add_argument('--output_dir', type=str, default=None)
	parser.add_argument('--weight_decay', type=float, default=0.1)
	parser.add_argument('--fp16', action='store_true')
	parser.add_argument('--bf16', action='store_true')
	parser.add_argument('--clip_grad_norm', type=float, default=1.0)
	parser.add_argument('--exp_name', type=str, default=None)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--offload', action='store_true')
	parser.add_argument('--zero_stage', type=int, default=0)
	parser.add_argument('--gradient_checkpointing', action='store_true')
	parser.add_argument('--local_rank', type=int, default=-1)
	parser.add_argument('--intra_doc_causal_mask', action='store_true')
	parser.add_argument('--window_size', type=int, default=500)
	# grpo
	parser.add_argument('--max_seq_len', type=int, default=2000)
	parser.add_argument('--clip_range', type=float, default=0.2)
	parser.add_argument('--kl_coeff', type=float, default=0.04)
	parser.add_argument('--use_dr_grpo', action='store_true')
	parser.add_argument('--completition_keep_ratio', type=float, default=0.25)
	# vllm
	parser.add_argument('--ngpus_for_vllm', type=int, default=1)
	# playout
	parser.add_argument('--per_device_playout_batch_size', type=int, default=4)
	parser.add_argument('--max_new_tokens', type=int, default=3600)
	parser.add_argument('--temperature', type=float, default=1)
	parser.add_argument('--topp', type=float, default=0.95)
	parser.add_argument('--group_size', type=int, default=8)
	# tensorboard
	parser.add_argument('--tensorboard_path', type=str, default=None)
	# debug
	parser.add_argument('--debug', action='store_true')
	# vl
	parser.add_argument('--min_vl_tokens', type=int, default=20)
	parser.add_argument('--max_vl_tokens', type=int, default=100)
	parser = deepspeed.add_config_arguments(parser)
	args = parser.parse_args()
	return args