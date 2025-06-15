import random
import torch
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm, trange
from grpo_utils import GRPOBuffer, caculate_grpo_loss, gather_logps, RepeatTokenStoppingCriteria, EarlyStoppingCrieria
from data_utils import read_jsonl
from llm_utils import get_train_ds_config
import os
from copy import deepcopy
from pprint import pprint
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, AdamW, StoppingCriteriaList
from args import get_grpo_args
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from vllm import LLM, SamplingParams
from unittest.mock import patch
from PIL import Image

def get_all_reduce_mean(tensor):
	torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
	tensor = tensor / torch.distributed.get_world_size()
	return tensor

@torch.no_grad()
def gather_object(object):
	torch.distributed.barrier()
	output_objects = [None for _ in range(8)]
	torch.distributed.all_gather_object(output_objects, object)
	# all_gather_object returns a list of lists, so we need to flatten it
	return [x for y in output_objects for x in y]

@torch.no_grad()
def broadcast_object_list(object_list, src=0):
	torch.distributed.barrier()
	torch.distributed.broadcast_object_list(object_list, src=src)
	return object_list

def safe_load_image(img_path):
	with open(img_path, "rb") as f:
		img = Image.open(f)
		img.load()
	return img

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, args):
		super(Trainer, self).__init__()
		self.args = args
		policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
		if self.args.kl_coeff > 0:
			ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
		else:
			ref_model = None
		self.processor = AutoProcessor.from_pretrained(args.model_path)
		self.processor.tokenizer.padding_side = 'left'
		self.data_buffer = GRPOBuffer(pad_token_id=self.processor.tokenizer.pad_token_id, eos_token_id=self.processor.tokenizer.eos_token_id, assistant_token_id=77091, tokenizer=self.processor.tokenizer, args=self.args, task_type='k12')
		# self.data_engineer = MathDataEngineer()
		self.init_ds(policy_model, ref_model)
		self.onload_or_offload('cpu', 'cpu')
		assert self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps % self.args.group_size == 0
		self.math_pool = read_jsonl(self.args.data_path_prefix + str(self.args.global_rank) + '.json')
		self.sampling_params = SamplingParams(
			n=self.args.group_size,
			temperature=self.args.temperature,
			top_p=self.args.topp,
			max_tokens=self.args.max_seq_len,
			stop_token_ids=[151643, 151645],
			detokenize=False
		)
		if self.args.global_rank == 0:
			self.writer = SummaryWriter(args.tensorboard_path)
			with patch("torch.distributed.get_world_size", return_value=self.args.ngpus_for_vllm):
				self.llm = LLM(
					model=self.args.model_path,
					tensor_parallel_size=self.args.ngpus_for_vllm,
					max_model_len=self.args.max_seq_len,
					gpu_memory_utilization=0.6,
					enforce_eager=False,
					enable_sleep_mode=True,
					mm_processor_kwargs={
						"min_pixels": 20 * 28 * 28,
						"max_pixels": 100 * 28 * 28,
						"fps": 1,
					},
				)
		torch.distributed.barrier()

	def onload_or_offload(self, ref_device=None, device=None):
		if ref_device is not None and self.ref_model_engine is not None:
			self.ref_model_engine.to(ref_device)
		if device is not None:
			self.model_engine.to(device)
	
	def init_ds(self, policy_model, ref_model):
		torch.cuda.set_device(self.args.local_rank)
		self.args.device = torch.device('cuda', self.args.local_rank)
		deepspeed.init_distributed()
		self.args.global_rank = torch.distributed.get_rank()
		self.args.n_gpus = torch.distributed.get_world_size()
		torch.distributed.barrier()
		no_decay = ["bias", "norm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in policy_model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.args.weight_decay, 
				"lr": args.lr
			},     
			{   
				"params": [p for n, p in policy_model.named_parameters() if any(nd in n for nd in no_decay)], 
				"weight_decay": 0.0, 
				"lr": self.args.lr
			}           
		]
		AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
		optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=self.args.lr, betas=(0.9, 0.95))

		# lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.total_steps)
		lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps)
		ds_config = get_train_ds_config(offload=args.offload, 
			stage=self.args.zero_stage, 
			global_batch_size=self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps*self.args.n_gpus,
			micro_batch_size=self.args.per_device_train_batch_size,
			grad_acc=self.args.gradient_accumulation_steps,
			bf16=self.args.bf16,
			job_name=self.args.exp_name)
		model_engine, _, _, _ = deepspeed.initialize(
			model=policy_model,
			optimizer=optimizer,
			args=self.args,
			config=ds_config,
			lr_scheduler=lr_scheduler,
			dist_init_required=True)
		if self.args.gradient_checkpointing:
			model_engine.gradient_checkpointing_enable()
		self.model_engine = model_engine
		if ref_model is not None:
			ref_model_engine = deepspeed.init_inference(
				model=ref_model, config={"dtype": 'bfloat16', "replace_with_kernel_inject": True})
			self.ref_model_engine = ref_model_engine
		else:
			self.ref_model_engine = None

	def generate_sequences(self, batch_prompts):
		outputs = self.llm.generate(batch_prompts, self.sampling_params)
		all_token_ids = []
		for output in outputs:
			for one in output.outputs:
				all_token_ids.append(list(one.token_ids))
		max_len = max(len(token_ids) for token_ids in all_token_ids)
		seqs = []
		for token_ids in all_token_ids:
			seqs.append(token_ids + [self.tokenizer.eos_token_id] * (max_len - len(token_ids)))
		return seqs

	def compute_logps(self, model, input_ids, attention_mask, logits_to_keep, offload=False):
		logits = model.forward(input_ids, attention_mask=attention_mask, use_cache=False).logits
		if offload:
			logits = logits.cpu()
			input_ids = input_ids.cpu()
		logits = logits[:, :-1]
		logits = logits[:, -logits_to_keep:]
		logps = gather_logps(logits + 1e-7, input_ids[:, -logits_to_keep:])
		return logps

	def batch_compute_logps(self, model, input_ids, attention_mask, logits_to_keep):
		logps = []
		for i in range(max(1, len(input_ids) // self.args.per_device_playout_batch_size)):
			logits = self.model_engine.forward(input_ids[i*self.args.per_device_playout_batch_size:(i+1)*self.args.per_device_playout_batch_size], attention_mask=attention_mask[i*self.args.per_device_playout_batch_size:(i+1)*self.args.per_device_playout_batch_size], use_cache=False).logits
			logits = logits[:, :-1]
			logits = logits[:, -logits_to_keep:]
			logps.append(gather_logps(logits + 1e-7, input_ids[i*self.args.per_device_playout_batch_size:(i+1)*self.args.per_device_playout_batch_size, -logits_to_keep:]))
			del logits
			torch.cuda.empty_cache()
		logps = torch.cat(logps, dim=0)
		return logps

	def generate_experiences(self, seqs, attention_mask, logits_to_keep):
		with torch.no_grad():
			self.onload_or_offload('cpu', 'cuda')
			logps = self.batch_compute_logps(self.model_engine, seqs, attention_mask, logits_to_keep)
			if self.ref_model_engine is not None:
				self.onload_or_offload('cuda', 'cpu')
				ref_logps = self.batch_compute_logps(self.ref_model_engine, seqs, attention_mask, logits_to_keep)
				self.onload_or_offload('cpu', 'cuda')
			else:
				ref_logps = logps
		self.model_engine.train()
		return logps, ref_logps

	def get_attention_mask(self, seqs, prompt_len):
		attention_mask = []
		for seq in seqs.tolist():
			start, end = self.get_start_and_end_index(seq, prompt_len)
			mask = [0] * start + [1] * (end - start + 1) + [0] * (len(seq) - end - 1)
			attention_mask.append(mask)
			# if self.args.local_rank == 0:
			# 	print(start, end, len(seq), len(mask))
			# 	print(seq, mask)
		attention_mask = torch.LongTensor(attention_mask).to(seqs.device)
		assert attention_mask.shape == seqs.shape
		return attention_mask

	def get_start_and_end_index(self, token_ids, prompt_len):
		start, end = 0, len(token_ids) - 1
		for idx, token_id in enumerate(token_ids):
			if token_id != self.tokenizer.pad_token_id:
				start = idx
				break
		for idx, token_id in enumerate(token_ids):
			if token_id == self.tokenizer.eos_token_id and idx > prompt_len:
				end = idx
				break
		return start, end

	def playout(self, samples, is_first):
		if not is_first:
			self.onload_or_offload(None, 'cpu')
			if self.args.global_rank == 0:
				torch.cuda.empty_cache()
				sd = self.model_engine.module.state_dict() if hasattr(self.model_engine, 'module') else self.model_engine.state_dict()
				self.llm.wake_up()
				llm_model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
				# for k, v in sd.items():
				# 	print(k, v[0])
				llm_model.load_weights((name, param) for name, param in sd.items())
				del sd
		all_texts, gold_labels = [], []
		# system_prompt = (
		# 	"You are a math expert to provide help for the user. ",
		# 	"You should think about the reasoning process in the mind and then provides the user with the answer. "
		# 	"Your response should only include two parts: the reasoning process and the answer. "
		# 	"In the reasoning part, check the correctness of each step in your solution, if anything goes wrong, reflect your think process and try new methods. "
		# 	"The answer part should be short and only need to contain the final result. "
		# 	"The reasoning process and answer are enclosed with <think> </think> and <answer> </answer> tags respectively,i.e.,<think> reasoning process here </think> <answer> number or expression here </answer>."
		# 	)
		system_prompt = 'You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.'
		for sample in samples:
			prompt = """<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>%s\n<|im_end|>\n<|im_start|>assistant\n<think>""" % sample['query']
			all_texts.append({'prompt': prompt, "multi_modal_data": {"image": safe_load_image('/mnt/data/R25039/workspace/R1-20250214/' + sample['image'])}})	
			gold_labels.append(sample['answer'])

		if self.args.global_rank < 1:
			print(self.args.global_rank, all_texts[0])
			# print(len(all_text))

		all_prompts = gather_object(all_texts)
		if self.args.global_rank == 0:
			seqs = self.generate_sequences(all_prompts)
			self.llm.sleep(level=1)
			torch.cuda.empty_cache()
		else:
			seqs = [None] * len(samples) * self.args.group_size * self.args.n_gpus
		seqs = broadcast_object_list(seqs)
		# local_prompts = all_prompts[self.args.local_rank * len(samples)*self.args.group_size:(self.args.local_rank+1) * len(samples)*self.args.group_size]
		# local_seqs = seqs[self.args.local_rank * len(samples)*self.args.group_size:(self.args.local_rank+1) * len(samples)*self.args.group_size]
		# model_inputs = self.tokenizer(all_text, padding=True, truncation=True, return_tensors="pt")
		# input_ids = model_inputs['input_ids'].repeat_interleave(self.args.group_size, dim=0)
		# local_seqs = torch.cat([input_ids, torch.LongTensor(local_seqs)], dim=-1).to(self.args.device)
		# prompt_len = model_inputs['input_ids'].shape[1]
		# logits_to_keep = local_seqs.shape[1] - prompt_len
		# attention_mask = self.get_attention_mask(local_seqs, prompt_len)
		# logps, ref_logps = self.generate_experiences(local_seqs, attention_mask, logits_to_keep)
		# generated_ids = [output_ids[prompt_len:] for output_ids in local_seqs.tolist()]
		# response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		# cnt = 0
		# for jdx in range(len(local_seqs) // self.args.group_size):
		# 	start = jdx * self.args.group_size
		# 	end = start + self.args.group_size
		# 	self.data_buffer.add(local_seqs[start:end], attention_mask[start:end], logps[start:end, prompt_len-1:], ref_logps[start:end, prompt_len-1:], 
		# 		response[start:end], gold_labels[cnt])
		# 	cnt += 1
		model_inputs = self.tokenizer(all_prompts, padding=True, truncation=True, return_tensors="pt")
		local_input_ids = model_inputs['input_ids'][self.args.local_rank*len(samples):(self.args.local_rank+1) * len(samples)].repeat_interleave(self.args.group_size, dim=0)
		local_seqs = seqs[self.args.local_rank * len(samples)*self.args.group_size:(self.args.local_rank+1) * len(samples)*self.args.group_size]
		prompt_len = model_inputs['input_ids'].shape[1]
		logits_to_keep = len(seqs[0])
		local_whole = torch.cat([local_input_ids, torch.LongTensor(local_seqs)], dim=-1).to(self.args.device)
		attention_mask = self.get_attention_mask(local_whole, prompt_len)
		logps, ref_logps = self.generate_experiences(local_whole, attention_mask, logits_to_keep)
		response = self.tokenizer.batch_decode(local_whole[:, -logits_to_keep:], skip_special_tokens=True)
		if self.args.global_rank < 1:
			print(local_whole.shape)
			print(len(response))
		cnt = 0
		for jdx in range(len(local_whole) // self.args.group_size):
			start = jdx * self.args.group_size
			end = start + self.args.group_size
			self.data_buffer.add(local_whole[start:end], attention_mask[start:end], logps[start:end], ref_logps[start:end], 
				response[start:end], gold_labels[cnt])
			cnt += 1
		assert cnt == len(samples)
		return logits_to_keep

	def get_samples(self, n):
		start = 0
		all_samples = []
		random.shuffle(self.math_pool)
		while start + n < len(self.math_pool):
			all_samples.append(self.math_pool[start:start+n])
			start += n
		return all_samples

	def train(self):
		global_step = 0
		is_first = True
		for _ in range(self.args.epochs):
			all_samples = self.get_samples(self.args.per_device_playout_batch_size)
			for samples in all_samples:
				logits_to_keep = self.playout(samples, is_first)
				is_first = False
				solve_rate, avg_len, max_len, avg_correct_reward, avg_format_reward = self.data_buffer.get_stat()
				solve_rate = get_all_reduce_mean(solve_rate.to(self.args.device)).item()
				avg_len = get_all_reduce_mean(avg_len.to(self.args.device)).item()
				max_len = get_all_reduce_mean(max_len.to(self.args.device)).item()
				avg_correct_reward = get_all_reduce_mean(avg_correct_reward.to(self.args.device)).item()
				avg_format_reward = get_all_reduce_mean(avg_format_reward.to(self.args.device)).item()
				if self.args.global_rank == 0:
					self.writer.add_scalars('Buffer/reward', {'solve_rate': solve_rate, 
						'avg_correct_reward': avg_correct_reward, 
						'avg_format_reward': avg_format_reward}, global_step)
					self.writer.add_scalars('Buffer/response_length', 
						{'avg_len': avg_len, 'max_len': max_len, 'size': len(self.data_buffer.all_advs)}, 
						global_step)
				torch.distributed.barrier()
				if True:
					step = 0
					for batch in self.data_buffer.get_all_batches(self.args.per_device_train_batch_size):
						# if self.args.global_rank == 0:
						# 	for t in batch:
						# 		print(t.shape)
						batch = tuple(t.to(self.args.device) for t in batch)
						logps = self.compute_logps(self.model_engine, batch[0], batch[1], logits_to_keep)
						loss, policy_loss, kl_loss = caculate_grpo_loss(self.args, logps, batch[2], batch[3], batch[4], batch[1][:, -logits_to_keep:], self.args.clip_range, self.args.kl_coeff)
						# optimizer.zero_grad()
						self.model_engine.backward(loss)
						# torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), self.args.clip_grad_norm)
						self.model_engine.step()
						step += 1
						loss_mean = get_all_reduce_mean(loss).item()
						policy_loss_mean = get_all_reduce_mean(policy_loss).item()
						kl_loss_mean = get_all_reduce_mean(kl_loss).item()
						if self.args.global_rank == 0 and step % self.args.gradient_accumulation_steps == 0:
							self.writer.add_scalars('Train', {'loss': loss_mean, 'policy_loss': policy_loss_mean, 'kl_loss': kl_loss_mean}, global_step)
							self.writer.add_scalar('Train/lr', self.model_engine.optimizer.optimizer.param_groups[0]['lr'], global_step)
							print('global_step: %s, loss: %.3f' % (global_step, loss_mean))
							global_step += 1
						
						if global_step % args.save_steps == 0 and global_step > 0 and self.args.global_rank == 0 and step % self.args.gradient_accumulation_steps == 0:
							print('saving model: %s' % global_step)
							self.save_model(global_step)
				self.data_buffer.reset()
				torch.cuda.empty_cache()
				torch.distributed.barrier()
		if self.args.global_rank == 0:
			self.save_model(global_step)
			print('saving final model: %s' % global_step)

	def save_model(self, global_step):
		os.makedirs(os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step), exist_ok=True)
		sd = self.model_engine.module.state_dict() if hasattr(self.model_engine, 'module') else self.model_engine.state_dict()
		torch.save(sd, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step, 'pytorch_model.bin'))
		os.system('cp %s/config.json %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))
		os.system('cp %s/merges.txt %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))
		os.system('cp %s/vocab.json %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))
		os.system('cp %s/generation_config.json %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))
		os.system('cp %s/tokenizer.json %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))
		os.system('cp %s/tokenizer_config.json %s' % (self.args.model_path, os.path.join(self.args.output_dir, 'checkpoint_%s' % global_step)))

if __name__ == '__main__':
	args = get_grpo_args()
	trainer = Trainer(args)
	trainer.train()