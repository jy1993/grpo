import json
import torch
import random
import numpy as np
from torch.distributions import Categorical
import copy
import re
from data_utils import read_json, to_jsonl, latex_norm
import pandas as pd
import os
from transformers import StoppingCriteria

def extract_answer(response):
	# if ds_name == 'MetaMathQA':
	# 	search_result = re.findall('The answer is: -?[0-9.]+', response)
	# 	if len(search_result) == 1:
	# 		return search_result[0].split(': ')[1]
	# 	sqrt_result = re.findall(r'The answer is: -?\\sqrt{[0-9]+}', response)
	# 	if len(sqrt_result) == 1:
	# 		return sqrt_result[0].split(': ')[1]
	# 	frac_result = re.findall(r'The answer is: -?\\frac{[0-9]+}{[0-9pi]+}', response)
	# 	if len(frac_result) == 1:
	# 		return frac_result[0].split(': ')[1]
	# 	if 'frac{4}' in response or 'sqrt{5}' in response:
	# 		print(response)
	# 		print(search_result)
	# 		print(sqrt_result)
	# 		print(frac_result)
	# else:
	# 	search_result = re.findall('\\boxed\{[\\[a-z]\{\}0-9.]+\}', response)
	# 	if len(search_result) == 1:
	# 		return search_result.replace('\\boxed{', '')[:-1]
	if 'boxed' in response:
		search_result = re.findall(r'\boxed{((?:[^{}]|{[^{}]*})*)}', response)
		if len(search_result) > 0:
			return search_result[0]
	# else:
	# 	search_result = re.findall('The answer is: -?[0-9.]+', response)
	# 	if len(search_result) == 1:
	# 		return search_result[0].split(': ')[1]

def extract_answer_match(line):
	line = line.replace('\\', '')
	search_result = re.findall(r'boxed{((?:[^{}]|{[^{}]*})*)}', line)
	if len(search_result) > 0:
		return search_result[0]

class MathDataEngine(object):
	"""docstring for MathDataEngine"""
	def __init__(self, meta_math_path, numia_math_dir):
		super(MathDataEngine, self).__init__()
		meta_math = read_json(meta_math_path)
		df_numia_math = pd.concat([pd.read_parquet(os.path.join(numia_math_dir, path)) for path in os.listdir(numia_math_dir)])
		self.math_pool = self.preprocess(meta_math, df_numia_math)
		random.shuffle(self.math_pool)

	def preprocess(self, meta_math, df_numia_math):
		math_pool = []
		for sample in meta_math:
			boxed_ans = extract_answer(sample['response'])
			if boxed_ans and self.to_keep(sample['query'], boxed_ans):
				math_pool.append((sample['query'], sample['response'], boxed_ans))
		
		for m in df_numia_math['messages']:
			assert len(m) == 2
			boxed_ans = extract_answer(m[1]['content'])
			if boxed_ans and self.to_keep(m[0]['content'], boxed_ans):
				math_pool.append((m[0]['content'], m[1]['content'], boxed_ans))
		# print(df_numia_math.head())
		print(len(math_pool))
		return math_pool

	def vis(self):
		for sample in self.math_pool[:1000]:
			if len(re.sub('-?[0-9\.(),/\\sqrtfrac{}]+', '', sample[2])) > 0:
				print('*' * 10)
				print(sample[0])
				print(sample[2])

	def to_keep(self, query, ans):
		if 'prove' in query or 'Prove' in query or 'Show' in query or 'correct' in query:
			return False
		if 'text' in ans or 'image' in ans or 'The' in ans:
			return False
		# TODO: convert to mcqa to qa
		if ans in ['A', 'B', 'C', 'D', 'E', 'AB', 'AC', 'AD', 'AE', 'BC', 'BD', 'BE', 'CD', 'CE', 'DE', 'ABC', 'ABD', 'ABE', 'BCD', 'BCE', 'CDE', 'ABCD', 'BCDE', 'ABCDE']:
			return False
		if re.findall('[ABCD][)\.:]', ans):
			return False
		if '(I)' in query and '(II)' in query:
			return False
		if '(1)' in query and '(2)' in query:
			return False
		if '(Ⅰ)' in query and '(Ⅱ)' in query:
			return False
		return True

	def get_samples(self, n):
		samples = self.math_pool[self.pt:self.pt+n]
		self.pt += n
		return samples

	def split_into_n_files(self, n):
		size_per_file = len(self.math_pool) // n
		for i in range(n):
			to_jsonl([{'query': sample[0], 'answer': sample[2]} for sample in self.math_pool[i*size_per_file:i*size_per_file+size_per_file]], 'data/train/math_part_%s.json' % i)

class GRPOBuffer(object):
	"""docstring for GRPOBuffer"""
	def __init__(self, pad_token_id, eos_token_id, assistant_token_id, tokenizer, args, task_type='logic'):
		super(GRPOBuffer, self).__init__()
		self.pad_token_id = pad_token_id
		self.eos_token_id = eos_token_id
		self.assistant_token_id = assistant_token_id
		self.tokenizer = tokenizer
		self.args = args
		self.task_type = task_type
		self.reset()

	def add(self, group_token_ids, group_attention_mask, group_old_logps, group_ref_logps, group_response, gold_label):
		advs, rewards, answers = self.cal_advs(group_response, gold_label)
		self.print_rewards(group_response, gold_label, rewards, answers)
		# if self.args.completition_keep_ratio > 0:
		# 	sorted_abs_advs = sorted([abs(adv) for adv in advs])
		# 	keep_point = sorted_abs_advs[-int(len(advs) * self.args.completition_keep_ratio)]
		# else:
		# 	keep_point = 0
		# cnt = 0
		# for token_ids, attention_mask, old_logps, ref_logps, adv in zip(group_token_ids, group_attention_mask, group_old_logps, group_ref_logps, advs):
		# 	if abs(adv) >= keep_point and cnt < int(len(advs) * self.args.completition_keep_ratio):
		# 		self.all_token_ids.append(token_ids)
		# 		self.all_attention_mask.append(attention_mask)
		# 		self.all_old_logps.append(old_logps)
		# 		self.all_ref_logps.append(ref_logps)
		# 		self.all_advs.append(adv)
		# 		cnt += 1
		# assert cnt == int(len(advs) * self.args.completition_keep_ratio)
		for token_ids, attention_mask, old_logps, ref_logps, adv in zip(group_token_ids, group_attention_mask, group_old_logps, group_ref_logps, advs):
			self.all_token_ids.append(token_ids)
			self.all_attention_mask.append(attention_mask)
			self.all_old_logps.append(old_logps)
			self.all_ref_logps.append(ref_logps)
			self.all_advs.append(adv)

	def print_rewards(self, group_response, gold_label, rewards, answers):
		if self.args.debug and self.args.local_rank < 1:
			for r, rw, answer in zip(group_response, rewards, answers):
				print('*' * 20)
				print(r)
				print('gold_label: ', gold_label)
				print('answer: ', answer)
				print('rw:', rw)

	def reset(self):
		self.all_token_ids = []
		self.all_attention_mask = []
		self.all_old_logps = []
		self.all_ref_logps = []
		self.all_advs = []
		self.correct = 0
		self.total = 0
		self.correct_reward = 0
		self.format_reward = 0
		self.len_dist = []

	def get_stat(self):
		solve_rate = self.correct / self.total if self.total > 0 else 0
		avg_len = sum(self.len_dist) / len(self.len_dist) if len(self.len_dist) > 0 else 0
		max_len = max(self.len_dist) if len(self.len_dist) > 0 else 0
		avg_correct_reward = self.correct_reward / self.total / self.args.group_size if self.total > 0 else 0
		avg_format_reward = self.format_reward / self.total / self.args.group_size if self.total > 0 else 0
		return torch.tensor(solve_rate), torch.tensor(avg_len), torch.tensor(max_len), torch.tensor(avg_correct_reward), torch.tensor(avg_format_reward)

	def inter_shuffle(self, data, ids):
		da = []
		for i in ids:
			da += data[i*self.args.group_size:(i+1)*self.args.group_size]
		return da

	def shuffle(self):
		# inter-group shuffle
		ids = list(range(len(self.all_token_ids) // self.args.group_size))
		random.shuffle(ids)
		self.all_token_ids = self.inter_shuffle(self.all_token_ids, ids)
		self.all_attention_mask = self.inter_shuffle(self.all_attention_mask, ids)
		self.all_old_logps = self.inter_shuffle(self.all_old_logps, ids)
		self.all_ref_logps = self.inter_shuffle(self.all_ref_logps, ids)
		self.all_advs = self.inter_shuffle(self.all_advs, ids)

	def get_all_batches(self, batch_size):
		# self.shuffle()
		all_batches = []
		for i in range(len(self.all_token_ids) // batch_size):
			all_batches.append((torch.stack(self.all_token_ids[i*batch_size:i*batch_size+batch_size], dim=0), 
				torch.stack(self.all_attention_mask[i*batch_size:i*batch_size+batch_size], dim=0), 
				torch.stack(self.all_old_logps[i*batch_size:i*batch_size+batch_size], dim=0),
				torch.stack(self.all_ref_logps[i*batch_size:i*batch_size+batch_size], dim=0),
				torch.stack(self.all_advs[i*batch_size:i*batch_size+batch_size], dim=0)))
		return all_batches

	def normalize_answer(self, string):
		return string.replace(' ', '').replace('\n', '').replace('\\', '').replace('$', '').replace('boxed{', '').replace('}', '')

	def number_equal(self, num1, num2):
		try:
			return float(num1) == float(num2)
		except:
			return False

	def is_number(self, answer):
		try:
			n = float(answer)
			return True
		except:
			return False

	def extract_answer_from_model_response(self, response):
		answer = None
		if '<answer>' in response and '</answer>' in response:
			answer = response[response.index('<answer>'):response.index('</answer>')][8:]
		elif '<answer>' in response:
			answer = response[response.index('<answer>'):][8:]
		if answer:
			answer = str(answer)
		else:
			answer = self.extract_last_box(response)
		if self.task_type == 'k12':
			if answer.count('$') > 2:
				answer = self.extract_latex_from_answer(answer, '$')
			elif answer.count('=') >= 2:
				answer = self.extract_latex_from_answer(answer, '=')
		return answer

	def extract_last_box(self, response):
		box = re.findall('boxed{[0-9a-z\.\+\\\{\}\s-]+}', response)
		if box:
			return box[0][6:-1]
		return ''

	def extract_thinking_from_model_response(self, response):
		thinking = ''
		if '</think>' in response:
			thinking = response[:response.index('</think>')]
		return thinking

	def cal_rewards(self, response, gold_label):
		answer = self.extract_answer_from_model_response(response)
		answer = self.normalize_answer(answer)
		label = self.normalize_answer(gold_label)
		if '=' in answer:
			answer = answer.split('=')[-1]
		if self.number_equal(answer, label):
			reward = 1
			correct = True
		elif self.is_number(answer):
			reward = 0.3
			correct = False
		else:
			reward = 0.0
			correct = False
		format_reward = self.get_format_reward(response)
		self.correct_reward += reward
		self.format_reward += format_reward
		return reward + format_reward, correct, answer

	def extract_knights(self, answer):
		result = re.findall('[kK]nights:[a-zA-Z, ]+', answer)
		if result: 
			knights = result[0].split(':')[1].split(',')
			return [k.strip() for k in knights]
		return []

	def extract_knaves(self, answer):
		result = re.findall('[kK]naves:[a-zA-Z, ]+', answer)
		if result: 
			knaves = result[0].split(':')[1].split(',')
			return [k.strip() for k in knaves]
		return []

	def count_names_in_thinking(self, kv, thinking):
		cnt = 0
		for k, v in kv.items():
			if k.lower() in thinking.lower():
				cnt += 1
		return cnt

	def parse_model_answer(self, answer_text, expected_names):
		pred_dict = {}
		knight_count = answer_text.lower().count('knight')
		knave_count = answer_text.lower().count('knave')
		if knight_count + knave_count != len(expected_names):
			return None
		for name in expected_names:
			pattern = re.compile(
				rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
				re.IGNORECASE
			)
			match = pattern.search(answer_text)
			if match:
				role = match.group(1).lower()
				pred_dict[name] = role
			else:
				return None
		return pred_dict

	def cal_rewards_for_logic(self, response, gold_label):
		# thinking = self.extract_thinking_from_model_response(response)
		answer = self.extract_answer_from_model_response(response)
		gold_dict = json.loads(gold_label)
		pred_dict = self.parse_model_answer(answer, list(gold_dict.keys()))
		correct, reward = False, 0
		if pred_dict:
			if gold_dict == pred_dict:
				correct = True
				reward = 1
			# else:
			# 	reward = 0.2
		format_reward = self.get_format_reward(response)
		self.correct_reward += reward
		self.format_reward += format_reward
		return reward + format_reward, correct, answer

	def extract_latex_from_answer(self, response, symbol):
		index = [i for i, x in enumerate(response) if x == symbol]
		# extract latex from last $ pair
		if symbol == '$':
			return response[index[-2]:index[-1]]
		# extract latex from last equation
		elif symbol == '=':
			return response[index[-1]+1:]

	def number_equal(self, a, b):
		if a == b:
			return True
		try:
			if eval(a) == eval(b):
				return True
		except:
			return False

	def cal_rewards_for_k12(self, response, gold_label):
		answer = self.extract_answer_from_model_response(response)
		answer_norm = latex_norm(answer)
		correct, reward = False, 0
		if self.number_equal(answer_norm, gold_label):
			correct = True
			reward = 1
		else:
			for unit in ['cm', '°', 'm']:
				if gold_label.endswith(unit) and answer_norm.endswith(unit):
					if self.number_equal(answer_norm[:-len(unit)], gold_label[:-len(unit)]):
						correct = True
						reward = 1
					break
				else:
					if gold_label.endswith(unit) and self.number_equal(answer_norm, gold_label[:-len(unit)]):
						correct = True
						reward = 0.7
						break
					elif answer_norm.endswith(unit) and self.number_equal(answer_norm[:-len(unit)], gold_label):
						correct = True
						reward = 1
						break
		format_reward = self.get_format_reward(response)
		self.correct_reward += reward
		self.format_reward += format_reward
		return reward + format_reward, correct, answer

	def get_format_reward(self, response):
		format_reward = 0.0
		# if '</think>' in response and '<answer>' in response and '</answer>' in response:
		# 	if response.count('</think>') == 1 and response.count('<answer>') == 1 and response.count('</answer>') == 1:
		# 		format_reward = 0.3
		# 		think_end = response.index('</think>')
		# 		answer_start = response.index('<answer>')
		# 		answer_end = response.index('</answer>')
		# 		if think_end < answer_start < answer_end:
		# 			format_reward += 0.3
		# 			if 'boxed{' in response[answer_start:answer_end].replace('\\', '') and '}' in response[answer_start:answer_end]:
		# 				format_reward += 0.3
		if re.findall('</think>\s*<answer>[\s\S]*?</answer>', response):
			format_reward = 0.2
		else:
			format_reward = 0.0
		return format_reward

	def get_step_reward(self, response):
		step_reward = 0
		response = response.lower()
		count1 = response.count('firstly') + response.count('secondly') + response.count('finally')
		count2 = response.count('first') + response.count('second')
		count3 = response.count('step 1') + response.count('step 2') + response.count('step 3')
		count = max(count1, count2, count3)
		step_reward = min(1, count / 3)
		return step_reward

	def get_len_reward(self, group_response, all_correct):
		min_len = min(len(x) for x in group_response)
		max_len = max(len(x) for x in group_response)
		len_rewards = []
		if min_len == max_len:
			return [0] * len(group_response)
		for r, correct in zip(group_response, all_correct):
			len_reward = 0.5 - (len(r) - min_len) / (max_len - min_len)
			if correct:
				len_rewards.append(len_reward)
			else:
				len_rewards.append(min(0, len_reward))
		return len_rewards

	def cal_advs(self, group_response, gold_label):
		# print(len(group_response), gold_label)
		rewards, all_correct, all_answers = [], [], []
		for r in group_response:
			if self.task_type == 'logic':
				reward, correct, answer = self.cal_rewards_for_logic(r, gold_label)
			elif self.task_type == 'k12':
				reward, correct, answer = self.cal_rewards_for_k12(r, gold_label)
			rewards.append(reward)
			all_correct.append(correct)
			all_answers.append(answer)
			self.len_dist.append(len(r.split()))
		rewards = torch.tensor(rewards)
		# len_rewards = torch.tensor(self.get_len_reward(group_response, all_correct))
		mean = torch.mean(rewards)
		std = torch.std(rewards)
		# if std.item() > 0.01:
		# 	advs = (rewards - mean) / std
		if self.args.use_dr_grpo:
			advs = rewards - mean
		else:
			advs = (rewards - mean) / (std + 1e-6)
		# print(rewards)
		self.total += 1
		if any(all_correct):
			self.correct += 1
		return advs, rewards, all_answers

def print_rank0(args, x):
	if args.local_rank < 1:
		print(x)

def caculate_grpo_loss(args, logps, old_logps, ref_logps, advs, loss_mask, clip_range, kl_coeff):
	advs = advs.unsqueeze(-1)
	# ratio = torch.exp(logps - logps.detach())
	ratio = torch.exp(logps - old_logps)
	# clip = torch.max(- advs * ratio, - advs * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))
	clip = - torch.min(advs * ratio, advs * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))
	# policy_loss = (- advs * ratio * loss_mask).sum() / loss_mask.sum()
	if args.use_dr_grpo:
		policy_loss = (clip * loss_mask).sum() / args.max_seq_len
	else:
		policy_loss = (clip * loss_mask).sum() / loss_mask.sum()
	kl_loss = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
	if args.use_dr_grpo:
		kl_loss = (kl_loss * loss_mask).sum() / args.max_seq_len
	else:
		kl_loss = (kl_loss * loss_mask).sum() / loss_mask.sum()
	kl_loss = torch.clamp(kl_loss, max=200)
	# print_rank0(args, [i for i, m in zip(logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
	# print_rank0(args, [i for i, m in zip(old_logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
	# print_rank0(args, [i for i, m in zip(ref_logps[0].tolist(), loss_mask[0].tolist()) if m == 1])
	# print_rank0(args, advs[0].tolist())
	return policy_loss + kl_coeff * kl_loss, policy_loss, kl_loss

def gather_logps(logits, ids, shift=True):
	# logps: bsz, seq_len, vocab_size
	# ids: bsz, seq_len
	# if shift:
	# 	logits = logits[:, :-1]
	# 	ids = ids[:, 1:]
	return torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
	# selected_logits = torch.gather(logits, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
	# # loop to reduce peak mem consumption
	# logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
	# per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
	# return torch.clamp(per_token_logps, min=-10)

class RepeatTokenStoppingCriteria(StoppingCriteria):
	def __init__(self, max_repeat: int = 5):
		self.max_repeat = max_repeat

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		if input_ids.shape[1] < self.max_repeat:
			return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
		last_tokens = input_ids[:, -self.max_repeat:]
		done_list = []
		for l in last_tokens:
			done_list.append(torch.all(l==l[0]).item())
		return torch.BoolTensor(done_list).to(input_ids.device)

class EarlyStoppingCrieria(StoppingCriteria):
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		done_list = []
		for one in input_ids:
			done_list.append(self.check_response(self.tokenizer.decode(one.tolist())))
		return torch.BoolTensor(done_list).to(input_ids.device)

	def check_response(self, answer):
		answer = answer[answer.index('<|im_start|>assistant'):].replace('<|im_end|>', '')
		if '</think>' in answer:
			think_end_index = answer.index('</think>')
			if '<answer>' in answer:
				answer_start_index = answer.index('<answer>')
			else:
				answer_start_index = len(answer)
			if answer_start_index - think_end_index > 20:
				return True
		if '</answer>' in answer:
			answer_end_index = answer.index('</answer>')
			if len(answer) - answer_end_index > 10:
				return True
		words = answer.split()
		if len(re.findall('[.,:;!?]', ''.join(words[-100:]))) == 0 and len(words) > 100:
			return True
		for word in words:
			# extremely long word
			if len(word) > 100 and re.sub('[a-zA-Z]', '', word) == '':
				return True
		return False