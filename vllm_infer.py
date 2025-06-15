from vllm import LLM, SamplingParams
from data_utils import read_jsonl, read_json, to_json
from args import ArgumentParser
import re
import torch

def parse_model_answer(answer_text, expected_names):
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

def judge_response(res, label):
	if '<answer>' in res and '</answer>' in res:
		answer = res[res.index('<answer>'):res.index('</answer>')][8:]
	else:
		answer = ''
	pred_dict = parse_model_answer(answer, list(label.keys()))
	if pred_dict == label:
		return True
	return False

parser = ArgumentParser()
parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--file_type', type=str, default='jsonl')
args = parser.parse_args()

if __name__ == '__main__':
	new_prompts, all_querys, gold_labels = [], [], []
	if args.file_type == 'jsonl':
		math_pool = read_jsonl(args.test_path)
		for row in math_pool:
			new_prompt = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) ...\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n%s\n<|im_end|>\n<|im_start|>assistant\n<think>""" % row['quiz']
			new_prompts.append(new_prompt)
			all_querys.append(row['quiz'])
			gold_labels.append({n: 'knight' if s else 'knave' for n, s in zip(row['names'], row['solution'])})
	else:
		math_pool = read_json(args.test_path)
		for row in math_pool['all_wrong']:
			new_prompt = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) ...\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n%s\n<|im_end|>\n<|im_start|>assistant\n<think>""" % row['query']
			new_prompts.append(new_prompt)
			all_querys.append(row['query'])
			gold_labels.append(row['label'])

	llm = LLM(
		model=args.model_path,
		tensor_parallel_size=torch.cuda.device_count(),
		max_model_len=2048,
		gpu_memory_utilization=0.85,
		enforce_eager=False,
		enable_prefix_caching=True
	)
	sampling_params = SamplingParams(
		n=args.n,
		temperature=1,
		top_p=0.95,
		max_tokens=2048,
	)
	outputs = llm.generate(new_prompts, sampling_params)
	# responses = [output.outputs[0].text for output in outputs]
	print(len(outputs), len(all_querys), len(gold_labels), len(new_prompts))

	result = {'all_correct': [], 'any_correct': [], 'all_wrong': []}
	cor = 0
	for query, label, output in zip(all_querys, gold_labels, outputs):
		group_responses = [o.text for o in output.outputs]
		group_judege = [judge_response(res, label) for res in group_responses] 
		if all(group_judege):
			result['all_correct'].append({'query': query, 'label': label, 'group_responses': group_responses, 'group_judege': group_judege})
		elif any(group_judege):
			result['any_correct'].append({'query': query, 'label': label, 'group_responses': group_responses, 'group_judege': group_judege})
		else:
			result['all_wrong'].append({'query': query, 'label': label, 'group_responses': group_responses, 'group_judege': group_judege})
		cor += group_judege.count(True) 
	print('test filename: %s' % args.test_path)
	print('acc: %.3f' % (cor / len(all_querys) / args.n))
	if args.n > 1:
		print(len(result['all_correct']), len(result['any_correct']), len(result['all_wrong']))
	if args.result_path:
		print('saving to: %s', args.result_path)
		to_json(result, args.result_path)
