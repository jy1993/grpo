from vllm import LLM, SamplingParams
from data_utils import read_jsonl, read_json, to_json, latex_norm
from args import ArgumentParser
import re
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def judge_response(res, label):
	if '<answer>' in res and '</answer>' in res:
		answer = res[res.index('<answer>'):res.index('</answer>')][8:]
	else:
		answer = ''
	answer_norm = latex_norm(answer)
	if answer_norm == label:
		return True
	return False

def safe_load_image(img_path):
	with open(img_path, "rb") as f:
		img = Image.open(f)
		img.load()
	return img

parser = ArgumentParser()
parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--file_type', type=str, default='jsonl')
args = parser.parse_args()

if __name__ == '__main__':
	new_prompts, all_querys, all_images, gold_labels = [], [], [], []
	if args.file_type == 'jsonl':
		math_pool = read_jsonl(args.test_path)
	else:
		math_pool = read_json(args.test_path)['all_wrong']
	for row in math_pool:
		prompt = """<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>%s\n<|im_end|>\n<|im_start|>assistant\n<think>""" % row['query']
		# new_prompts.append({'prompt': prompt, "multi_modal_data": {"image": Image.open('/mnt/data/R25039/workspace/R1-20250214/' + row['image'])}})
		new_prompts.append({'prompt': prompt, "multi_modal_data": {"image": safe_load_image('/mnt/data/R25039/workspace/R1-20250214/' + row['image'])}})	
		all_querys.append(row['query'])
		all_images.append(row['image'])
		gold_labels.append(row['answer'])

	llm = LLM(
		model=args.model_path,
		tensor_parallel_size=torch.cuda.device_count(),
		max_model_len=2048,
		gpu_memory_utilization=0.85,
		enforce_eager=False,
		enable_prefix_caching=True,
		mm_processor_kwargs={
			"min_pixels": 20 * 28 * 28,
			"max_pixels": 100 * 28 * 28,
			"fps": 1,
		},
	)
	sampling_params = SamplingParams(
		n=args.n,
		temperature=1,
		top_p=0.95,
		max_tokens=2048,
	)
	outputs = llm.generate(new_prompts, sampling_params)
	# responses = [output.outputs[0].text for output in outputs]
	print(len(outputs), len(all_querys), len(gold_labels), len(all_images), len(new_prompts))

	result = {'all_correct': [], 'any_correct': [], 'all_wrong': []}
	cor = 0
	for query, label, image, output in zip(all_querys, gold_labels, all_images, outputs):
		group_responses = [o.text for o in output.outputs]
		group_judege = [judge_response(res, label) for res in group_responses] 
		if all(group_judege):
			result['all_correct'].append({'query': query, 'label': label, 'image': image, 'group_responses': group_responses, 'group_judege': group_judege})
		elif any(group_judege):
			result['any_correct'].append({'query': query, 'label': label, 'image': image, 'group_responses': group_responses, 'group_judege': group_judege})
		else:
			result['all_wrong'].append({'query': query, 'label': label, 'image': image, 'group_responses': group_responses, 'group_judege': group_judege})
		cor += group_judege.count(True) 
	print('test filename: %s' % args.test_path)
	print('acc: %.3f' % (cor / len(all_querys) / args.n))
	if args.n > 1:
		print(len(result['all_correct']), len(result['any_correct']), len(result['all_wrong']))
	if args.result_path:
		print('saving to: %s' % args.result_path)
		to_json(result, args.result_path)
