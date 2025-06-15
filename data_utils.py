import json
import sys
import re

def to_jsonl(data, fout):
	with open(fout, 'w', encoding='utf8') as f:
		for row in data:
			f.write(json.dumps(row, ensure_ascii=False) + '\n')

def read_jsonl(filename):
	data = []
	with open(filename, 'r', encoding='utf8') as f:
		for line in f.readlines():
			data.append(json.loads(line))
	return data

def read_json(filename):
	with open(filename, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def to_json(data, fout):
	with open(fout, 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)	

# def latex_norm(answer):
# 	answer = answer.replace('$', '').strip().replace(' ', '')
# 	answer = answer.replace('^{\\circ}', '°').replace('^\\circ', '°')
# 	return answer

def latex_norm(answer):
	if answer.endswith(';') or answer.endswith('.'):
		answer = answer[:-1]
	answer = answer.replace('\\left', '').replace('\\right', '')
	answer = answer.replace('－', '-').replace('∶', ':').replace('，', ',')
	answer = answer.replace('$', '').strip().replace(' ', '')
	# number{x} ==> x
	number = re.findall('number{[0-9\.]+}', answer)
	if number:
		if '\\' + number[0] in answer:
			answer = answer.replace('\\' + number[0], number[0][number[0].index('{')+1:-1])
		else:
			answer = answer.replace(number[0], number[0][number[0].index('{')+1:-1])
	# text{a} ==> a
	text = re.findall('text{[0-9\.]+}', answer)
	if text:
		if '\\' + text[0] in answer:
			answer = answer.replace('\\' + text[0], text[0][text[0].index('{')+1:-1])
		else:
			answer = answer.replace(text[0], text[0][text[0].index('{')+1:-1])
	# ^circ ==> °
	answer = answer.replace('^{\\circ}', '°').replace('{}^\\circ', '°').replace('^\\circ', '°').replace('degrees', '°').replace('^{°}', '°')
	degree = re.findall('{{[0-9\.]+}°}', answer)
	if degree:
		answer = answer.replace(degree[0], degree[0][2:-3] + '°')
	# meters ==> m
	answer = answer.replace('meters', 'm')
	# pi ==> π
	answer = answer.replace('\\pi', 'π')
	# a:b ==> a/b
	answer = answer.replace(':', '/')
	# pm ==> ±
	answer = answer.replace('\\pm', '±')
	# le ==> ≤
	answer = answer.replace('\\leqslant', '≤').replace('\\le', '≤')
	# ge ==> ≥
	answer = answer.replace('\\ge', '≥')
	# %
	answer = answer.replace('\\%', '%')
	# angel ==> ∠
	answer = answer.replace('\\angle', '∠')
	# parallel ==> ||
	answer = answer.replace('\\parallel', '||')
	# perp ==> ⊥
	answer = answer.replace('\\perp', '⊥')
	# sqrt{x} ==> √x
	sqrt = re.findall('sqrt{[0-9π\.]+}', answer)
	if sqrt:
		if '\\' + sqrt[0] in answer:
			answer = answer.replace('\\' + sqrt[0], '√' + sqrt[0][5:-1])
		else:
			answer = answer.replace(sqrt[0], '√' + sqrt[0][5:-1])
	# dfrac/tfrac ==> frac
	answer = answer.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
	# frac{a}{b} ==> a/b
	frac = re.findall('frac{[0-9π√]+}{[0-9π√]+}', answer)
	if frac:
		if '\\' + frac[0] in answer:
			answer = answer.replace('\\' + frac[0], frac[0][frac[0].index('{')+1: frac[0].index('}')] + '/' + frac[0][frac[0].index('}')+2:-1])
		else:
			answer = answer.replace(frac[0], frac[0][frac[0].index('{')+1: frac[0].index('}')] + '/' + frac[0][frac[0].index('}')+2:-1])
	# unit{cm} ==> cm
	unit = re.findall('unit{[a-z°]+}', answer)
	if unit:
		if '\\' + unit[0] in answer:
			answer = answer.replace('\\' + unit[0], unit[0][unit[0].index('{')+1:-1])
		else:
			answer = answer.replace(unit[0], unit[0][unit[0].index('{')+1:-1])
	answer = answer.replace('\\text{cm}^2', 'cm²')
	answer = answer.replace('{{m}^{2}}', 'm²')
	# TODO: expression: -2a+b-ab
	return answer