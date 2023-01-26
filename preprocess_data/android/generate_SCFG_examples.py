import re

key_words = ['OVERVIEW_BUTTON', 'HOME', 'BACK', 'LONG_PRESS', 'PRESS', 'DOUBLE_PRESS', 'ENTER', 'SWIPE', 'OPEN']


def read_grammar(SCFG_path):
	SCFG_pt = open(file=SCFG_path, mode='r')
	grammar_dict = {}
	for line in SCFG_pt.readlines():
		if line.strip() and not line.strip().startswith('%'):
			#print(line.strip().split('\t'))
			print(line.strip().split('\t'))
			lhs, rhs_nl, rhs_lf = line.strip().split('\t')
			if lhs in grammar_dict:
				grammar_dict[lhs].append((rhs_nl, rhs_lf))
			else:
				grammar_dict[lhs] = []
				grammar_dict[lhs].append((rhs_nl, rhs_lf))
	return grammar_dict

def replace_one_token(current_str, lhs, sub_rhs_str):
	token_list = []
	is_replaced = False
	for token in current_str.split(' '):
		if token == lhs and not is_replaced:
			token_list.append(sub_rhs_str)
			is_replaced = True
		else:
			token_list.append(token)
	return " ".join(token_list)

def generate_examples(grammar, rhs_list, depth, max_depth):
	#print(rhs_list)
	all_rhs_list = []
	for rhs_nl, rhs_lf in rhs_list:
		lhs_in_nl = [token for token in rhs_nl.split(' ') if token.startswith("@")]
		#print(lhs_in_nl)
		if not lhs_in_nl:
			all_rhs_list.append((rhs_nl, rhs_lf))
		elif depth < max_depth:
			temp_rhs_list = [(rhs_nl, rhs_lf)]
			for lhs in lhs_in_nl:
				updated_temp_rhs_list = []
				#print(grammar[lhs])
				sub_rhs_list = generate_examples(grammar, rhs_list=grammar[lhs], depth=depth + 1, max_depth=max_depth)
				for sub_rhs_nl, sub_rhs_lf in sub_rhs_list:
					for current_rhs_nl, current_rhs_lf in temp_rhs_list:
						#updated_rhs_nl = current_rhs_nl.replace(lhs, sub_rhs_nl)
						updated_rhs_nl = replace_one_token(current_rhs_nl, lhs, sub_rhs_nl)
						#updated_rhs_lf = current_rhs_lf.replace(lhs, sub_rhs_lf)
						updated_rhs_lf = replace_one_token(current_rhs_lf, lhs, sub_rhs_lf)
						updated_temp_rhs_list.append((updated_rhs_nl, updated_rhs_lf))
				temp_rhs_list = updated_temp_rhs_list
			all_rhs_list.extend(temp_rhs_list)

	return all_rhs_list


def remove_empty_string(examples):
	pruned_examples = []
	for index, (nl, lf) in enumerate(examples):
		pruned_examples.append((" ".join([token for token in nl.split(' ') if not token == 'EmptyString']), lf))
	return pruned_examples


def prune_examples(examples):
	import ast
	component_path = "../../preprocess_data/android/user_supplementary_files/raw_component.txt"
	component_names = open(file=component_path, mode='r')

	app_dict = {}
	for line in component_names.readlines():
		# print(line.split(":")[1])
		app_name = "app:"+"_".join(line.split(":")[0].lower().split(" "))
		component_list = ast.literal_eval(line.split(":")[1].strip())
		app_dict[app_name] = set(["component:"+"_".join(com.lower().split(" ")) for com in component_list])

	pruned_examples = []
	for index, (nl, lf) in enumerate(examples):

		key_words_dict = {k: 0 for k in key_words}
		add_example = True
		for lf_token in lf.split(' '):
			if lf_token in key_words_dict and key_words_dict[lf_token] > 0:
				add_example = False
				break
			elif lf_token in key_words_dict:
				key_words_dict[lf_token] += 1

		is_app = False
		comp_list = None
		for lf_token in lf.split(' '):
			if lf_token in app_dict:
				is_app = True
				comp_list = app_dict[lf_token]
				break

		obey_constraint = True
		if is_app:
			for lf_token in lf.split(' '):
				if lf_token.startswith("component:") and not (lf_token in comp_list):
					obey_constraint = False


		if add_example and obey_constraint:
			pruned_examples.append((nl, lf))



	return pruned_examples


def add_ids(exp_str):
	entity_count = {}
	token_list = []
	for token in exp_str.split(' '):
		if token.startswith('_VV') or token.startswith('\'_VV'):
			if token.startswith('\'_VV'):
				entity_token = token[1:-1]
			else:
				entity_token = token
			if entity_token in entity_count:
				entity_count[entity_token] = entity_count[entity_token] + 1
			else:
				entity_count[entity_token] = 0
			if token.startswith('\'_VV'):
				token_list.append("\'" + entity_token + str(entity_count[entity_token]) + "\'")
			else:
				token_list.append(entity_token + str(entity_count[entity_token]))
		else:
			token_list.append(token)
	return " ".join(token_list)


def add_ids_to_examples(examples):
	edited_examples = []
	for index, (nl, lf) in enumerate(examples):
		edited_examples.append((add_ids(nl), add_ids(lf)))
	return edited_examples


def postedit_examples(examples):
	pruned_examples = remove_empty_string(examples)
	pruned_examples = prune_examples(pruned_examples)
	#pruned_examples = add_ids_to_examples(pruned_examples)
	return pruned_examples


def write_examples(examples, dump_path):
	dump_pt = open(file=dump_path, mode='w')
	for nl, lf in examples:
		dump_pt.write(nl + '\t' + lf + '\n')
	dump_pt.close()


if __name__ == '__main__':
	SCFG_path = "user_supplementary_files/SCFG.txt"
	example_path = "user_supplementary_files/andriod_examples.txt"
	grammar = read_grammar(SCFG_path)
	examples = generate_examples(grammar, grammar["@ROOT"], depth=0, max_depth=1)
	#print(examples)
	examples = postedit_examples(examples)
	write_examples(examples, example_path)
