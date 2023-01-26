import nlpaug.augmenter.word as naw
exp_schema = "ui_parser/datasets/android_user/upper_train.txt"

exp_train = "ui_parser/datasets/android_user/train.txt"
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-cased', action="insert", aug_max=1)
#augmented_text = aug.augment(text)

exp_train_f = open(file=exp_train, mode='w')

exp_schema_f = open(file=exp_schema, mode='r')
previous_utter = set()
for line in exp_schema_f.readlines():
	line_split = line.split('\t')
	utter = line_split[0].strip()
	lf = line_split[1].strip()
	schema = line_split[2].strip()
	if not utter in previous_utter:
		exp_train_f.write(utter + '\t' + lf  +  '\t' + schema + '\n')
	previous_utter.add(utter)

	if not (utter.startswith('type') or utter.startswith('write') or utter.startswith('enter') or utter.startswith('input')):
		aug_utter = aug.augment(utter,n=1)[0]
		#print(aug_utter)
		if not aug_utter in previous_utter:
			exp_train_f.write(aug_utter + '\t' + lf  +  '\t' + schema  + '\n')
		previous_utter.add(aug_utter)


		if not utter.lower() in previous_utter:
			exp_train_f.write(utter.lower() + '\t' + lf  +  '\t' + schema  + '\n')
		previous_utter.add(utter.lower())



exp_train_f.close()
exp_schema_f.close()