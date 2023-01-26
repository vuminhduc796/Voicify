# ui_parser

conda create --name parser python=3.8

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

conda install tqdm

pip install sacrebleu

conda install -c huggingface transformers

conda install scikit-learn

pip install nltk

conda install flask

# input format

utterance ||| button name list

example wtihout button list:

show me Reel on firefox |||

example with button list:

click community ||| today, account icon, discover, community, premium, covid-19


