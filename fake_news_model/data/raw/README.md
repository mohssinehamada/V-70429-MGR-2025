# Dataset Downloads

Please download the following datasets and place them in the appropriate directories:

1. FEVER dataset: https://fever.ai/dataset/fever.html
   - Place in data/raw/fever/

2. LIAR dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
   - Extract and place in data/raw/liar/

3. PolitiFact dataset: You'll need to scrape or find a pre-processed version
   - Place in data/raw/politifact/

Once downloaded, run:
```
python -m utils.preprocessing
```
To prepare the data for training.
