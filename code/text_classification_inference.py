import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.x.items()}
        return item

    def __len__(self):
        return len(self.x['input_ids'])

def tokenized_dataset(data, tokenizer):
    tokenized = tokenizer(list(data['text']), return_tensors="pt", padding=True)
    return tokenized

test_df = pd.read_csv('/opt/ml/dacon/data/test_text.csv')

MODEL_NAME='distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

classifier = AutoModelForSequenceClassification.from_pretrained('/opt/ml/dacon/best_model')

tokenized_test = tokenized_dataset(test_df, tokenizer)
test_dataset = TextDataset(tokenized_test)

test_args = TrainingArguments(
    output_dir='../text_classification',
    do_predict = True,
    per_device_eval_batch_size = 32,   
    dataloader_drop_last = False 
)

trainer = Trainer(
              model = classifier, 
              args = test_args
              )

test_results = trainer.predict(test_dataset)
probs = test_results.predictions
preds = test_results.predictions.argmax(-1)
submission = pd.read_csv('/opt/ml/dacon/data/sample_submission.csv')
submission['label'] = preds
submission['probs'] = probs
submission.to_csv('../text_classification.csv', index=False)