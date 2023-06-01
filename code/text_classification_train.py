import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import sklearn
from sklearn.metrics import f1_score
import torch


class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.x.items()}
        item['labels'] = torch.tensor(self.y[idx])
        return item

    def __len__(self):
        return len(self.y)

def tokenized_dataset(data, tokenizer):
    tokenized = tokenizer(list(data['text']), return_tensors="pt", padding=True)
    return tokenized

def compute_loss(output):
    preds=output.predictions.argmax(-1)
    labels = train_df.label
    f1 = calc_micro_f1_score(labels, preds)
    accuracy = calc_accuracy(labels, preds)
    return {'micro f1 score': f1,
            'accuracy_score': accuracy}

def calc_micro_f1_score(labels, preds):
    return sklearn.metrics.f1_score(labels, preds, average="micro") * 100.0

def calc_accuracy(labels, preds):
    return sklearn.metrics.accuracy_score(labels, preds) * 100.0


train_df = pd.read_csv('/opt/ml/dacon/data/train_text.csv')
test_df = pd.read_csv('/opt/ml/dacon/data/test_text.csv')

MODEL_NAME='distilbert-base-uncased'

model_config = AutoConfig.from_pretrained(MODEL_NAME)
model_config.num_labels = 6
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = model_config)

tokenized_train = tokenized_dataset(train_df, tokenizer)
train_label= list(train_df['label'])

train_dataset = TextDataset(tokenized_train, train_label)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

classifier.parameters
classifier.to(device)

training_args = TrainingArguments(
    output_dir='/opt/ml/dacon/results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/opt/ml/dacon/logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True 
  )
trainer = Trainer(
  model=classifier,                         # the instantiated 🤗 Transformers model to be trained
  args=training_args,                  # training arguments, defined above
  train_dataset=train_dataset,         # training dataset
  eval_dataset=train_dataset,             # evaluation dataset
  compute_metrics=compute_loss         # define metrics function
)

# train model
trainer.train()
classifier.save_pretrained('/opt/ml/dacon/best_model')