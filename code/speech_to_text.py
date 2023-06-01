import pandas as pd
from tqdm.auto import tqdm
from huggingsound import SpeechRecognitionModel


def get_audio(df):
    paths = []
    for path in tqdm(df['path']):
        # librosa패키지를 사용하여 wav 파일 load
        path='/opt/ml/dacon/data'+path[1:]
        paths.append(path)
    return paths

train_df = pd.read_csv('/opt/ml/dacon/data/train.csv')
test_df = pd.read_csv('/opt/ml/dacon/data/test.csv')

train_paths = get_audio(train_df)
test_paths = get_audio(test_df)

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

train_texts = model.transcribe(train_paths)
test_texts = model.transcribe(test_paths)

train_text_df = pd.DataFrame()
test_text_df = pd.DataFrame()

for output, df in zip(train_texts, train_df.itertuples()):
    text = output['transcription']
    train_text_df = pd.concat([train_text_df, pd.DataFrame({'id':[df.id], 'text':[text], 'label':[df.label]})])

for output, df in zip(test_texts, test_df.itertuples()):
    text = output['transcription']
    test_text_df = pd.concat([test_text_df, pd.DataFrame({'id':[df.id], 'text':[text]})])

train_text_df.to_csv('/opt/ml/dacon/data/train_text.csv', index = False)
test_text_df.to_csv('/opt/ml/dacon/data/test_text.csv', index = False)