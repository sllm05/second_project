import os
import pandas as pd
import torch
import ast


class hate_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""

    def __init__(self, hate_dataset, labels):
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(dataset_dir):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    print("dataframe 의 형태")
    print("-" * 100)
    print(dataset.head())
    return dataset


def construct_tokenized_dataset(dataset, tokenizer, max_length):
    """입력값(input)에 대하여 토크나이징"""
    print("tokenizer 에 들어가는 데이터 형태")
    print(dataset["input"][:5])

    tokenized_senetences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        # return_token_type_ids=False,  # BERT 이후 모델(RoBERTa 등) 사용할때 False
    )
    print("tokenizing 된 데이터 형태")
    print("-" * 100)
    print(tokenized_senetences[:5])
    return tokenized_senetences


def prepare_dataset(dataset_dir, tokenizer, max_len):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    train_dataset = load_data(os.path.join(dataset_dir, "augmented_train_llm.csv")) 
    valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    print("--- data loading Done ---")
    
# train_dataset
    train_dataset = train_dataset.rename(columns={'sentence_form': 'input', 'hate_speech_idx': 'output'})
    train_dataset['output'] = train_dataset['output'].astype(int)

    # valid_dataset
    valid_dataset = valid_dataset.rename(columns={'sentence_form': 'input', 'hate_speech_idx': 'output'})
    valid_dataset['output'] = valid_dataset['output'].astype(int)

    # test_dataset
    # test.csv에는 레이블(output)이 없으므로, 컬럼 이름만 변경하고 레이블은 따로 설정하지 않습니다.
    test_dataset = test_dataset.rename(columns={'id': 'id', 'input': 'input'}) 
    
    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    # test_dataset에는 'output' 컬럼이 없으므로, 레이블을 None으로 설정하거나 빈 리스트로 둡니다.
    test_label = None 

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len)
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len)
    # test_dataset은 레이블을 포함하지 않으므로, 토크나이징 시 레이블 정보를 추가할 필요가 없습니다.
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    # test 데이터셋은 레이블이 없으므로, 모델 예측을 위한 데이터셋으로만 사용합니다.
    # hate_dataset 클래스는 레이블을 필수로 받기 때문에, 빈 리스트를 전달합니다.
    hate_test_dataset = hate_dataset(tokenized_test, [0] * len(test_dataset)) 
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset