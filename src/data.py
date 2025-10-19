import os
import pandas as pd
import torch
import ast
from sklearn.model_selection import StratifiedKFold


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


def load_data(dataset_dir, silent=False):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    if not silent:
        print(f"✓ 데이터 로드: {len(dataset)}개 샘플")
    return dataset


def construct_tokenized_dataset(dataset, tokenizer, max_length, silent=False):
    """입력값(input)에 대하여 토크나이징"""
    tokenized_senetences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    if not silent:
        print(f"✓ 토크나이징 완료")
    return tokenized_senetences


def prepare_dataset(dataset_dir, tokenizer, max_len, silent=False):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    if not silent:
        print("\n데이터셋 준비 중...")
    
    # load_data
    train_dataset = load_data(os.path.join(dataset_dir, "train.csv"), silent=silent)
    valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"), silent=silent)
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"), silent=silent)
    
    # train_dataset - 컬럼명이 다를 수 있으므로 확인 후 변경
    if 'sentence_form' in train_dataset.columns:
        train_dataset = train_dataset.rename(columns={'sentence_form': 'input'})
    if 'hate_speech_idx' in train_dataset.columns:
        train_dataset = train_dataset.rename(columns={'hate_speech_idx': 'output'})
    
    # 컬럼명 표준화 확인
    if 'input' not in train_dataset.columns:
        print(f"Error: 'input' column not found in train.csv. Available columns: {train_dataset.columns.tolist()}")
    if 'output' not in train_dataset.columns:
        print(f"Error: 'output' column not found in train.csv. Available columns: {train_dataset.columns.tolist()}")
    
    train_dataset['output'] = train_dataset['output'].astype(int)

    # valid_dataset - 컬럼명이 다를 수 있으므로 확인 후 변경
    if 'sentence_form' in valid_dataset.columns:
        valid_dataset = valid_dataset.rename(columns={'sentence_form': 'input'})
    if 'hate_speech_idx' in valid_dataset.columns:
        valid_dataset = valid_dataset.rename(columns={'hate_speech_idx': 'output'})
    
    valid_dataset['output'] = valid_dataset['output'].astype(int)
    
    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    test_label = [0] * len(test_dataset)  # 더미 레이블

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len, silent=silent)
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len, silent=silent)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len, silent=silent)

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    
    if not silent:
        print(f"✓ Train: {len(hate_train_dataset)}, Valid: {len(hate_valid_dataset)}, Test: {len(hate_test_dataset)}")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset

def prepare_kfold_dataset(dataset_dir, fold_idx, k_folds, tokenizer, max_len):
    """K-Fold 교차 검증을 위한 데이터셋 준비"""
    # train.csv와 dev.csv 통합
    train_df = load_data(os.path.join(dataset_dir, "train.csv"), silent=True)
    dev_df = load_data(os.path.join(dataset_dir, "dev.csv"), silent=True)
    
    # 컬럼명 통일
    if 'sentence_form' in train_df.columns:
        train_df = train_df.rename(columns={'sentence_form': 'input'})
    if 'hate_speech_idx' in train_df.columns:
        train_df = train_df.rename(columns={'hate_speech_idx': 'output'})
        
    if 'sentence_form' in dev_df.columns:
        dev_df = dev_df.rename(columns={'sentence_form': 'input'})
    if 'hate_speech_idx' in dev_df.columns:
        dev_df = dev_df.rename(columns={'hate_speech_idx': 'output'})
    
    combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    # StratifiedKFold 설정
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # fold_idx에 해당하는 데이터 분할
    for i, (train_index, valid_index) in enumerate(skf.split(combined_df["input"], combined_df["output"])):
        if i == fold_idx:
            train_fold_df = combined_df.iloc[train_index]
            valid_fold_df = combined_df.iloc[valid_index]
            break
    
    # 토크나이징
    tokenized_train = construct_tokenized_dataset(train_fold_df, tokenizer, max_len, silent=True)
    tokenized_valid = construct_tokenized_dataset(valid_fold_df, tokenizer, max_len, silent=True)
    
    # PyTorch Dataset 생성
    hate_train_dataset = hate_dataset(tokenized_train, train_fold_df["output"].values)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_fold_df["output"].values)
    
    return hate_train_dataset, hate_valid_dataset

def prepare_test_dataset(dataset_dir, tokenizer, max_len):
    """테스트 데이터셋 준비"""
    test_df = load_data(os.path.join(dataset_dir, "test.csv"), silent=True)
    
    tokenized_test = construct_tokenized_dataset(test_df, tokenizer, max_len, silent=True)
    
    # test.csv에 label이 없으므로 더미 label 사용
    dummy_labels = [0] * len(test_df)
    hate_test_dataset = hate_dataset(tokenized_test, dummy_labels)
    
    return hate_test_dataset, test_df