import pandas as pd
from collections import Counter
import os
from sklearn.metrics import accuracy_score, f1_score
import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import load_model_for_inference
from torch.utils.data import TensorDataset, DataLoader
import json
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def inference(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    all_logits = []

    # tqdm 진행바 숨기기
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0)

def infer_and_eval(model_name, model_dir, record_time, dataset_dir, max_len, eval_file="dev.csv", silent=True):
    """학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 모델 로딩 - 경고 숨기기
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    tokenizer, model = load_model_for_inference(model_name, model_dir)
    model.to(device)

    # 평가용 데이터 불러오기
    eval_path = os.path.join(dataset_dir, eval_file)
    if not os.path.exists(eval_path):
        if not silent:
            print(f"Warning: {eval_file} not found, using train.csv for evaluation")
        eval_path = os.path.join(dataset_dir, "train.csv")
    
    test_dataset = pd.read_csv(eval_path)
    
    # output 컬럼이 있는지 확인 (평가용)
    has_labels = "output" in test_dataset.columns
    if has_labels:
        test_label = test_dataset["output"].values
    else:
        test_label = None
    
    # input 컬럼 확인
    if "input" not in test_dataset.columns:
        if not silent:
            print(f"Error: 'input' column not found in {eval_path}")
        return 0.0, 0.0
    
    tokenized_test_data = tokenizer(
        test_dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    dataset = TensorDataset(
        tokenized_test_data['input_ids'],
        tokenized_test_data['attention_mask']
    )

    logits = inference(model, dataset, device)
    pred = np.argmax(logits, axis=1)
    
    # 평가 (라벨이 있는 경우만)
    if has_labels:
        acc = accuracy_score(test_label, pred)
        f1 = f1_score(test_label, pred, average='macro')
    else:
        acc, f1 = 0.0, 0.0

    # 예측 결과를 CSV 파일로 저장
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    safe_model_name = model_name.replace("/", "_")

    # id 컬럼이 있는지 확인
    if "id" in test_dataset.columns:
        id_column = test_dataset["id"]
    else:
        id_column = range(len(test_dataset))

    output = pd.DataFrame(
        {
            "id": id_column,
            "input": test_dataset["input"],
            "output": pred,
            "logits_class_0": logits[:, 0],
            "logits_class_1": logits[:, 1]
        }
    )
    output.to_csv(os.path.join(result_path, f"{safe_model_name}_{record_time}.csv"), index=False)
    
    return acc, f1

def vote(*labels):
    """입력된 label들 중 가장 많이 있는 label을 선정"""
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

def save_results_as_jsonl(result_data, filename, result_path="./prediction/"):
    """결과를 JSONL 형식으로 저장"""
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    filepath = os.path.join(result_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in result_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    return filepath

def evaluate_hard_voting(model_list, test_data, time, save_jsonl=True, silent=True):
    """각 모델의 결과를 입력받아 hard voting 후 평가"""
    
    result_labels = []

    for i in range(len(test_data)):
        labels = [model['output'].iloc[i] for model in model_list]
        voted_label = vote(*labels)
        result_labels.append(voted_label)
    
    # 평가 (라벨이 있는 경우만)
    has_labels = 'output' in test_data.columns and test_data['output'].notna().any()
    if has_labels:
        labels = test_data['output'].values
        acc = accuracy_score(labels, result_labels)
        f1 = f1_score(labels, result_labels, average='macro')
    else:
        acc, f1 = 0.0, 0.0

    # 기존 CSV 저장 방식 유지
    result_df = pd.DataFrame({'voted_label': result_labels})
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_df.to_csv(
        os.path.join(result_path, f"ensemble_{len(model_list)}_{time}.csv"), index=False
    )
    
    # JSONL 형식으로도 저장
    if save_jsonl:
        if 'id' in test_data.columns:
            ids = test_data['id'].tolist()
        else:
            ids = [f"item_{i:06d}" for i in range(len(result_labels))]
        
        jsonl_data = []
        for i, (id_val, pred_label) in enumerate(zip(ids, result_labels)):
            jsonl_data.append({
                "id": id_val,
                "output": int(pred_label)
            })
        
        jsonl_filename = f"hard_voting_ensemble_{len(model_list)}_{time}.jsonl"
        save_results_as_jsonl(jsonl_data, jsonl_filename)
    
    return acc, f1

def evaluate_soft_voting(model_list, test_data, time, save_jsonl=True, silent=True):
    """각 모델의 확률값을 입력받아 soft voting 후 평가"""
    final_logits = np.zeros((len(test_data), 2))

    for model in model_list:
        final_logits[:, 0] += model['logits_class_0'].values
        final_logits[:, 1] += model['logits_class_1'].values

    final_logits /= len(model_list)
    pred = np.argmax(final_logits, axis=1)

    # 평가 (라벨이 있는 경우만)
    has_labels = 'output' in test_data.columns and test_data['output'].notna().any()
    if has_labels:
        labels = test_data['output'].values
        acc = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred, average='macro')
    else:
        acc, f1 = 0.0, 0.0

    # 기존 CSV 저장 방식 유지
    result_df = pd.DataFrame({'voted_label': pred})
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_df.to_csv(
        os.path.join(result_path, f"soft_ensemble_{len(model_list)}_{time}.csv"), index=False
    )
    
    # JSONL 형식으로도 저장
    if save_jsonl:
        if 'id' in test_data.columns:
            ids = test_data['id'].tolist()
        else:
            ids = [f"item_{i:06d}" for i in range(len(pred))]
        
        jsonl_data = []
        for i, (id_val, pred_label) in enumerate(zip(ids, pred)):
            jsonl_data.append({
                "id": id_val,
                "output": int(pred_label)
            })
        
        jsonl_filename = f"soft_voting_ensemble_{len(model_list)}_{time}.jsonl"
        save_results_as_jsonl(jsonl_data, jsonl_filename)
    
    return acc, f1

def single_model_inference(args, model_name, eval_file="dev.csv", save_jsonl=True):
    """단일 모델 추론 및 성능 반환"""
    record_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    
    local_model_path = os.path.join(args.model_dir, model_name.replace("/", "_"))
    if os.path.exists(local_model_path):
        model_dir = local_model_path
    else:
        model_dir = model_name

    acc, f1 = infer_and_eval(model_name, model_dir, record_time, args.dataset_dir, args.max_len, eval_file, silent=True)
    
    # JSONL 형식으로도 저장 (test.csv 파일에 대해서만)
    if save_jsonl and eval_file == "test.csv":
        safe_model_name = model_name.replace("/", "_")
        csv_path = f"./prediction/{safe_model_name}_{record_time}.csv"
        
        try:
            result_df = pd.read_csv(csv_path)
            jsonl_data = []
            for _, row in result_df.iterrows():
                jsonl_data.append({
                    "id": row["id"],
                    "output": int(row["output"])
                })
            
            jsonl_filename = f"{safe_model_name}_{record_time}.jsonl"
            save_results_as_jsonl(jsonl_data, jsonl_filename)
            
        except FileNotFoundError:
            pass
    
    return acc, f1, record_time

def ensemble_inference(args, model_names_subset, record_time=None, eval_file="dev.csv", save_jsonl=True):
    """앙상블 추론 및 평가"""
    if record_time is None:
        record_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    
    # 각 모델에 대해 추론을 수행하고 결과를 CSV로 저장
    for model_name in model_names_subset:
        local_model_path = os.path.join(args.model_dir, model_name.replace("/", "_"))
        if os.path.exists(local_model_path):
            model_dir = local_model_path
        else:
            model_dir = model_name

        infer_and_eval(model_name, model_dir, record_time, args.dataset_dir, args.max_len, eval_file, silent=True)
    
    # 저장된 CSV 파일들을 불러와서 앙상블 수행
    ans_list = []
    for model_name in model_names_subset:
        safe_model_name = model_name.replace("/", "_")
        csv_path = f"./prediction/{safe_model_name}_{record_time}.csv"
        
        try:
            ans_list.append(pd.read_csv(csv_path))
        except FileNotFoundError:
            pass

    if not ans_list:
        return None, None
    
    # 첫 번째 모델의 데이터를 기준으로 사용
    eval_path = os.path.join(args.dataset_dir, eval_file)
    original_data = pd.read_csv(eval_path)
    
    # Hard Voting 실행 및 평가
    hard_acc, hard_f1 = evaluate_hard_voting(ans_list, test_data=original_data, time=record_time, save_jsonl=save_jsonl and eval_file == "test.csv", silent=True)
    
    # Soft Voting 실행 및 평가
    soft_acc, soft_f1 = evaluate_soft_voting(ans_list, test_data=original_data, time=record_time, save_jsonl=save_jsonl and eval_file == "test.csv", silent=True)
    
    return (hard_acc, hard_f1), (soft_acc, soft_f1)