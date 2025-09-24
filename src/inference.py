from torch.utils.data import DataLoader
import pandas as pd
import torch
import os

import numpy as np
from tqdm import tqdm
from .model import load_model_for_inference
from .data import prepare_dataset

def inference(model, tokenized_sent, device):
    """학습된(trained) 모델을 통해 결과를 추론하는 function"""
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
    model.eval()
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
    return (np.concatenate(output_pred).tolist(),)

def infer_and_eval(model_name,model_dir):
    """학습된 모델로 추론(infer)한 후에 예측한 결과(pred)를 평가(eval)"""
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model & tokenizer
    tokenizer, model = load_model_for_inference(model_name,model_dir)
    model.to(device)

    # set data
    _,_, hate_test_dataset, test_dataset = prepare_dataset("./NIKL_AU_2023_COMPETITION_v1.0",tokenizer,256)

    # predict answer
    pred_answer = inference(model, hate_test_dataset, device)  # model에서 class 추론
    pred = pred_answer[0]
    print("--- Prediction done ---")

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            "id": test_dataset["id"],
            "input": test_dataset["input"],
            "output": pred,
        }
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    result_path = "./prediction/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # output.to_csv(
    #     os.path.join(result_path,"result.csv"), index=False
    # )
    # JSONL 파일로 변경
    output_path = os.path.join(result_path, "result_llm_arg.jsonl")
    output.to_json(
        output_path, 
        orient="records", 
        lines=True, 
        force_ascii=False
    )

    print("--- Save result ---")
    return output

if __name__ == "__main__":
    model_name = "beomi/KcELECTRA-base-v2022"
    model_dir = "./best_model"

    infer_and_eval(model_name,model_dir)
    