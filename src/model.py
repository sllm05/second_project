import pytorch_lightning as pl
import torch
from utils import compute_metrics
from data import prepare_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
import os
import logging
from peft import LoraConfig, get_peft_model

# transformers 로깅 레벨 조정
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_tokenizer_and_model_for_train(args, silent=False):
    """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
    MODEL_NAME = args.model_name
    
    if not silent:
        print(f"\n모델 로딩 중: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    
    if not silent:
        print(f"✓ 모델 로드 완료")
    
    return tokenizer, model

# FacebookAI/xlm-roberta-large 모델 돌리려면 lora 필요
# def load_tokenizer_and_model_for_train(args, silent=False):
#     """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
#     MODEL_NAME = args.model_name
    
#     if not silent:
#         print(f"\n모델 로딩 중: {MODEL_NAME}")
    
#     # 토크나이저 로드 (모든 모델 동일하게 처리)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     # setting model hyperparameter
#     model_config = AutoConfig.from_pretrained(MODEL_NAME)
#     model_config.num_labels = 2

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME, config=model_config
#     )

#         # LoRA 설정
#     lora_config = LoraConfig(
#         r=8,  # rank
#         lora_alpha=16,
#         target_modules=["query", "value"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="SEQ_CLS"
#     )
    
#     model = get_peft_model(model, lora_config)
    
#     if not silent:
#         print(f"✓ 모델 로드 완료")
#         model.print_trainable_parameters()
    
#     return tokenizer, model

def load_model_for_inference(model_name, model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load"""
    # transformers 경고 숨기기
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # KoBERT 특수 처리
    if "skt/kobert" in model_name.lower():
        try:
            from kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained(model_name)
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "kobert-tokenizer"])
            from kobert_tokenizer import KoBERTTokenizer
            tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)

    return tokenizer, model


def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset, silent=False):
    """학습(train)을 위한 huggingface trainer 설정"""
    
    # WandB 사용 여부 결정
    use_wandb = not args.quiet if hasattr(args, 'quiet') else True
    
    training_args = TrainingArguments(
        output_dir=args.save_path + "/results",
        save_total_limit=args.save_limit,
        save_steps=args.save_step,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "logs",
        logging_steps=args.logging_step,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
        report_to="wandb" if use_wandb else "none",  # 조건부 WandB
        run_name=args.run_name,
        disable_tqdm=silent,
        logging_first_step=False,
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=20, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    
    if not silent:
        print("✓ Trainer 설정 완료")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=len(hate_train_dataset) * args.epochs,
            ),
        ),
    )

    return trainer


def train(args, silent=False):
    """모델을 학습(train)하고 best model을 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not silent:
        print(f"✓ 디바이스: {device}")

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args, silent=silent)
    model.to(device)

    # set data
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_dir, tokenizer, args.max_len, silent=silent)
    )

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset, silent=silent
    )

    # train model
    if not silent:
        print("\n" + "="*80)
        print("학습 시작")
        print("="*80)
    
    trainer.train()
    
    if not silent:
        print("\n" + "="*80)
        print("학습 완료")
        print("="*80)
    
    safe_model_name = args.model_name.replace("/", "_")
    model_save_path = os.path.join(args.model_dir, safe_model_name)
    model.save_pretrained(model_save_path)
    
    if not silent:
        print(f"✓ 모델 저장: {model_save_path}")