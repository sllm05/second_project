import os
import argparse
import wandb
from model import train
from inference_ens import single_model_inference, ensemble_inference
import itertools
import pandas as pd
from datetime import datetime
import sys

def parse_args():
    """학습(train)과 추론(infer)에 사용되는 arguments를 관리하는 함수"""
    parser = argparse.ArgumentParser(description="Training and Inference arguments")

    parser.add_argument("--dataset_dir", type=str, default="./NIKL_AU_2023_COMPETITION_v1.0")
    parser.add_argument("--model_name", type=str, default="beomi/KcELECTRA-base-v2022")
    parser.add_argument("--save_path", type=str, default="./model")
    parser.add_argument("--save_step", type=int, default=200)
    parser.add_argument("--logging_step", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--model_dir", type=str, default="./best_model")
    parser.add_argument("--run_name", type=str, default="bert-test")
    parser.add_argument("--eval_file", type=str, default="dev.csv", choices=['train.csv', 'dev.csv', 'test.csv'])
    parser.add_argument("--mode", type=str, default="advanced_ensemble", 
                       choices=['train', 'single_inference', 'advanced_ensemble', 'test_submission'])
    parser.add_argument("--save_jsonl", action='store_true', default=True)

    return parser.parse_args()


def print_header(title, width=80):
    """깔끔한 헤더 출력"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def evaluate_single_models(args, all_models):
    """모든 단일 모델을 평가하고 성능을 기록"""
    print_header("단일 모델 성능 평가")
    
    model_performances = {}
    
    for i, model_name in enumerate(all_models, 1):
        model_short = model_name.split('/')[-1]
        print(f"\n[{i}/{len(all_models)}] {model_short}...", end=' ', flush=True)
        
        try:
            acc, f1, record_time = single_model_inference(args, model_name, args.eval_file, save_jsonl=args.save_jsonl)
            model_performances[i] = {
                'model_name': model_name,
                'accuracy': acc,
                'f1_score': f1,
                'record_time': record_time
            }
            
            print(f"✓ Acc: {acc*100:5.1f}% | F1: {f1*100:5.1f}%")
            
        except Exception as e:
            print(f"✗ 실패")
            model_performances[i] = {
                'model_name': model_name,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'record_time': None
            }
    
    # 성능 결과 저장 및 요약 출력
    results_df = pd.DataFrame.from_dict(model_performances, orient='index')
    timestamp = datetime.now().strftime('%m_%d_%H_%M')
    results_file = f"./single_model_performances_{timestamp}.csv"
    results_df.to_csv(results_file, index=True)
    
    print_header("단일 모델 성능 요약")
    print(f"{'순위':<6} {'모델명':<40} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 75)
    
    sorted_performances = sorted(model_performances.items(), 
                               key=lambda x: x[1]['f1_score'], reverse=True)
    
    for rank, (i, perf) in enumerate(sorted_performances, 1):
        model_short = perf['model_name'].split('/')[-1]
        if len(model_short) > 35:
            model_short = model_short[:32] + "..."
        print(f"{rank:<6} {model_short:<40} {perf['accuracy']*100:5.1f}%        {perf['f1_score']*100:5.1f}%")
    
    print(f"\n✓ 결과 저장: {results_file}")
    return model_performances


def find_best_model(model_performances, metric='f1_score'):
    """가장 성능이 좋은 모델을 찾는 함수"""
    best_model_id = max(model_performances.keys(), 
                       key=lambda x: model_performances[x][metric])
    best_model_info = model_performances[best_model_id]
    
    print(f"\n✨ 최고 성능 모델: #{best_model_id} {best_model_info['model_name']}")
    print(f"   {metric.upper()}: {best_model_info[metric]:.4f} ({best_model_info[metric]*100:.1f}%)")
    
    return best_model_id, best_model_info


def generate_ensemble_combinations(best_model_id, all_models):
    """최고 성능 모델을 기준으로 모든 앙상블 조합 생성"""
    other_model_indices = [i for i in range(1, len(all_models) + 1) if i != best_model_id]
    combinations = []
    
    # 2개 조합 (best + 1개)
    for other_id in other_model_indices:
        combinations.append([best_model_id, other_id])
    
    # 3개 조합 (best + 2개)
    for combo in itertools.combinations(other_model_indices, 2):
        combinations.append([best_model_id] + list(combo))
    
    # 4개 조합 (best + 3개)
    for combo in itertools.combinations(other_model_indices, 3):
        combinations.append([best_model_id] + list(combo))
    
    # 5개 조합 (best + 4개 = 전체)
    combinations.append([best_model_id] + other_model_indices)
    
    return combinations


def find_best_ensemble_interactive(ensemble_results, all_models):
    """최고 성능 앙상블 조합 찾기 (동점 시 사용자 선택)"""
    if not ensemble_results:
        return None, None
    
    # ✨ 수정 1: 전체에서 최고 F1 찾기 (Hard/Soft 통합)
    max_f1 = 0
    best_voting_type = None
    
    for res in ensemble_results.values():
        if res['hard_voting_f1'] > max_f1:
            max_f1 = res['hard_voting_f1']
            best_voting_type = 'hard'
        if res['soft_voting_f1'] > max_f1:
            max_f1 = res['soft_voting_f1']
            best_voting_type = 'soft'
    
    print(f"\n✨ 전체 최고 F1 Score: {max_f1*100:.2f}% ({best_voting_type.upper()} Voting)")
    
    # ✨ 수정 2: 최고 F1을 가진 모든 조합 찾기 (Hard/Soft 구분 없이)
    candidates = []
    
    for name, res in ensemble_results.items():
        # Hard Voting이 최고 F1과 같은 경우
        if abs(res['hard_voting_f1'] - max_f1) < 0.0001:
            candidates.append((name, res, 'Hard Voting', res['hard_voting_f1'], res['hard_voting_acc']))
        
        # Soft Voting이 최고 F1과 같은 경우
        if abs(res['soft_voting_f1'] - max_f1) < 0.0001:
            candidates.append((name, res, 'Soft Voting', res['soft_voting_f1'], res['soft_voting_acc']))
    
    # ✨ 수정 3: 동점 처리 단순화
    if len(candidates) == 1:
        # 동점 없음 - 바로 선택
        name, res, voting_type, f1, acc = candidates[0]
        print(f"→ 유일한 최고 조합 자동 선택")
        return (name, res), voting_type
    
    else:
        # 동점 있음 - 사용자 선택
        print("\n" + "="*80)
        print(f"⚠️  최고 F1 점수({max_f1*100:.2f}%)를 가진 조합이 {len(candidates)}개 있습니다.")
        print("="*80)
        
        # Accuracy 기준으로 정렬
        candidates.sort(key=lambda x: x[4], reverse=True)
        
        print("\n동점 조합 목록 (Accuracy 순):")
        for idx, (name, res, voting_type, f1, acc) in enumerate(candidates, 1):
            model_names = [all_models[i-1].split('/')[-1] for i in res['combination']]
            print(f"  [{idx}] 조합 {res['combination']} - {voting_type}")
            print(f"      F1: {f1*100:.2f}% | Acc: {acc*100:.2f}%")
            print(f"      모델: {', '.join(model_names)}")
        
        while True:
            try:
                choice = input(f"\n선택하세요 (1-{len(candidates)}) [Enter = Accuracy 최고]: ").strip()
                if choice == "":
                    selected_idx = 0
                    print(f"→ Accuracy가 가장 높은 조합 선택 ({candidates[0][4]*100:.2f}%)")
                    break
                else:
                    selected_idx = int(choice) - 1
                    if 0 <= selected_idx < len(candidates):
                        print(f"→ [{choice}]번 조합 선택")
                        break
                    else:
                        print(f"❌ 1-{len(candidates)} 사이의 숫자를 입력하세요.")
            except ValueError:
                print("❌ 올바른 숫자를 입력하세요.")
        
        name, res, voting_type, f1, acc = candidates[selected_idx]
        return (name, res), voting_type

def find_best_ensemble_comprehensive(ensemble_results, all_models):
    """Hard 최고, Soft 최고, 전체 최고 모두 찾기"""
    if not ensemble_results:
        return None, None, None
    
    # 1. Hard Voting 최고 찾기
    max_hard_f1 = max(res['hard_voting_f1'] for res in ensemble_results.values())
    hard_candidates = [
        (name, res) for name, res in ensemble_results.items()
        if abs(res['hard_voting_f1'] - max_hard_f1) < 0.0001
    ]
    hard_candidates.sort(key=lambda x: x[1]['hard_voting_acc'], reverse=True)
    best_hard = hard_candidates[0] if hard_candidates else None
    
    # 2. Soft Voting 최고 찾기
    max_soft_f1 = max(res['soft_voting_f1'] for res in ensemble_results.values())
    soft_candidates = [
        (name, res) for name, res in ensemble_results.items()
        if abs(res['soft_voting_f1'] - max_soft_f1) < 0.0001
    ]
    soft_candidates.sort(key=lambda x: x[1]['soft_voting_acc'], reverse=True)
    best_soft = soft_candidates[0] if soft_candidates else None
    
    # 3. 전체 최고 (정보용)
    max_f1 = max(max_hard_f1, max_soft_f1)
    best_overall_type = 'Hard' if max_hard_f1 >= max_soft_f1 else 'Soft'
    
    return best_hard, best_soft, (max_f1, best_overall_type)

def run_advanced_ensemble(args, all_models):
    """고급 앙상블 전략 실행"""
    
    record_time = datetime.now().strftime("%m_%d_%H_%M")
    
    # 1-3단계: 동일
    model_performances = evaluate_single_models(args, all_models)
    best_model_id, best_model_info = find_best_model(model_performances, 'f1_score')
    combinations = generate_ensemble_combinations(best_model_id, all_models)
    
    print_header(f"앙상블 실험 ({len(combinations)}가지 조합)")
    ensemble_results = {}
    
    # 앙상블 실험 (dev.csv로 평가만)
    for i, combo in enumerate(combinations, 1):
        combo_models = [all_models[idx-1] for idx in combo]
        combo_name = f"best({best_model_id})+{'+'.join(map(str, [x for x in combo if x != best_model_id]))}"
        print(f"[{i:2d}/{len(combinations)}] {len(combo)}개 모델...", end=' ', flush=True)
        
        try:
            hard_results, soft_results = ensemble_inference(
                args, combo_models, record_time, args.eval_file, save_jsonl=False
            )
            
            if hard_results and soft_results:
                hard_acc, hard_f1 = hard_results
                soft_acc, soft_f1 = soft_results
                
                ensemble_results[combo_name] = {
                    'combination': combo,
                    'models': combo_models,
                    'hard_voting_acc': hard_acc,
                    'hard_voting_f1': hard_f1,
                    'soft_voting_acc': soft_acc,
                    'soft_voting_f1': soft_f1,
                    'num_models': len(combo)
                }
                
                print(f"✓ Hard F1: {hard_f1*100:5.1f}% | Soft F1: {soft_f1*100:5.1f}%")
            else:
                print("✗ 실패")
        except Exception as e:
            print(f"✗ 오류")
    
    # 결과 저장
    if ensemble_results:
        results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
        results_file = f"./ensemble_results_{record_time}.csv"
        results_df.to_csv(results_file, index=True)
        
        print_header("최종 앙상블 결과")
        
        # Hard/Soft 최고 찾기
        best_hard, best_soft, _ = find_best_ensemble_comprehensive(ensemble_results, all_models)
        
        # Hard Voting 최고
        if best_hard:
            hard_name, hard_res = best_hard
            print("\n🥇 Hard Voting 최고 조합:")
            print(f"   조합: {hard_res['combination']}")
            print(f"   F1: {hard_res['hard_voting_f1']*100:.2f}% | Acc: {hard_res['hard_voting_acc']*100:.2f}%")
            hard_model_names = [all_models[i-1].split('/')[-1] for i in hard_res['combination']]
            print(f"   모델: {', '.join(hard_model_names)}")
        
        # Soft Voting 최고
        if best_soft:
            soft_name, soft_res = best_soft
            print("\n🥈 Soft Voting 최고 조합:")
            print(f"   조합: {soft_res['combination']}")
            print(f"   F1: {soft_res['soft_voting_f1']*100:.2f}% | Acc: {soft_res['soft_voting_acc']*100:.2f}%")
            soft_model_names = [all_models[i-1].split('/')[-1] for i in soft_res['combination']]
            print(f"   모델: {', '.join(soft_model_names)}")
        
        # 전체 비교 (정보만 출력)
        if best_hard and best_soft:
            hard_f1 = hard_res['hard_voting_f1']
            soft_f1 = soft_res['soft_voting_f1']
            
            print("\n" + "="*80)
            print("📊 Hard vs Soft 비교")
            print("="*80)
            
            if hard_f1 > soft_f1:
                winner = "Hard Voting"
                winner_f1 = hard_f1
                winner_acc = hard_res['hard_voting_acc']
                diff = hard_f1 - soft_f1
            elif soft_f1 > hard_f1:
                winner = "Soft Voting"
                winner_f1 = soft_f1
                winner_acc = soft_res['soft_voting_acc']
                diff = soft_f1 - hard_f1
            else:
                winner = "동점"
                winner_f1 = hard_f1
                winner_acc = max(hard_res['hard_voting_acc'], soft_res['soft_voting_acc'])
                diff = 0
            
            print(f"   Hard: F1 {hard_f1*100:.2f}% | Acc {hard_res['hard_voting_acc']*100:.2f}%")
            print(f"   Soft: F1 {soft_f1*100:.2f}% | Acc {soft_res['soft_voting_acc']*100:.2f}%")
            print(f"\n   🏆 최종 우승: {winner}")
            print(f"      F1: {winner_f1*100:.2f}% | Acc: {winner_acc*100:.2f}%")
            if diff > 0:
                print(f"      격차: +{diff*100:.2f}%p")
            
            # 단일 모델 대비 향상도
            best_single_f1 = max(model_performances.values(), key=lambda x: x['f1_score'])['f1_score']
            improvement = winner_f1 - best_single_f1
            
            print(f"\n   📈 단일 모델 대비:")
            print(f"      단일 모델 최고: {best_single_f1*100:.2f}%")
            print(f"      앙상블 최고:    {winner_f1*100:.2f}%")
            print(f"      향상도:         {improvement*100:+.2f}%p")
        
        print(f"\n✓ 상세 결과 저장: {results_file}")
        
        # ✨ test.csv로 Hard/Soft 최고 조합 JSONL 생성
        print("\n" + "="*80)
        print("📦 제출용 JSONL 파일 생성 중...")
        print("="*80)
        
        test_path = os.path.join(args.dataset_dir, "test.csv")
        if os.path.exists(test_path):
            submission_time = datetime.now().strftime("%m_%d_%H_%M")
            
            # Hard 최고 조합 JSONL
            if best_hard:
                hard_name, hard_res = best_hard
                hard_models = hard_res['models']
                print(f"\n1️⃣  Hard Voting 최고 조합 → 제출 파일 생성")
                print(f"   조합: {hard_res['combination']}")
                print(f"   모델: {', '.join([m.split('/')[-1] for m in hard_models])}")
                try:
                    ensemble_inference(
                        args, hard_models, submission_time + "_hard_best", 
                        "test.csv", save_jsonl=True
                    )
                    print(f"   ✅ hard_voting_ensemble_{len(hard_models)}_{submission_time}_hard_best.jsonl")
                except Exception as e:
                    print(f"   ❌ 생성 실패: {str(e)[:50]}")
            
            # Soft 최고 조합 JSONL
            if best_soft:
                soft_name, soft_res = best_soft
                soft_models = soft_res['models']
                print(f"\n2️⃣  Soft Voting 최고 조합 → 제출 파일 생성")
                print(f"   조합: {soft_res['combination']}")
                print(f"   모델: {', '.join([m.split('/')[-1] for m in soft_models])}")
                try:
                    ensemble_inference(
                        args, soft_models, submission_time + "_soft_best", 
                        "test.csv", save_jsonl=True
                    )
                    print(f"   ✅ soft_voting_ensemble_{len(soft_models)}_{submission_time}_soft_best.jsonl")
                except Exception as e:
                    print(f"   ❌ 생성 실패: {str(e)[:50]}")
            
            print("\n" + "="*80)
            print("✅ 제출용 파일 생성 완료!")
            print("="*80)
            print("📁 ./prediction/ 폴더에서 다음 파일을 확인하세요:")
            print(f"   1. hard_voting_ensemble_{len(hard_models)}_{submission_time}_hard_best.jsonl")
            print(f"   2. soft_voting_ensemble_{len(soft_models)}_{submission_time}_soft_best.jsonl")
            print("\n💡 리더보드 제출 팁:")
            if winner == "Hard Voting":
                print(f"   → Hard가 dev.csv에서 더 높았으니 Hard 먼저 제출 추천!")
            elif winner == "Soft Voting":
                print(f"   → Soft가 dev.csv에서 더 높았으니 Soft 먼저 제출 추천!")
            else:
                print(f"   → 둘 다 동점이니 둘 다 제출해보세요!")
        else:
            print(f"\n⚠️  test.csv를 찾을 수 없습니다: {test_path}")
    else:
        print("\n✗ 앙상블 결과가 없습니다.")


def run_test_submission(args, all_models):
    """test.csv에 대해 최종 제출용 추론 실행"""
    print_header("테스트 데이터 제출용 추론")
    
    test_path = os.path.join(args.dataset_dir, "test.csv")
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found!")
        return
    
    print("1. 단일 모델들로 test.csv 추론 중...")
    
    for i, model_name in enumerate(all_models, 1):
        model_short = model_name.split('/')[-1]
        print(f"\n[{i}/{len(all_models)}] {model_short}...", end=' ')
        
        try:
            acc, f1, record_time = single_model_inference(args, model_name, "test.csv", save_jsonl=True)
            print(f"✓ JSONL 생성됨")
        except Exception as e:
            print(f"✗ 실패")
    
    print("\n2. 앙상블 추론 중...")
    record_time = datetime.now().strftime("%m_%d_%H_%M")
    
    try:
        ensemble_inference(args, all_models, record_time, "test.csv", save_jsonl=True)
        print("✓ 앙상블 JSONL 생성됨")
    except Exception as e:
        print(f"✗ 앙상블 실패")
    
    print_header("제출용 파일 생성 완료")
    print("./prediction/ 폴더 확인:")
    print("- 단일 모델: [모델명]_[시간].jsonl")
    print("- 앙상블: hard_voting_ensemble_[개수]_[시간].jsonl")
    print("           soft_voting_ensemble_[개수]_[시간].jsonl")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    all_models = [
        "team-lucid/deberta-v3-base-korean",
        "monologg/kobert",
        "beomi/KcELECTRA-base-v2022",
        "snunlp/KR-ELECTRA-discriminator",
        "kykim/electra-kor-base"
    ]
    
    if args.mode == "train":
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DISABLED", None)
        
        print("="*80)
        print(f"모델 학습: {args.model_name}")
        print("="*80)
        print(f"• Run Name:      {args.run_name}")
        print(f"• Epochs:        {args.epochs}")
        print(f"• Batch Size:    {args.batch_size}")
        print(f"• Learning Rate: {args.lr}")
        print(f"• Dataset:       {args.dataset_dir}")
        print("="*80)
        
        import wandb
        print("\n📊 WandB 초기화 중...")
        wandb.init(
            project="ssac_hate_speech",
            name=args.run_name,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_len": args.max_len,
            }
        )
        print(f"📈 WandB 대시보드: {wandb.run.get_url()}")
        print("="*80)
        
        train(args, silent=False)
        wandb.finish()
        
        print("\n" + "="*80)
        print("✓ 학습 완료!")
        print("="*80)
        
    else:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        
        if args.mode == "single_inference":
            model_performances = evaluate_single_models(args, all_models)
                    
        elif args.mode == "advanced_ensemble":
            print_header("한국어 혐오 표현 분류 모델 앙상블 시스템")
            print("모델 목록:")
            for i, model in enumerate(all_models, 1):
                print(f"   {i}. {model}")
            run_advanced_ensemble(args, all_models)
        
        elif args.mode == "test_submission":
            run_test_submission(args, all_models)
        
        else:
            print(f"오류: 알 수 없는 모드 '{args.mode}'")
