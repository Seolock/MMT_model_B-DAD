import sys
import logging

def parse_scores(filename):
    """Fairseq 결과 파일에서 점수(log-likelihood)와 ID를 추출"""
    scores = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Fairseq generate 출력 라인 중 H- 로 시작하는 줄이 점수 정보입니다.
            # 형식: H-{id} <tab> {score} <tab> {hypothesis}
            if line.startswith('H-'):
                try:
                    parts = line.strip().split('\t')
                    # parts[0]: H-123 형태 -> 123 추출
                    sample_id = int(parts[0].split('-')[1])
                    # parts[1]: 점수 (문자열) -> 실수형 변환
                    score = float(parts[1])
                    scores[sample_id] = score
                except Exception as e:
                    print(f"Warning: parsing error on line: {line.strip()} -> {e}")
    return scores

def main():
    model_path = sys.argv[1]
    correct_file = model_path+'/final_score_correct.log'
    incorrect_file = model_path+'/final_score_incorrect.log'

    logging.basicConfig(
        filename=model_path+"/commute.log",       # 저장할 로그 파일 이름
        level=logging.INFO,       # 로그 레벨
        format="%(asctime)s - %(message)s"
    )
    
    print(f"Loading scores from {correct_file}...")
    logging.info(f"Loading scores from {correct_file}...")
    scores_corr = parse_scores(correct_file)
    print(f"Loading scores from {incorrect_file}...")
    logging.info(f"Loading scores from {incorrect_file}...")
    scores_incorr = parse_scores(incorrect_file)
    
    # 두 파일 모두에 존재하는 ID만 추려서 평가 (데이터 정렬 확인용)
    common_ids = set(scores_corr.keys()) & set(scores_incorr.keys())
    
    if len(common_ids) == 0:
        print("Error: No matching sample IDs found between the two files.")
        print("Check if the generate output format is correct.")
        return

    print(f"Evaluated {len(common_ids)} samples.")
    logging.info(f"Evaluated {len(common_ids)} samples.")
    
    correct_count = 0
    total_count = 0
    
    for sid in common_ids:
        s_corr = scores_corr[sid]
        s_incorr = scores_incorr[sid]
        
        # CoMMuTE 평가 로직:
        # 정답 문장의 점수(Log-Likelihood)가 오답 문장보다 높으면 정답 맞춤
        if s_corr > s_incorr:
            correct_count += 1
        total_count += 1
        
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("-" * 40)
    print(f"Final CoMMuTE Accuracy: {accuracy:.2f}%")
    print("-" * 40)
    logging.info("-" * 40)
    logging.info(f"Final CoMMuTE Accuracy: {accuracy:.2f}%")
    logging.info("-" * 40)

if __name__ == "__main__":
    main()