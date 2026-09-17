# Dual-Attention Decoder

멀티모달 기계 번역을 위한 듀얼 어텐션 디코더
Dual-Attention Decoder for Multimodal Machine Translation

[fairseq_mmt](https://github.com/libeineu/fairseq_mmt) (ACL 2022, *On Vision Features in Multimodal Machine Translation*) 포크 기반으로 구현됨.

---

## 한 번에 실행 (훈련 + 테스트)

```bash
bash shell.sh <실험이름>
```

예: `bash shell.sh entropy_test`

이 한 줄이 학습부터 평가까지 전부 돌립니다. 결과는 `checkpoints/multi30k-en2de-vit_base_patch16_384-<실험이름>/` 에 쌓입니다.

`shell.sh`가 순서대로 하는 일:

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1 | `fairseq-train` 학습 | `checkpoint*.pt`, `train.log` |
| 2 | 마지막 10개 체크포인트 평균 | `last10.ensemble.pt` |
| 3 | test / test1 / test2 번역 + BLEU | `translation_test*.log`, `hypo_test*.sorted` |
| 4 | METEOR 측정 | `meteor_test*.log` |
| 5 | CoMMuTE correct/incorrect 스코어링 | `commute_result/final_score_*.log` |
| 6 | CoMMuTE 정확도 계산 | `commute_result/commute.log` |

테스트셋 이름 대응: `test` = test2016, `test1` = test2017, `test2` = MSCOCO.

실행한 `shell.sh` 사본이 `save_dir`에 함께 복사되므로, 나중에 그 실험의 하이퍼파라미터를 그대로 확인할 수 있습니다.

### 자주 바꾸는 설정

`shell.sh` 상단에서 직접 수정합니다.

```bash
device=6                        # 사용할 GPU
image_feat=vit_tiny_patch16_384 # 이미지 피처 (dim은 자동 매핑)
for task in multi30k-en2de;     # multi30k-en2fr 추가 가능
```

`image_feat` → 차원 매핑은 스크립트가 알아서 처리합니다:
`vit_tiny` 192 · `vit_small` 384 · `vit_base` 768 · `vit_large` 1024




## 개별 실행

| 스크립트 | 용도 |
|---|---|
| `bash test.sh` | 학습 없이 평가만. 상단의 `name`, `_image_feat`, `test` 변수를 직접 수정해서 사용 |
| `bash commute.sh <image_feat> <name>` | CoMMuTE만 따로 측정 |
| `bash map.sh` → `python3 map.py` | 토큰별 attention map 추출 및 시각화 |
| `python3 visual2.py` | ViT 패치 attention을 원본 이미지 위에 히트맵으로 오버레이 |
| `bash preprocess.sh` / `preprocess_mmt.sh` | 텍스트 바이너리화 (마스킹 없음 / 마스킹) |

---

## 준비물

### 환경

```bash
pip install --editable ./
```

PyTorch 1.9.1 · Python 3.9 · timm 0.4.12 · vizseq 0.1.15 · nltk 3.6.4 · sacrebleu 1.5.1


### 데이터

텍스트는 이미 바이너리화되어 있습니다.

```
data-bin/
├── multi30k.en-de/          # train/valid/test/test1/test2
│   ├── correct/             # CoMMuTE 정답 번역
│   ├── incorrect/           # CoMMuTE 오답 번역
│   └── mask1~4, maskc, maskp   # 원본 논문의 probing 데이터
├── multi30k.en-de-auged/
└── multi30k.en-fr/
```

### 이미지 피처

`shell.sh`는 `~/<image_feat>/` 에서 피처를 찾습니다.

```
~/vit_tiny_patch16_384/
├── train.pth  valid.pth  test.pth  test1.pth  test2.pth
└── commute/en-de/          # CoMMuTE용
```

새로 추출하려면:

```bash
python3 scripts/get_img_feat.py --dataset train --model vit_base_patch16_384 --path ../flickr30k
python3 scripts/get_img_feat_detr.py --dataset train --path ../flickr30k
```

ViT는 timm 코드 수정이, DETR은 공식 코드 수정이 선행되어야 합니다 — `scripts/README.md` 참고.

---

