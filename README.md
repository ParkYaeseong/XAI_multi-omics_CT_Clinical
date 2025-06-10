# 통합 암 분석: 다중 오믹스, 임상 데이터 및 CT 영상 분석 프로젝트 (with XAI)

## 1\. 프로젝트 개요

본 프로젝트는 암 환자의 다중 오믹스(Multi-omics) 데이터, 임상 정보(Clinical Data), 그리고 CT 영상 데이터를 통합적으로 분석하여 암의 특성을 예측하고 해석하는 파이프라인을 제공합니다. 프로젝트는 크게 두 가지 핵심 워크플로우로 구성됩니다.

1.  **오믹스/임상 데이터 기반 머신러닝 분석**: 유전체, 전사체, 단백체 등 다중 오믹스 데이터와 환자의 임상 정보를 결합하여 샘플의 상태(암 vs 정상)를 분류하고, SHAP(SHapley Additive exPlanations)을 이용해 모델의 예측 근거를 설명합니다.
2.  **CT 영상 데이터 기반 딥러닝 분석**: DICOM 형식의 CT 영상을 NIfTI 형식으로 변환하고 공간적으로 정규화한 뒤, 최신 딥러닝 모델인 SwinUNETR을 사용하여 암 병기(초기/중기/말기)를 분류합니다. Grad-CAM을 통해 모델이 영상의 어떤 부분을 보고 예측했는지 시각적으로 설명합니다.

이 두 워크플로우를 통해 단일 데이터 소스로는 알 수 없었던 복합적인 암의 특성을 이해하고, 해석 가능한(XAI) 분석 결과를 도출하는 것을 목표로 합니다.

## 2\. 주요 특징

  * **다중 모달리티 데이터 처리**: 다중 오믹스, 임상 정보, 3D 의료 영상(CT) 등 이종 데이터를 처리하는 파이프라인 제공
  * **두 가지 독립적인 분석 워크플로우**:
      * **머신러닝 파이프라인**: `scikit-learn`, `XGBoost`, `LightGBM` 등 다양한 모델을 활용한 분류 및 교차 검증
      * **딥러닝 파이프라인**: `MONAI`와 `PyTorch`를 활용한 3D 의료 영상 분류
  * **해석 가능한 AI (XAI)**:
      * **SHAP**: 오믹스/임상 데이터 모델의 특징 중요도 분석
      * **Grad-CAM**: CT 영상 모델의 예측 근거 시각화
  * **재현성 및 모듈성**: 데이터 전처리, 모델 학습, 결과 분석 과정을 스크립트별로 모듈화하여 코드의 재사용성과 재현성을 높였습니다.

## 3\. 프로젝트 아키텍처 및 워크플로우

본 프로젝트는 아래와 같이 두 개의 독립적인 워크플로우로 진행됩니다.

\<details\>
\<summary\>워크플로우 상세 설명 (텍스트)\</summary\>

### 워크플로우 A: 오믹스 & 임상 데이터 분석

1.  **입력**: 원본 오믹스 데이터 파일들(유전자 발현, 돌연변이, CNV 등), 원본 임상 데이터 파일
2.  **`process_omics_clinic.py` 실행**:
      * 각 오믹스 데이터의 특징(유전자, 돌연변이 등)을 추출하고 정규화
      * 임상 데이터에서 주요 특징을 선택하고 전처리 (결측치 처리, 스케일링, 원-핫 인코딩)
      * 고차원 오믹스 데이터에 PCA를 적용하여 차원 축소
      * 전처리된 오믹스 및 임상 특징을 환자 ID 기준으로 병합하여 최종 특징 행렬(X)과 타겟(y) 파일 생성 (`final_sample_based_features_clin_omics_revised.csv`, `..._target.csv`)
3.  **`xai_clinic_omics.py` 실행**:
      * `process_omics_clinic.py`에서 생성된 특징/타겟 파일을 입력으로 사용
      * 로지스틱 회귀, 랜덤 포레스트, XGBoost 등 다양한 머신러닝 모델 교차 검증 수행
      * 모델별 성능(정확도, F1, AUC 등) 평가 및 비교
      * SHAP 라이브러리를 사용해 모델의 예측에 중요한 영향을 미친 특징 분석 및 시각화
      * PCA로 생성된 주요 주성분에 대해 원본 특징의 기여도(Loading)를 분석하여 시각화

### 워크플로우 B: CT 영상 데이터 분석

1.  **입력**: 원본 DICOM CT 영상 파일, 원본 임상 데이터 파일
2.  **`ct_clinic.py` 실행 (사전 준비 단계)**:
      * 원본 임상 데이터에서 암 병기(Stage) 정보만 추출하고, 다양한 표기법(e.g., 'Stage IIA', 'IIA')을 표준화된 레이블(e.g., 'Stage IIA')로 정리
      * 환자 ID와 표준화된 병기 레이블이 포함된 `merged_clinical_data_final_preprocessed.csv` 파일 생성
3.  **`preprocess_dicom_to_nifti.py` 실행**:
      * `ct_clinic.py`에서 생성된 임상 레이블 파일을 참조하여, DICOM CT 영상에 해당하는 병기 정보 연결
      * MONAI 프레임워크를 사용해 각 DICOM 시리즈를 3D NIfTI 파일(`.nii.gz`)로 변환
      * 모든 영상의 해상도(Spacing)와 방향(Orientation)을 표준화 (e.g., 1.5x1.5x2.0mm, RAS)
      * 전처리된 NIfTI 파일 경로와 해당 환자의 레이블 정보가 담긴 `manifest.csv` 파일 생성
4.  **`train(unetr).py` 실행**:
      * `manifest.csv`를 입력으로 받아 데이터셋 구성
      * 암 병기를 '초기', '중기', '말기' 3개의 대분류로 재정의
      * SwinUNETR 모델(사전 학습된 가중치 사용 가능)을 이용해 3D CT 영상의 병기 분류 학습
      * 데이터 증강(Augmentation) 기법 적용하여 모델 성능 및 일반화 능력 향상
      * 학습 완료 후, 테스트셋에 대한 최종 성능 평가
      * Grad-CAM을 이용해 모델이 예측 시 활성화된 영역을 3D CT 영상 위에 시각화

\</details\>

## 4\. 디렉토리 구조

프로젝트 실행 전, 아래와 같은 디렉토리 구조를 권장합니다.

```
/your_project_root/
├── data/
│   ├── raw_clinical/
│   │   └── clinical_data.csv         # 원본 임상 데이터 파일
│   ├── raw_omics/
│   │   ├── merged_gene_expression_cancer.csv
│   │   ├── merged_gene_expression_normal.csv
│   │   └── ...                       # 기타 오믹스 데이터 파일들
│   └── dicom_ct/                     # 원본 DICOM 파일 루트 폴더
│       ├── TCGA-XX-YYYY/
│       │   └── ...
│       └── ...
├── process_omics_clinic.py
├── xai_clinic_omics.py
├── ct_clinic.py
├── preprocess_dicom_to_nifti.py
├── train(unetr).py
├── ct_classification_train_with_xai_using_nifti.py # 참고용 이전 버전 스크립트
└── README.md
```

**실행 후 생성되는 파일 및 폴더:**

  * `final_sample_based_features_clin_omics_revised.csv`: (A) 전처리된 오믹스/임상 특징 데이터
  * `final_sample_based_target_clin_omics_revised.csv`: (A) 특징 데이터에 대한 타겟(정상/암)
  * `pca_models_and_features/`: (A) PCA 모델 및 원본 특징 정보 저장
  * `modeling_plots/`: (A) SHAP, 모델 성능 등 시각화 결과 저장
  * `merged_clinical_data_final_preprocessed.csv`: (B) 표준화된 병기 레이블 파일
  * `preprocessed_nifti_data/`: (B) 전처리된 NIfTI 파일 및 `manifest.csv` 저장
  * `classification_results_SwinUNETR_.../`: (B) 딥러닝 모델, 로그, 최종 성능 지표 저장
  * `xai_gradcam_outputs_SwinUNETR_.../`: (B) Grad-CAM 시각화 결과 저장

## 5\. 설치 및 준비

### 5.1. 사전 요구사항

  * Python 3.8 이상
  * NVIDIA GPU (딥러닝 워크플로우에 강력히 권장)
  * CUDA 및 cuDNN

### 5.2. 라이브러리 설치

아래 명령어를 사용하여 필요한 Python 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

\<details\>
\<summary\>requirements.txt 예시\</summary\>

```
pandas
numpy
scikit-learn
joblib
xgboost
lightgbm
shap
matplotlib
seaborn
SimpleITK
monai[nibabel, tqdm]
torch
torchaudio
torchvision
imblearn
```

\</details\>

## 6\. 사용 방법

### 6.1. 스크립트 설정

각 Python 스크립트 상단에는 파일 경로 등 사용자가 설정해야 하는 변수들이 정의되어 있습니다. 코드를 실행하기 전, 자신의 환경에 맞게 이 경로들을 반드시 수정해야 합니다.

**★ 주요 설정 변수 ★**

  * `process_omics_clinic.py`: `CLINICAL_DATA_FILE`, `omics_files_info` 내 파일 경로
  * `xai_clinic_omics.py`: `feature_file`, `target_file` 경로
  * `ct_clinic.py`: `clinical_data_for_ct_path`
  * `preprocess_dicom_to_nifti.py`: `CT_ROOT_DIR`, `CLINICAL_DATA_FILE`
  * `train(unetr).py`: `MANIFEST_FILE_PATH`, `PRETRAINED_WEIGHTS_PATH`

### 6.2. 실행 순서

#### 워크플로우 A: 오믹스 & 임상 데이터 분석

1.  **데이터 전처리**: `process_omics_clinic.py`를 실행하여 특징 및 타겟 데이터를 생성합니다.
    ```bash
    python process_omics_clinic.py
    ```
2.  **모델 학습 및 XAI**: `xai_clinic_omics.py`를 실행하여 모델을 학습하고 해석합니다.
    ```bash
    python xai_clinic_omics.py
    ```

#### 워크플로우 B: CT 영상 데이터 분석

1.  **임상 레이블 표준화**: `ct_clinic.py`를 실행하여 CT 영상에 사용할 병기 레이블 파일을 생성합니다.
    ```bash
    python ct_clinic.py
    ```
2.  **DICOM to NIfTI 전처리**: `preprocess_dicom_to_nifti.py`를 실행하여 NIfTI 데이터셋과 manifest 파일을 구축합니다.
    ```bash
    python preprocess_dicom_to_nifti.py
    ```
3.  **딥러닝 모델 학습 및 XAI**: `train(unetr).py`를 실행하여 CT 영상 분류 모델을 학습하고 Grad-CAM 분석을 수행합니다.
    ```bash
    python train(unetr).py
    ```

## 7\. 스크립트 상세 설명

  * **`process_omics_clinic.py`**: 여러 종류의 오믹스 데이터와 임상 데이터를 통합하여 머신러닝 모델이 학습할 수 있는 형태의 특징 행렬로 가공합니다. PCA를 이용한 차원 축소, 결측치 처리, 정규화, 인코딩 등 복잡한 전처리 과정을 자동화합니다.
  * **`xai_clinic_omics.py`**: 전처리된 데이터를 바탕으로 다양한 머신러닝 분류 모델의 성능을 교차 검증을 통해 평가하고, SHAP 분석을 통해 예측에 대한 각 특징의 기여도를 정량적으로 보여줍니다.
  * **`ct_clinic.py`**: CT 영상 분류 모델의 정답지로 사용할 임상 데이터의 '암 병기' 정보를 추출하고, 다양한 표기법을 일관된 형태로 표준화하는 역할을 수행하는 보조 스크립트입니다.
  * **`preprocess_dicom_to_nifti.py`**: 의료 영상 표준인 DICOM 파일을 딥러닝에서 다루기 쉬운 NIfTI 형식으로 변환합니다. MONAI를 사용하여 모든 영상의 해상도와 물리적 방향을 통일시켜 모델 학습의 안정성을 높입니다.
  * **`train(unetr).py`**: Swin Transformer 기반의 최신 의료 영상 분할/분류 모델인 SwinUNETR을 사용하여 3D CT 영상의 암 병기를 분류합니다. 데이터 증강, 동적 분류 헤드 구성, 사전 학습된 가중치 활용, Grad-CAM 기반의 XAI 시각화 등 고급 딥러닝 기술이 적용되어 있습니다.
  * `ct_classification_train_with_xai_using_nifti.py`: `ResNet34` 기반의 NIfTI 분류 모델 학습 스크립트로, 현재는 `train(unetr).py`가 더 발전된 버전입니다. 참고용으로 포함되어 있습니다.

## 8\. 라이선스

본 프로젝트는 [MIT 라이선스](https://www.google.com/search?q=LICENSE)를 따릅니다.


## 9. 참고 기술 (Referenced Technologies)

본 프로젝트는 최신 데이터 과학 및 딥러닝 기술을 기반으로 구현되었습니다. 각 워크플로우에서 활용된 주요 기술 스택은 다음과 같습니다.

### 딥러닝 & 의료 영상 (Deep Learning & Medical Imaging)

* **`PyTorch`**: 딥러닝 모델의 구현, 학습 및 추론을 위한 핵심 프레임워크로 사용됩니다.
* **`MONAI`**: 의료 영상 딥러닝을 위한 오픈소스 프레임워크로, 데이터 로딩(`LoadImageD`), 전처리(`SpacingD`, `ResizeD`), 데이터 증강(`RandFlipd`, `RandRotate90d`) 및 3D 모델(`SwinUNETR`) 구현에 핵심적으로 활용됩니다.
* **`SwinUNETR`**: Vision Transformer(ViT)를 3D 의료 영상에 적용한 최신 아키텍처로, CT 영상의 병기 분류를 위한 메인 모델로 사용됩니다.
* **`SimpleITK` / `Nibabel`**: DICOM 및 NIfTI와 같은 의료 영상 파일을 읽고 쓰는 데 사용되는 라이브러리입니다. MONAI의 백엔드에서 주로 활용됩니다.

### 머신러닝 & 데이터 분석 (Machine Learning & Data Analysis)

* **`Scikit-learn`**: 데이터 전처리(스케일링, 인코딩), PCA를 통한 차원 축소, 파이프라인 구축, 모델 학습(로지스틱 회귀, SVM 등) 및 성능 평가의 전반적인 과정을 위해 사용됩니다.
* **`XGBoost` / `LightGBM`**: Gradient Boosting 기반의 고성능 머신러닝 라이브러리로, 오믹스/임상 데이터 분류 모델로 활용됩니다.
* **`Imbalanced-learn`**: SMOTE 알고리즘을 사용하여 데이터 불균형 문제를 처리하고 모델의 학습 안정성을 높이는 데 사용됩니다.

### 데이터 처리 및 시각화 (Data Handling & Visualization)

* **`Pandas`**: CSV 파일 입출력, 데이터프레임 조작 등 정형 데이터 처리를 위한 필수 라이브러리입니다.
* **`NumPy`**: 모든 데이터의 기반이 되는 다차원 배열을 효율적으로 처리하고 수치 연산을 수행하기 위해 사용됩니다.
* **`Matplotlib` / `Seaborn`**: 모델 성능 비교, 특징 중요도, PCA 로딩 등 분석 결과를 시각화하는 데 사용됩니다.

### 설명 가능한 AI (XAI)

* **`SHAP (SHapley Additive exPlanations)`**: 오믹스/임상 데이터 기반 머신러닝 모델의 예측 결과를 해석하고, 각 특징(Feature)의 중요도를 파악하기 위해 사용됩니다.
* **`Grad-CAM (Gradient-weighted Class Activation Mapping)`**: 딥러닝 모델이 CT 영상의 특정 영역을 어떻게 보고 병기를 예측했는지 히트맵(Heatmap) 형태로 시각화하여 모델의 판단 근거를 직관적으로 보여줍니다.