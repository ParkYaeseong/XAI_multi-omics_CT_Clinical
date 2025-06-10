# --- OpenMP 중복 라이브러리 로드 허용 ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import glob
import numpy as np
import pandas as pd
import monai
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD,
    ScaleIntensityRangePercentilesD, ResizeD, Compose,
    EnsureTypeD, RandFlipd, RandRotate90d, LambdaD,
    RandAffineD, RandGaussianNoiseD, RandAdjustContrastD
)
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.utils import set_determinism
from monai.visualize import GradCAM
from monai.networks.nets import SwinUNETR # UNETR, resnet34는 사용하지 않으므로 일단 제외
from monai.networks.blocks.mlp import MLPBlock # MLPBlock은 SwinTransformer 내부에서 사용될 수 있음
from monai.data import MetaTensor

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
import logging
import traceback
import copy
import matplotlib.pyplot as plt
from collections import Counter


# --- 설정값 ---
MANIFEST_FILE_PATH = os.path.join(os.getcwd(), "preprocessed_nifti_data", "preprocessed_manifest.csv")
PATIENT_ID_COL = 'bcr_patient_barcode'
LABEL_COL = 'original_label'

BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "classification_results_SwinUNETR_BTCV_3stage_v1_dynamic_head") # 폴더명 변경
MODEL_CHECKPOINT_NAME = "best_ct_classification_SwinUNETR_BTCV_3stage_model_v1_dynamic_head.pth" # 모델명 변경
XAI_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "xai_gradcam_outputs_SwinUNETR_BTCV_3stage_v1_dynamic_head") # XAI 폴더명 변경
LOG_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "ct_classification_SwinUNETR_BTCV_3stage_log_v1_dynamic_head.txt") # 로그 파일명 변경
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)

RESIZE_SHAPE = (96, 96, 96) # 이 값을 CTClassifier와 SwinUNETR 초기화 시 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
set_determinism(RANDOM_SEED)

MODEL_ARCH = "swinunetr"
MODEL_FEATURE_SIZE = 48 # SwinUNETR 초기 feature_size

PRETRAINED_WEIGHTS_PATH = r"C:/Users/21/Desktop/TCGA-OV/model.pt" # 사용자 경로

NUM_CLASSES = 3
NEW_STAGE_CATEGORIES_MAP = {0: "초기", 1: "중기", 2: "말기"}

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_EPOCHS = 100
K_FOLDS = 0
TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.15
FREEZE_FEATURE_EXTRACTOR_EPOCHS = 10
NUM_WORKERS_DATALOADER = 0

XAI_NUM_SAMPLES_TO_VISUALIZE = 5
XAI_TARGET_LAYER_NAME = "layers3.0.blocks.1.norm1"

# --- 로거 설정 ---
logger = logging.getLogger("train_nifti_SwinUNETR_BTCV_3stage_dynamic_head") # 로거 이름 변경
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s %(name)s - %(module)s:%(lineno)d - %(message)s'))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s'))
logger.addHandler(stream_handler)

# --- 사용자 정의: 원본 레이블 -> 대분류("초기", "중기", "말기") 매핑 ---
ORIGINAL_STRING_LABEL_TO_NEW_STAGE_STRING_MAP = {
    "Stage IA": "초기", "Stage IB": "초기", "Stage IC": "초기",
    "STAGE IC": "초기",
    "Stage IIA": "중기", "Stage IIB": "중기", "Stage IIC": "중기",
    "STAGE IIC": "중기",
    "Stage IIIA": "말기", "Stage IIIB": "말기", "Stage IIIC": "말기",
    "Stage IV": "말기",
}
logger.info(f"사용자 정의 원본 레이블 -> 대분류 매핑: {ORIGINAL_STRING_LABEL_TO_NEW_STAGE_STRING_MAP}")


# --- 오류 처리 Dataset 및 Collate 함수 정의 ---
class ErrorTolerantDataset(Dataset):
    def __init__(self, data, transform, patient_id_col_name='bcr_patient_barcode'):
        super().__init__(data, transform)
        self.patient_id_col_name = patient_id_col_name

    def _transform(self, index):
        item_dict_original = self.data[index]
        try:
            if self.transform:
                return self.transform(item_dict_original.copy())
            return item_dict_original.copy()
        except Exception as e:
            pid = item_dict_original.get(self.patient_id_col_name, "N/A_pid_in_dataset")
            image_path = item_dict_original.get("image", "N/A_path_in_dataset")
            error_msg_short = str(e).splitlines()[0]
            logger.warning(f"Error transforming item PID: {pid}, Path: {image_path} in ErrorTolerantDataset - returning None: {type(e).__name__} - {error_msg_short}")
            logger.debug(f"Full Traceback for PID {pid} (Path: {image_path}):\n{traceback.format_exc()}")
            return None

def safe_list_data_collate(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    return list_data_collate(valid_batch)


# --- 데이터 로드 및 전처리 함수 ---
def load_and_prepare_data_from_manifest(manifest_file_path, patient_id_col, string_label_col_in_csv,
                                         original_label_to_new_stage_map, new_stage_map_for_xai):
    try:
        manifest_df = pd.read_csv(manifest_file_path)
        logger.info(f"Manifest 파일 로드 완료: {manifest_file_path}, 총 {len(manifest_df)}개 항목.")
    except FileNotFoundError:
        logger.error(f"Manifest 파일({manifest_file_path})을 찾을 수 없습니다."); return [], None

    required_cols = ['image_nifti_path', patient_id_col, string_label_col_in_csv]
    for col in required_cols:
        if col not in manifest_df.columns:
            logger.error(f"Manifest 파일에 필요한 컬럼 '{col}'이 없습니다."); return [], None

    new_stage_string_to_int_map = {v: k for k, v in new_stage_map_for_xai.items()}
    logger.info(f"대분류 문자열 -> 정수 레이블 매핑: {new_stage_string_to_int_map}")

    all_data_dicts = []
    num_excluded_by_mapping = 0
    original_label_counts = Counter()
    new_stage_label_counts = Counter()

    for _, row in manifest_df.iterrows():
        img_path = row['image_nifti_path']
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            logger.warning(f"NIfTI 파일 경로가 유효하지 않거나 찾을 수 없습니다: {img_path}. 이 항목은 건너뜁니다. (PID: {row[patient_id_col]})")
            continue

        current_original_label_str = str(row[string_label_col_in_csv]).strip()
        original_label_counts[current_original_label_str] += 1

        new_stage_str = original_label_to_new_stage_map.get(current_original_label_str)

        if new_stage_str is None:
            num_excluded_by_mapping += 1
            logger.debug(f"PID {row[patient_id_col]}: 원본 레이블 '{current_original_label_str}'에 대한 대분류 매핑 규칙을 찾을 수 없어 제외됩니다.")
            continue

        final_model_label = new_stage_string_to_int_map.get(new_stage_str)
        if final_model_label is None:
            logger.error(f"PID {row[patient_id_col]}: 대분류 문자열 '{new_stage_str}'을 정수 레이블로 변환할 수 없습니다. 이 항목은 제외됩니다.")
            num_excluded_by_mapping +=1
            continue

        new_stage_label_counts[new_stage_str] += 1
        data_dict = {
            "image": img_path,
            "label": torch.tensor(final_model_label, dtype=torch.long),
            patient_id_col: row[patient_id_col],
            "original_label_str": current_original_label_str,
            "new_stage_label_str": new_stage_str
        }
        all_data_dicts.append(data_dict)

    logger.info(f"원본 레이블 분포 (매핑 전): {original_label_counts}")
    logger.info(f"총 {num_excluded_by_mapping}개의 항목이 대분류 매핑 규칙에 없어 제외되었습니다.")
    if not all_data_dicts:
        logger.error("Manifest에서 유효한 데이터를 로드하지 못했습니다 (대분류 매핑 후).")
        return [], None

    logger.info(f"최종 학습 데이터 대분류 분포: {new_stage_label_counts}")
    return all_data_dicts, new_stage_map_for_xai


# --- MONAI Transforms 정의 ---
train_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),

    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),

    RandAffineD(
        keys=['image'],
        prob=0.5,
        rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
        scale_range=(0.15, 0.15, 0.15),
        translate_range=(10, 10, 5),
        padding_mode='border',
        mode="bilinear",
        device=DEVICE # 가능한 경우 GPU에서 증강 연산
    ),
    RandGaussianNoiseD(keys=['image'], prob=0.3, mean=0.0, std=0.05),
    RandAdjustContrastD(keys=['image'], prob=0.3, gamma=(0.7, 1.3)),

    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True),
    EnsureTypeD(keys=["image"], dtype=torch.float32),
    LambdaD(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else torch.as_tensor(x))
])

val_test_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True),
    EnsureTypeD(keys=["image"], dtype=torch.float32),
    LambdaD(keys=["image"], func=lambda x: x.as_tensor() if hasattr(x, 'as_tensor') else torch.as_tensor(x))
])

# --- 모델 정의 (CTClassifier) ---
class CTClassifier(nn.Module):
    # 생성자에서 img_size 인자 제거
    def __init__(self, num_classes_final, model_arch="swinunetr",
                 in_channels=1, feature_size=48, # feature_size는 SwinUNETR의 초기 feature_size
                 pretrained_weights_path=None, freeze_feature_extractor=True):
        super().__init__()
        self.model_arch = model_arch
        self.in_channels = in_channels
        self.swin_unetr_initial_feature_size = feature_size # 명확성을 위해 변수명 변경

        logger.info(f"모델 초기화: {self.model_arch} 사용, 최종 클래스 수: {num_classes_final}")
        num_ftrs_for_head = None

        if self.model_arch == "swinunetr":
            # SwinUNETR 생성 시 img_size=RESIZE_SHAPE 전달 (MONAI 1.3+에서는 경고 발생 가능)
            swin_unetr_base = SwinUNETR(
                img_size=RESIZE_SHAPE, # 전역 변수 RESIZE_SHAPE 사용
                in_channels=self.in_channels,
                out_channels=14, # 분류 작업에서는 디코더를 사용하지 않으므로 이 값은 중요하지 않음
                feature_size=self.swin_unetr_initial_feature_size,
                use_checkpoint=True,
            )
            self.feature_extractor = swin_unetr_base.swinViT
            self.pool = nn.AdaptiveAvgPool1d(1)

            # num_ftrs_for_head를 동적으로 결정
            # 모델 파라미터가 현재 어떤 device에 있는지 확인 (일반적으로 CPU로 초기화됨)
            try:
                temp_device = next(self.feature_extractor.parameters()).device
            except StopIteration: # 파라미터가 없는 모델 (예: nn.Identity)의 경우 대비
                temp_device = torch.device("cpu") # 기본값으로 CPU 사용
                logger.warning("Feature extractor에 파라미터가 없어 temp_device를 CPU로 설정합니다.")


            with torch.no_grad(): # 그래디언트 계산 비활성화
                # 더미 입력 생성 시 전역 변수 RESIZE_SHAPE 사용
                dummy_input = torch.randn(1, self.in_channels, *RESIZE_SHAPE, device=temp_device)
                
                # self.feature_extractor가 temp_device에 있는지 확인 (이미 같은 디바이스라면 이동 안 함)
                self.feature_extractor.to(temp_device) 

                features_output = self.feature_extractor(dummy_input)
                
                if isinstance(features_output, (list, tuple)):
                    features_seq = features_output[-1]
                else:
                    features_seq = features_output

                if features_seq.dim() == 5: # (B, C, D_f, H_f, W_f)
                    features_seq = features_seq.flatten(2).transpose(1,2) # B, N, C
                
                num_ftrs_for_head = features_seq.size(-1) # 마지막 차원 (임베딩 차원) 크기를 가져옴
                logger.info(f"SwinUNETR (swinViT) 특징 추출기. 동적으로 결정된 최종 특징 차원 (num_ftrs_for_head): {num_ftrs_for_head}")
        
        # 주: UNETR, ResNet34 등의 다른 아키텍처를 다시 사용하려면,
        # 해당 아키텍처에 맞는 num_ftrs_for_head 결정 로직 (동적 또는 수식)이 필요합니다.
        elif self.model_arch == "unetr":
            logger.error("UNETR 아키텍처는 현재 이 스크립트에서 num_ftrs_for_head 동적 결정 로직이 구현되지 않았습니다.")
            raise NotImplementedError("UNETR num_ftrs_for_head 결정 필요")
        elif self.model_arch == "resnet34":
            logger.error("ResNet34 아키텍처는 현재 이 스크립트에서 num_ftrs_for_head 동적 결정 로직이 구현되지 않았습니다.")
            raise NotImplementedError("ResNet34 num_ftrs_for_head 결정 필요")
        else:
            raise ValueError(f"지원되지 않는 모델 아키텍처: {self.model_arch}")

        # --- 사전 훈련된 가중치 로드 로직 ---
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                # 가중치 로드 시 CPU로 먼저 로드하는 것이 안전할 수 있음 (특히 다른 환경에서 저장된 경우)
                state_dict_bundle = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False) # weights_only=True 권장
                
                state_dict = state_dict_bundle
                if isinstance(state_dict_bundle, dict) and 'state_dict' in state_dict_bundle:
                    state_dict = state_dict_bundle['state_dict']
                elif isinstance(state_dict_bundle, dict) and not any(isinstance(v, torch.Tensor) for v in state_dict_bundle.values()):
                    # MONAI Bundle에서 'model' 또는 'network' 키 아래에 state_dict가 있을 수 있음
                    if 'model' in state_dict_bundle: state_dict = state_dict_bundle['model']
                    elif 'network' in state_dict_bundle: state_dict = state_dict_bundle['network']

                feature_extractor_state_dict = {}
                expected_prefix = "swinViT."
                module_ddp_prefix = "module." # DataParallel/DistributedDataParallel 학습 시 추가될 수 있는 접두사

                keys_loaded_count = 0
                for k, v in state_dict.items():
                    key_to_check = k
                    if key_to_check.startswith(module_ddp_prefix):
                        key_to_check = key_to_check[len(module_ddp_prefix):]

                    if key_to_check.startswith(expected_prefix):
                        new_key = key_to_check[len(expected_prefix):]
                        feature_extractor_state_dict[new_key] = v
                        keys_loaded_count+=1
                
                if keys_loaded_count > 0:
                    logger.info(f"{keys_loaded_count}개의 가중치를 '{expected_prefix}' 접두사를 사용하여 추출했습니다.")
                    # self.feature_extractor는 현재 CPU에 있음 (모델 전체를 DEVICE로 옮기기 전)
                    missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(feature_extractor_state_dict, strict=False)
                    logger.info(f"모델({self.model_arch}의 특징 추출기 - SwinViT)에 사전 훈련된 가중치 로드 완료 (로드 위치: CPU): {pretrained_weights_path}")
                    if missing_keys: logger.warning(f"SwinViT 가중치 로드 시 누락된 키: {missing_keys}")
                    if unexpected_keys: logger.warning(f"SwinViT 가중치 로드 시 예상치 못한 키 (디코더 등일 수 있음): {unexpected_keys}")
                else:
                    logger.warning(f"'{expected_prefix}' 또는 '{module_ddp_prefix}{expected_prefix}' 접두사를 가진 키를 찾지 못했습니다. 특징 추출기 전체에 직접 로드를 시도합니다.")
                    cleaned_state_dict = {k.replace(module_ddp_prefix, ''): v for k,v in state_dict.items()}
                    missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(cleaned_state_dict, strict=False)
                    logger.info(f"모델({self.model_arch}의 특징 추출기 - SwinViT)에 사전 훈련된 가중치 (접두사 없이) 직접 로드 시도 완료 (로드 위치: CPU).")
                    if missing_keys: logger.warning(f"직접 로드 시 누락된 키 (SwinViT): {missing_keys}")
                    if unexpected_keys: logger.warning(f"직접 로드 시 예상치 못한 키 (SwinViT): {unexpected_keys}")

            except Exception as e: logger.error(f"사전 훈련된 가중치 로드 실패: {e}."); logger.debug(traceback.format_exc())
        elif pretrained_weights_path: logger.warning(f"사전 훈련된 가중치 파일을 찾을 수 없습니다: {pretrained_weights_path}.")
        else: logger.info("사전 훈련된 가중치 경로가 제공되지 않았습니다. 특징 추출기가 무작위로 초기화됩니다.")

        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("특징 추출기 동결됨.")

        if num_ftrs_for_head is None:
             raise ValueError("분류기 헤드를 위한 특징 수를 결정할 수 없습니다 (num_ftrs_for_head가 None).")

        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs_for_head, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_final)
        )
        logger.info(f"새로운 분류기 헤드 구성 (LayerNorm 사용): Linear({num_ftrs_for_head}, 256) -> ... -> Linear(256, {num_classes_final})")

    def forward(self, x):
        if self.model_arch == "swinunetr":
            features_output = self.feature_extractor(x)

            if isinstance(features_output, (list, tuple)):
                # logger.debug(f"SwinViT 출력은 리스트/튜플입니다 (길이: {len(features_output)}). 마지막 요소를 사용합니다.")
                features_seq = features_output[-1]
            else:
                features_seq = features_output

            if features_seq.dim() == 5 :
                # logger.debug(f"SwinViT 특징 (텐서)이 5D {features_seq.shape}입니다. flatten 및 transpose 적용합니다.")
                features_seq = features_seq.flatten(2).transpose(1,2)
            elif features_seq.dim() != 3:
                raise ValueError(f"SwinUNETR feature_extractor의 최종 특징 텐서 차원이 예상과 다릅니다: {features_seq.shape}. (B, N, C) 또는 (B, C, Df, Hf, Wf) 형태여야 합니다.")

            features = self.pool(features_seq.transpose(1, 2))
            features = features.contiguous().view(-1, features_seq.size(-1))

        else:
            raise ValueError(f"지원되지 않는 모델 아키텍처 forward: {self.model_arch}")

        return self.classifier_head(features)

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        logger.info("특징 추출기 전체 동결 해제됨.")

# --- 학습 및 평가 함수 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, current_num_classes, patient_id_col_key):
    model.train()
    running_loss = 0.0; correct_predictions = 0; total_predictions = 0
    all_labels_epoch = []; all_preds_proba_epoch = []

    prog_bar = tqdm(enumerate(dataloader), desc="Training", leave=False, total=len(dataloader))

    for batch_idx, batch_data in prog_bar:
        if batch_data is None: logger.warning(f"Skipping batch {batch_idx} in Training (all items failed transform or collate returned None)."); continue
        if not batch_data: logger.warning(f"Skipping batch {batch_idx} in Training because batch_data dictionary is empty."); continue

        inputs_tensor = batch_data.get("image")
        labels_tensor = batch_data.get("label")

        pids_in_batch_list = batch_data.get(patient_id_col_key, ["PID_Unknown_In_Batch"])
        pid_for_log = pids_in_batch_list[0] if isinstance(pids_in_batch_list, list) and len(pids_in_batch_list) > 0 else str(pids_in_batch_list)

        if inputs_tensor is not None and (torch.isnan(inputs_tensor).any() or torch.isinf(inputs_tensor).any()):
            logger.error(f"Batch {batch_idx} (First PID: {pid_for_log}) contains NaNs or Infs in input images! Skipping batch.")
            continue

        if inputs_tensor is None or (isinstance(inputs_tensor, torch.Tensor) and inputs_tensor.numel() == 0):
            logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: 'image' data is missing, None, or an empty tensor."); continue
        if labels_tensor is None:
            logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: 'label' data is missing or None."); continue

        if isinstance(inputs_tensor, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
            if inputs_tensor.size(0) == 0:
                logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: 'image' tensor batch size is 0."); continue
            if inputs_tensor.size(0) != labels_tensor.size(0):
                logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: Mismatched batch sizes for images ({inputs_tensor.size(0)}) and labels ({labels_tensor.size(0)})."); continue
            if labels_tensor.numel() == 0 and inputs_tensor.numel() > 0 :
                logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: 'label' tensor is empty while 'image' tensor is not."); continue
        elif isinstance(inputs_tensor, torch.Tensor) and not isinstance(labels_tensor, torch.Tensor):
            logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in Training: 'image' is a tensor, but 'label' is not (type: {type(labels_tensor)})."); continue

        inputs = inputs_tensor.to(device); labels = labels_tensor.to(device)

        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        running_loss += loss.item() * inputs.size(0); _, predicted_classes = torch.max(outputs, 1)
        correct_predictions += (predicted_classes == labels).sum().item(); total_predictions += labels.size(0)
        all_labels_epoch.extend(labels.cpu().numpy()); all_preds_proba_epoch.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
    prog_bar.close()

    epoch_loss = running_loss / total_predictions if total_predictions > 0 else float('inf')
    epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    epoch_auc = 0.0
    if total_predictions > 0 and len(all_labels_epoch) > 0 and current_num_classes >= 2 : # AUC 계산을 위해 최소 2개 클래스 필요
        try:
            unique_labels_in_epoch = np.unique(all_labels_epoch)
            if len(unique_labels_in_epoch) > 1: # 실제 레이블 종류가 2개 이상일 때만 AUC 계산
                # multi_class='ovr' 또는 'ovo', labels는 모든 가능한 클래스 [0, 1, ..., num_classes-1]
                roc_labels_range = list(range(current_num_classes))
                epoch_auc = roc_auc_score(all_labels_epoch, all_preds_proba_epoch, multi_class='ovr', average='macro', labels=roc_labels_range)
            else:
                logger.debug(f"훈련 중 AUC 계산 불가: 에포크 내 실제 레이블 종류 부족 ({len(unique_labels_in_epoch)}개). 클래스 수: {current_num_classes}")
        except ValueError as e:
            logger.warning(f"훈련 중 AUC 계산 오류: {e}. AUC는 0.0. Labels: {np.unique(all_labels_epoch)}. Probas shape: {np.array(all_preds_proba_epoch).shape}")
    return epoch_loss, epoch_acc, epoch_auc

def evaluate_model(model, dataloader, criterion, device, phase="Validation", current_num_classes=None, patient_id_col_key=None):
    if current_num_classes is None: current_num_classes = NUM_CLASSES

    model.eval(); running_loss = 0.0; correct_predictions = 0; total_predictions = 0
    all_labels_eval = []; all_predicted_classes_eval = []; all_preds_proba_eval = []
    prog_bar = tqdm(enumerate(dataloader), desc=phase, leave=False, total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch_data in prog_bar:
            if batch_data is None: logger.warning(f"Skipping batch {batch_idx} in {phase} (all items failed transform or collate returned None)."); continue
            if not batch_data: logger.warning(f"Skipping batch {batch_idx} in {phase} because batch_data dictionary is empty."); continue
            inputs_tensor = batch_data.get("image"); labels_tensor = batch_data.get("label")

            pids_in_batch_list = batch_data.get(patient_id_col_key, ["PID_Unknown_In_Batch"])
            pid_for_log = pids_in_batch_list[0] if isinstance(pids_in_batch_list, list) and len(pids_in_batch_list) > 0 else str(pids_in_batch_list)

            if inputs_tensor is not None and (torch.isnan(inputs_tensor).any() or torch.isinf(inputs_tensor).any()):
                logger.error(f"Batch {batch_idx} in {phase} (First PID: {pid_for_log}) contains NaNs or Infs in input images! Skipping batch.")
                continue
            if inputs_tensor is None or (isinstance(inputs_tensor, torch.Tensor) and inputs_tensor.numel() == 0): logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: 'image' data is missing, None, or an empty tensor."); continue
            if labels_tensor is None: logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: 'label' data is missing or None."); continue
            if isinstance(inputs_tensor, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
                if inputs_tensor.size(0) == 0: logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: 'image' tensor batch size is 0."); continue
                if inputs_tensor.size(0) != labels_tensor.size(0):
                    logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: Mismatched batch sizes for images ({inputs_tensor.size(0)}) and labels ({labels_tensor.size(0)})."); continue
                if labels_tensor.numel() == 0 and inputs_tensor.numel() > 0 : logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: 'label' tensor is empty while 'image' tensor is not."); continue
            elif isinstance(inputs_tensor, torch.Tensor) and not isinstance(labels_tensor, torch.Tensor):
                logger.warning(f"Skipping batch {batch_idx} (First PID: {pid_for_log}) in {phase}: 'image' is a tensor, but 'label' is not (type: {type(labels_tensor)})."); continue

            inputs = inputs_tensor.to(device); labels = labels_tensor.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0); _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == labels).sum().item(); total_predictions += labels.size(0)
            all_labels_eval.extend(labels.cpu().numpy()); all_predicted_classes_eval.extend(predicted_classes.cpu().numpy()); all_preds_proba_eval.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    prog_bar.close()
    eval_loss = running_loss / total_predictions if total_predictions > 0 else float('inf')
    eval_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    f1, precision, recall, auc_score_val = 0.0, 0.0, 0.0, 0.0
    metric_labels_range = list(range(current_num_classes))

    if total_predictions > 0 and len(all_labels_eval) > 0 and current_num_classes >=2: # AUC 등 계산 위해 최소 2개 클래스
        # 모든 메트릭에 대해 모든 클래스 레이블 사용 [0, 1, ..., num_classes-1]
        eval_metric_labels = metric_labels_range

        f1 = f1_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=eval_metric_labels)
        precision = precision_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=eval_metric_labels)
        recall = recall_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0, labels=eval_metric_labels)
        try:
            unique_labels_in_eval = np.unique(all_labels_eval)
            if len(unique_labels_in_eval) > 1: # 실제 레이블 종류가 2개 이상일 때만 AUC 계산
                auc_score_val = roc_auc_score(all_labels_eval, all_preds_proba_eval, multi_class='ovr', average='macro', labels=eval_metric_labels)
            else :
                logger.debug(f"{phase} 중 AUC 계산 불가: 실제 레이블 종류 부족 ({len(unique_labels_in_eval)}개). 클래스 수: {current_num_classes}")
        except ValueError as e:
            logger.warning(f"{phase} 중 AUC 계산 오류: {e}. AUC는 0.0. Labels: {np.unique(all_labels_eval)}. Probas shape: {np.array(all_preds_proba_eval).shape}. Eval Metric Labels: {eval_metric_labels}")

    conf_matrix_val = np.zeros((current_num_classes, current_num_classes), dtype=int)
    if total_predictions > 0 and metric_labels_range :
        try: # confusion_matrix는 labels 인자에 실제 데이터에 없는 클래스가 있어도 괜찮음
            conf_matrix_val = confusion_matrix(all_labels_eval, all_predicted_classes_eval, labels=metric_labels_range)
        except Exception as e_cm:
            logger.error(f"{phase} 중 혼동 행렬 생성 오류: {e_cm}")


    return {"loss": eval_loss, "accuracy": eval_acc, "f1_score": f1, "precision": precision, "recall": recall, "auc": auc_score_val, "confusion_matrix": conf_matrix_val}

# save_gradcam_slices
def save_gradcam_slices(original_image_np, cam_map_np, patient_id, pred_class_name_str, true_class_name_str, output_dir, filename_prefix="gradcam"):
    if original_image_np.ndim == 4 and original_image_np.shape[0] == 1: original_image_np = original_image_np.squeeze(0)
    elif original_image_np.ndim == 4 and original_image_np.shape[0] > 1 :
        logger.warning(f"XAI: 원본 이미지가 멀티채널({original_image_np.shape}) 입니다. 첫 번째 채널을 시각화에 사용합니다. (PID: {patient_id})")
        original_image_np = original_image_np[0] # 첫 번째 채널 사용
    if original_image_np.ndim != 3:
        logger.error(f"XAI 시각화 오류: 최종 원본 이미지 차원({original_image_np.shape})이 (D,H,W)가 아님 (PID: {patient_id})."); return

    if cam_map_np.ndim == 4 and cam_map_np.shape[0] == 1: cam_map_np = cam_map_np.squeeze(0)
    if cam_map_np.ndim != 3: logger.error(f"XAI 시각화 오류: CAM 맵 차원({cam_map_np.shape})이 3이 아님 (PID: {patient_id})."); return

    if original_image_np.shape != cam_map_np.shape:
        logger.warning(f"XAI 시각화 (PID: {patient_id}): 원본({original_image_np.shape})과 CAM({cam_map_np.shape}) 차원이 불일치. CAM 리사이징 시도.")
        try:
            cam_map_tensor = torch.tensor(cam_map_np[np.newaxis, np.newaxis, ...]).float()
            resizer = ResizeD(keys=["img"], spatial_size=original_image_np.shape, mode="trilinear", align_corners=True)
            resized_output = resizer({"img": cam_map_tensor})
            cam_map_np_resized = resized_output["img"].squeeze().numpy()

            if cam_map_np_resized.shape == original_image_np.shape: cam_map_np = cam_map_np_resized
            else: logger.error(f"XAI (PID: {patient_id}): CAM 맵 리사이즈 후에도 차원 불일치: {cam_map_np_resized.shape} vs {original_image_np.shape}. 시각화 건너뜁니다."); return
        except Exception as e_resize: logger.error(f"XAI (PID: {patient_id}): CAM 맵 리사이즈 실패: {e_resize}. 시각화 건너뜁니다."); logger.debug(traceback.format_exc()); return

    depth, height, width = original_image_np.shape
    slices_to_show = {"axial": (original_image_np[depth // 2, :, :], cam_map_np[depth // 2, :, :]),
                      "coronal": (original_image_np[:, height // 2, :], cam_map_np[:, height // 2, :]),
                      "sagittal": (original_image_np[:, :, width // 2], cam_map_np[:, :, width // 2])}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_str = f"Grad-CAM: Patient {patient_id} (Predicted: {pred_class_name_str}, True: {true_class_name_str})"
    fig.suptitle(title_str[:150] + ('...' if len(title_str) > 150 else ''), fontsize=16)
    for i, (view_name, (img_slice, cam_slice)) in enumerate(slices_to_show.items()):
        if img_slice.ndim != 2 or cam_slice.ndim != 2: logger.error(f"XAI 시각화 오류 (PID: {patient_id}): {view_name} 뷰 슬라이스가 2D 아님 (Img: {img_slice.shape}, CAM: {cam_slice.shape})."); continue
        axes[i].imshow(np.rot90(img_slice), cmap="gray"); axes[i].imshow(np.rot90(cam_slice), cmap="jet", alpha=0.5)
        axes[i].set_title(f"{view_name.capitalize()} View"); axes[i].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_pred = "".join(c if c.isalnum() else '_' for c in str(pred_class_name_str))[:30]
    safe_true = "".join(c if c.isalnum() else '_' for c in str(true_class_name_str))[:30]
    safe_pid = "".join(c if c.isalnum() else '_' for c in str(patient_id))[:50]
    save_path = os.path.join(output_dir, f"{filename_prefix}_pid_{safe_pid}_pred_{safe_pred}_true_{safe_true}.png")
    try:
        plt.savefig(save_path)
    except Exception as e_save:
        logger.error(f"Grad-CAM 시각화 저장 실패 ({save_path}): {e_save}")
    finally:
        plt.close(fig)


# --- main 실행 부분 ---
if __name__ == "__main__":
    logger.info(f"--- NIfTI 영상 기반 [{MODEL_ARCH}] 3단계(초기/중기/말기) 암 분류 모델 학습 (증강, 동적 헤드) 및 XAI 적용 시작 ---")
    logger.info(f"사용 디바이스: {DEVICE}"); logger.info(f"결과 저장 폴더: {BASE_OUTPUT_DIR}");
    logger.info(f"XAI 결과 저장 폴더: {XAI_OUTPUT_DIR}"); logger.info(f"데이터 로더 워커 수: {NUM_WORKERS_DATALOADER}")
    logger.info(f"선택된 모델 아키텍처: {MODEL_ARCH}, SwinUNETR 초기 특징 크기: {MODEL_FEATURE_SIZE}") # 로그 메시지 명확화
    logger.info(f"사전 훈련 가중치 경로: {PRETRAINED_WEIGHTS_PATH if PRETRAINED_WEIGHTS_PATH and os.path.exists(PRETRAINED_WEIGHTS_PATH) else '사용 안 함 또는 경로 오류'}")
    logger.info(f"최종 분류 클래스 수: {NUM_CLASSES} ({NEW_STAGE_CATEGORIES_MAP})")
    logger.info(f"훈련 데이터 증강 파이프라인 활성화됨.")


    all_data_dicts, final_int_to_str_label_map_for_xai = load_and_prepare_data_from_manifest(
        manifest_file_path=MANIFEST_FILE_PATH,
        patient_id_col=PATIENT_ID_COL,
        string_label_col_in_csv=LABEL_COL,
        original_label_to_new_stage_map=ORIGINAL_STRING_LABEL_TO_NEW_STAGE_STRING_MAP,
        new_stage_map_for_xai=NEW_STAGE_CATEGORIES_MAP
    )

    if not all_data_dicts: logger.error("Manifest에서 데이터를 로드하지 못했습니다. 시스템을 종료합니다."); sys.exit(1)

    loaded_labels_check_initial = [d['label'].item() for d in all_data_dicts]
    logger.info(f"매니페스트 로드 및 초기 매핑 후 레이블 분포: {Counter(loaded_labels_check_initial)}")
    if len(set(loaded_labels_check_initial)) < NUM_CLASSES and loaded_labels_check_initial: # 모든 클래스가 있는지 확인
        logger.warning(f"초기 로드된 데이터의 고유 클래스 수({len(set(loaded_labels_check_initial))})가 NUM_CLASSES({NUM_CLASSES})보다 적습니다! 일부 클래스 데이터가 없을 수 있습니다.")
    elif not loaded_labels_check_initial:
        logger.error("초기 로드된 데이터가 없습니다. 학습을 진행할 수 없습니다."); sys.exit(1)

    logger.info(f"최종 모델 레이블(0,1,2) -> 대분류 문자열 레이블 매핑 (XAI/로깅용): {final_int_to_str_label_map_for_xai}")

    valid_data_dicts_final = []; problematic_items_log = []; transforms_to_test_validity = val_test_transforms
    prog_bar_val_test = tqdm(all_data_dicts, desc="Testing Data Item Transformations", leave=False)
    for i, item_dict in enumerate(prog_bar_val_test):
        try:
            transformed_item = transforms_to_test_validity(item_dict.copy())
            if transformed_item is None or not isinstance(transformed_item, dict) or "image" not in transformed_item :
                raise ValueError("Transformation returned None, invalid dictionary, or missing key 'image'.")

            img_tensor_val = transformed_item.get("image")
            if img_tensor_val is None :
                 pid = item_dict.get(PATIENT_ID_COL, "N/A_val_test")
                 logger.error(f"항목 {i} (PID: {pid}) 변환 후 이미지 텐서가 None입니다. 제외됨.")
                 problematic_items_log.append(f"항목 {i} (PID: {pid}): Transformed image is None.")
                 continue

            if isinstance(img_tensor_val, MetaTensor):
                img_tensor_val = img_tensor_val.as_tensor()

            if torch.isnan(img_tensor_val).any() or torch.isinf(img_tensor_val).any():
                pid = item_dict.get(PATIENT_ID_COL, "N/A_val_test")
                logger.error(f"항목 {i} (PID: {pid}) 변환 후 NaN/Inf 발견. 제외됨.")
                problematic_items_log.append(f"항목 {i} (PID: {pid}): Transformed image contains NaN/Inf.")
                continue
            valid_data_dicts_final.append(item_dict)
        except Exception as e_item_transform:
            pid = item_dict.get(PATIENT_ID_COL, "N/A_val_test"); image_path = item_dict.get("image", "N/A_path_val_test")
            log_msg = f"항목 {i} (PID: {pid}, Path: {image_path}) 변환 중 오류( 제외됨): {type(e_item_transform).__name__} - {str(e_item_transform).splitlines()[0]}"
            logger.warning(log_msg); problematic_items_log.append(log_msg)
            logger.debug(f"Full Traceback for item {i} (PID: {pid}, Path: {image_path}):\n{traceback.format_exc()}")
    prog_bar_val_test.close()

    logger.info(f"데이터 유효성 검사 완료. 이전 필터링된 {len(all_data_dicts)}개 중 {len(valid_data_dicts_final)}개 유효 (NaN/Inf 및 변환 오류 제외).")
    if problematic_items_log:
        logger.warning("--- 문제 발생 항목 요약 (학습에서 제외됨) ---")
        for log_entry in problematic_items_log[:20]: logger.warning(log_entry)
        if len(problematic_items_log) > 20: logger.warning(f"... 외 {len(problematic_items_log) - 20}개의 추가 문제 항목이 있습니다.")

    all_data_dicts = valid_data_dicts_final
    if not all_data_dicts: logger.error("변환 가능한 유효한 데이터가 없어 학습을 진행할 수 없습니다."); sys.exit(1)

    labels_after_validation = [d['label'].item() for d in all_data_dicts]
    if not labels_after_validation : logger.error("유효성 검사 후 모든 데이터가 제외되어 학습할 데이터가 없습니다."); sys.exit(1)

    logger.info(f"데이터 유효성 검사 후 최종 학습/분할용 데이터 레이블 분포: {Counter(labels_after_validation)}")
    unique_final_labels_actually_present = len(set(labels_after_validation))
    if unique_final_labels_actually_present == 0:
        logger.error("모든 클래스의 데이터가 유효성 검사에서 제외되어 학습 불가."); sys.exit(1)
    if unique_final_labels_actually_present < NUM_CLASSES: # 모든 클래스가 있는지 확인
         logger.warning(f"경고: 유효성 검사 후 실제 데이터의 고유 클래스 수({unique_final_labels_actually_present})가 NUM_CLASSES({NUM_CLASSES})보다 적습니다! 일부 클래스가 없을 수 있습니다.")
         if unique_final_labels_actually_present < 2 and (K_FOLDS > 1 or TEST_SPLIT_RATIO > 0 or VAL_SPLIT_RATIO > 0): # 2개 미만이면 계층화 분할 불가
             logger.warning("고유 클래스가 2개 미만이므로 계층화 분할이 불가능할 수 있습니다.")

    labels_for_stratify = [d['label'].item() for d in all_data_dicts]
    if not labels_for_stratify : logger.error("학습/분할에 사용할 데이터가 없습니다."); sys.exit(1)
    if any(l < 0 or l >= NUM_CLASSES for l in labels_for_stratify):
        logger.error(f"오류 FATAL: 최종 레이블이 [0, {NUM_CLASSES-1}] 범위를 벗어납니다! Labels: {Counter(labels_for_stratify)}"); sys.exit("레이블 범위 오류로 중단합니다.")

    can_stratify_overall = False
    if len(set(labels_for_stratify)) >= 2: # 계층화 분할은 최소 2개 클래스 필요
        min_samples_per_class_for_any_split = 2 # train_test_split을 위해 각 클래스당 최소 2개 샘플 필요
        can_stratify_overall = all(c >= min_samples_per_class_for_any_split for c in Counter(labels_for_stratify).values())
        if not can_stratify_overall:
            logger.warning(f"일부 클래스의 샘플 수가 부족하여({Counter(labels_for_stratify)}) 전체 데이터셋에 대한 계층화 분할이 어려울 수 있습니다 (필요 최소 샘플 수: {min_samples_per_class_for_any_split} per class for stratification). 일반 분할을 시도합니다.")
    else:
        logger.warning("전체 데이터셋의 고유 클래스 수가 2개 미만이므로 계층화 분할이 불가능합니다.")

    criterion = None; final_trained_model = None
    if K_FOLDS <= 1:
        logger.info("--- Hold-out 검증 또는 전체 데이터 학습 시작 ---")
        train_indices, val_indices, test_indices = [], [], []; all_indices = list(range(len(all_data_dicts)))

        current_train_val_indices = all_indices
        current_labels_for_stratify_local = labels_for_stratify

        if TEST_SPLIT_RATIO > 0 and len(current_train_val_indices) > 0 :
            num_test_samples = int(np.ceil(len(current_train_val_indices) * TEST_SPLIT_RATIO)) # np.ceil로 올림하여 최소 1개 확보 시도
            if num_test_samples < 1 and len(current_train_val_indices) > 0 :
                 logger.warning(f"테스트셋 비율({TEST_SPLIT_RATIO})이 너무 작아 실제 테스트 샘플을 만들 수 없습니다 ({num_test_samples}개). 테스트셋 없이 진행합니다.")
                 train_val_indices_split = current_train_val_indices
                 test_indices_temp = []
            elif len(current_train_val_indices) <=1 : # 분할할 샘플이 1개 이하일 때
                 logger.warning(f"테스트셋 분할을 위한 전체 샘플 수가 부족합니다({len(current_train_val_indices)}개). 테스트셋 없이 진행합니다.")
                 train_val_indices_split = current_train_val_indices
                 test_indices_temp = []
            else:
                can_stratify_test_split_local = False
                if len(set(current_labels_for_stratify_local)) >= 2 and all(c >= 2 for c in Counter(current_labels_for_stratify_local).values()):
                    can_stratify_test_split_local = True
                try:
                    train_val_indices_split, test_indices_temp = train_test_split(
                        current_train_val_indices,
                        test_size=TEST_SPLIT_RATIO,
                        stratify=current_labels_for_stratify_local if can_stratify_test_split_local else None,
                        random_state=RANDOM_SEED
                    )
                except ValueError as e:
                    logger.warning(f"Test 분할 시 Stratify 오류 ({e}). Stratify 없이 분할.");
                    train_val_indices_split, test_indices_temp = train_test_split(
                        current_train_val_indices,
                        test_size=TEST_SPLIT_RATIO,
                        random_state=RANDOM_SEED
                    )
            test_indices = test_indices_temp
            current_train_val_indices = train_val_indices_split
            current_labels_for_stratify_local = [labels_for_stratify[i] for i in current_train_val_indices] if current_train_val_indices else []
        else:
            if TEST_SPLIT_RATIO > 0 : logger.info("테스트 분할 건너뜀 (분할할 데이터 부족 또는 비율 0).")

        if VAL_SPLIT_RATIO > 0 and len(current_train_val_indices) > 0 :
            denominator_for_relative_val = (1.0 - TEST_SPLIT_RATIO) # 이미 test_split된 후의 비율이므로 1.0
            relative_val_size = 0.0
            if abs(denominator_for_relative_val) > 1e-9 : # 분모가 0이 아닐 때
                relative_val_size = VAL_SPLIT_RATIO / denominator_for_relative_val
            elif VAL_SPLIT_RATIO > 0: # 분모가 0인데 VAL_SPLIT_RATIO가 0보다 크면 val 못 만듬
                 logger.warning("Test split ratio is 1.0 or close to it, cannot calculate relative validation size correctly for non-zero VAL_SPLIT_RATIO.")

            num_val_samples = int(np.ceil(len(current_train_val_indices) * relative_val_size)) if relative_val_size > 0 else 0

            # 학습 데이터가 최소 1개는 남아야 함
            if relative_val_size >= 1.0 or relative_val_size <= 1e-9 or num_val_samples < 1 or (len(current_train_val_indices) - num_val_samples < 1) :
                train_indices = current_train_val_indices; val_indices = []
                if VAL_SPLIT_RATIO > 0 and not (relative_val_size <= 1e-9) : # val_split_ratio가 0이 아닌데 val 못만들면 경고
                    logger.warning("검증셋 생성 불가 (비율 또는 남은 데이터 부족). 남은 데이터를 모두 훈련용으로 사용합니다.")
            elif len(current_train_val_indices) <=1 :
                 logger.warning(f"검증셋 분할을 위한 train_val 샘플 수가 부족합니다({len(current_train_val_indices)}개). 검증셋 없이 진행합니다.")
                 train_indices = current_train_val_indices; val_indices = []
            else:
                can_stratify_val_split_local = False
                if len(set(current_labels_for_stratify_local)) >= 2 and all(c >= 2 for c in Counter(current_labels_for_stratify_local).values()):
                    can_stratify_val_split_local = True
                try:
                    train_indices_final, val_indices_temp = train_test_split(
                        current_train_val_indices,
                        test_size=relative_val_size,
                        stratify=current_labels_for_stratify_local if can_stratify_val_split_local else None,
                        random_state=RANDOM_SEED
                    )
                    train_indices = train_indices_final; val_indices = val_indices_temp
                except ValueError as e:
                    logger.warning(f"Train/Val 분할 시 Stratify 오류 ({e}). Stratify 없이 분할합니다.")
                    train_indices_final, val_indices_temp = train_test_split(
                        current_train_val_indices,
                        test_size=relative_val_size,
                        random_state=RANDOM_SEED
                    )
                    train_indices = train_indices_final; val_indices = val_indices_temp
        else:
            if VAL_SPLIT_RATIO > 0 : logger.info("검증 분할 건너뜀 (분할할 데이터 부족 또는 비율 0).")
            train_indices = current_train_val_indices; val_indices = []

        train_data = [all_data_dicts[i] for i in train_indices] if train_indices else []
        val_data = [all_data_dicts[i] for i in val_indices] if val_indices else []
        test_data = [all_data_dicts[i] for i in test_indices] if test_indices else []

        logger.info(f"데이터 분할: 훈련 {len(train_data)}개, 검증 {len(val_data)}개, 테스트 {len(test_data)}개")
        if train_data: logger.info(f"  훈련 데이터 레이블 분포: {Counter(d['label'].item() for d in train_data)}")
        if val_data: logger.info(f"  검증 데이터 레이블 분포: {Counter(d['label'].item() for d in val_data)}")
        if test_data: logger.info(f"  테스트 데이터 레이블 분포: {Counter(d['label'].item() for d in test_data)}")

        if not train_data: logger.error("훈련 데이터가 없습니다. 종료합니다."); sys.exit(1)

        train_labels_for_weights = [d['label'].item() for d in train_data]
        class_weights_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32).to(DEVICE) 
        if not train_labels_for_weights:
            logger.error("훈련 데이터에 레이블이 없어 가중치 계산 불가. 균일 가중치 사용.")
        else:
            unique_classes_in_train_data, counts_in_train_data = np.unique(train_labels_for_weights, return_counts=True)
            logger.info(f"훈련 데이터 내 최종 레이블 분포 (가중치 계산용): {dict(zip(unique_classes_in_train_data, counts_in_train_data))}")

            if len(unique_classes_in_train_data) > 0 : # 하나라도 클래스가 있어야 가중치 계산
                try:
                    # classes 인자에 모든 가능한 클래스 [0, 1, ..., NUM_CLASSES-1] 전달
                    class_weights_sklearn = compute_class_weight(
                        class_weight='balanced',
                        classes=np.array(range(NUM_CLASSES)),
                        y=train_labels_for_weights
                    )
                    class_weights_tensor = torch.tensor(class_weights_sklearn, dtype=torch.float32).to(DEVICE)
                    if len(class_weights_sklearn) != NUM_CLASSES: # 혹시 모를 상황 대비
                         logger.warning(f"Sklearn compute_class_weight 반환 가중치 수({len(class_weights_sklearn)})가 NUM_CLASSES({NUM_CLASSES})와 다름. 확인 필요.")
                except ValueError as e_cw: # y에 없는 클래스가 classes에만 있을 때 발생하지 않음. y의 모든 값이 classes에 있어야 함.
                    logger.error(f"클래스 가중치 계산 중 오류: {e_cw}. 균일 가중치 사용.")
            else:
                logger.error("훈련 데이터에 유효한 레이블이 전혀 없어 가중치 계산 불가. 균일 가중치 사용.")

        logger.info(f"적용될 클래스 가중치 (0,1,2 순서): {class_weights_tensor.cpu().numpy().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        logger.info(f"손실 함수: CrossEntropyLoss (가중치 적용됨), 모델 학습을 위한 최종 NUM_CLASSES: {NUM_CLASSES}")

        train_dataset = ErrorTolerantDataset(data=train_data, transform=train_transforms, patient_id_col_name=PATIENT_ID_COL)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)

        val_loader = None
        if val_data:
            val_dataset = ErrorTolerantDataset(data=val_data, transform=val_test_transforms, patient_id_col_name=PATIENT_ID_COL)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)

        test_loader = None
        if test_data:
            test_dataset = ErrorTolerantDataset(data=test_data, transform=val_test_transforms, patient_id_col_name=PATIENT_ID_COL)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=safe_list_data_collate)

        # CTClassifier 생성 시 img_size 인자 없이 호출
        model = CTClassifier(num_classes_final=NUM_CLASSES,
                             model_arch=MODEL_ARCH,
                             in_channels=1, # CTClassifier 생성자에 맞게 수정
                             feature_size=MODEL_FEATURE_SIZE, # SwinUNETR의 초기 feature_size 전달
                             pretrained_weights_path=PRETRAINED_WEIGHTS_PATH,
                             freeze_feature_extractor=True).to(DEVICE)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

        best_val_metric_holdout = -1.0; best_model_state_holdout = None; best_epoch_holdout = -1

        logger.info(f"총 {NUM_EPOCHS} 에포크 학습 시작. 처음 {FREEZE_FEATURE_EXTRACTOR_EPOCHS} 에포크 동안 특징 추출기 동결.")

        for epoch in range(NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")

            if epoch == FREEZE_FEATURE_EXTRACTOR_EPOCHS and (PRETRAINED_WEIGHTS_PATH and os.path.exists(PRETRAINED_WEIGHTS_PATH)):
                is_frozen = all(not param.requires_grad for param in model.feature_extractor.parameters())
                if not is_frozen:
                     logger.info("특징 추출기가 이미 동결 해제 상태이거나, 동결된 적이 없습니다. 옵티마이저 재설정만 진행.")
                else:
                    logger.info("특징 추출기 동결 해제 및 옵티마이저 재설정 (학습률 조정).")
                    model.unfreeze_feature_extractor()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)

            train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, current_num_classes=NUM_CLASSES, patient_id_col_key=PATIENT_ID_COL)
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

            current_epoch_val_metric_for_best_model = train_auc

            if val_loader:
                val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, phase="Validation", current_num_classes=NUM_CLASSES, patient_id_col_key=PATIENT_ID_COL)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")
                logger.debug(f"Val Confusion Matrix:\n{val_metrics['confusion_matrix']}")
                current_epoch_val_metric_for_best_model = val_metrics['auc']
            else:
                logger.info("검증 데이터로더(val_loader)가 없습니다. 훈련 AUC를 기준으로 최적 모델을 저장합니다.")

            if current_epoch_val_metric_for_best_model >= best_val_metric_holdout:
                best_val_metric_holdout = current_epoch_val_metric_for_best_model
                best_model_state_holdout = copy.deepcopy(model.state_dict())
                best_epoch_holdout = epoch + 1
                logger.info(f"새로운 최적 {'검증 AUC' if val_loader else '훈련 AUC'}: {best_val_metric_holdout:.4f} (Epoch {best_epoch_holdout})")

        if best_model_state_holdout:
            final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME);
            torch.save(best_model_state_holdout, final_model_path);
            logger.info(f"최종 최적 모델(Hold-out/Full Train) 저장 완료: {final_model_path} (Epoch {best_epoch_holdout}에서 달성된 AUC: {best_val_metric_holdout:.4f})")
            # 모델 로드 시에도 CTClassifier 생성자에서 img_size 인자 없이 호출
            final_trained_model = CTClassifier(num_classes_final=NUM_CLASSES,
                                               model_arch=MODEL_ARCH,
                                               in_channels=1,
                                               feature_size=MODEL_FEATURE_SIZE,
                                               pretrained_weights_path=None, # 이미 학습된 가중치를 로드하므로 None
                                               freeze_feature_extractor=False).to(DEVICE)
            final_trained_model.load_state_dict(best_model_state_holdout)
        elif model:
            final_trained_model = model
            logger.warning("저장된 최적 모델 상태(best_model_state_holdout)가 없습니다. 마지막 에폭의 모델을 사용하고 저장합니다.")
            final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME.replace(".pth", "_last_epoch.pth"));
            torch.save(model.state_dict(), final_model_path);
            logger.info(f"마지막 에폭 모델 저장 완료: {final_model_path}")
        else:
            logger.error("훈련된 모델을 찾을 수 없습니다."); final_trained_model = None

    # 테스트 및 XAI
    if final_trained_model and test_loader:
        logger.info("\n--- 테스트셋 최종 평가 ---"); final_trained_model.eval()
        if criterion is None : criterion = nn.CrossEntropyLoss()

        test_metrics = evaluate_model(final_trained_model, test_loader, criterion, DEVICE, phase="Test", current_num_classes=NUM_CLASSES, patient_id_col_key=PATIENT_ID_COL)
        logger.info(f"Test Results - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}");
        cm_to_print = test_metrics.get('confusion_matrix', np.array([]))
        if not isinstance(cm_to_print, np.ndarray): cm_to_print = np.array(cm_to_print)
        logger.info(f"Test Confusion Matrix (rows: true, cols: pred):\n{cm_to_print}")
        try:
            test_labels_dist = Counter([d['label'].item() for d in test_data]) if test_data else Counter()
            test_metrics_to_save = test_metrics.copy()
            test_metrics_to_save['confusion_matrix'] = str(test_metrics_to_save['confusion_matrix'].tolist())
            test_metrics_to_save['test_label_distribution'] = str(dict(test_labels_dist))

            test_results_df = pd.DataFrame([test_metrics_to_save]);
            test_results_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "final_test_metrics.csv"), index=False)
            logger.info(f"최종 테스트 결과 저장 완료: {os.path.join(BASE_OUTPUT_DIR, 'final_test_metrics.csv')}")
        except Exception as e_save_metric: logger.error(f"최종 테스트 결과 저장 중 오류: {e_save_metric}")

        if XAI_NUM_SAMPLES_TO_VISUALIZE > 0 and test_data:
            logger.info("\n--- XAI (Grad-CAM) 결과 생성 시작 ---")
            grad_cam_target_layer_full_name_for_monai = f"feature_extractor.{XAI_TARGET_LAYER_NAME}"

            logger.info(f"Grad-CAM 대상 레이어로 시도할 전체 경로: {grad_cam_target_layer_full_name_for_monai}")
            logger.warning(f"XAI_TARGET_LAYER_NAME ('{XAI_TARGET_LAYER_NAME}')이(가) 실제 SwinViT 모델 구조 (model.feature_extractor 내부)와 일치하는지 확인이 매우 중요합니다. "
                           "정확한 레이어 이름은 `model.feature_extractor.named_modules()` 또는 `model.named_modules()`로 확인하고 설정하세요.")

            try:
                logger.info(f"Grad-CAM 초기화 시도: nn_module=final_trained_model, target_layers='{grad_cam_target_layer_full_name_for_monai}'")
                grad_cam_obj = GradCAM(nn_module=final_trained_model, target_layers=grad_cam_target_layer_full_name_for_monai)
                logger.info("Grad-CAM 객체 생성 완료.")

                visualized_count = 0; samples_processed_for_xai = 0
                for xai_batch_idx, xai_batch_data in tqdm(enumerate(test_loader), desc="Generating Grad-CAM from TestLoader", total=len(test_loader), leave=False):
                    if visualized_count >= XAI_NUM_SAMPLES_TO_VISUALIZE: break
                    if xai_batch_data is None: continue

                    inputs_for_xai = xai_batch_data["image"].to(DEVICE)
                    labels_for_xai_cpu = xai_batch_data["label"]
                    pids_in_batch = xai_batch_data.get(PATIENT_ID_COL, [])

                    for i in range(inputs_for_xai.size(0)):
                        if visualized_count >= XAI_NUM_SAMPLES_TO_VISUALIZE: break

                        single_input_gpu = inputs_for_xai[i].unsqueeze(0)
                        original_image_np_for_vis_ch_first = single_input_gpu.squeeze(0).cpu().numpy()

                        original_image_np_for_vis = None
                        if original_image_np_for_vis_ch_first.ndim == 3:
                            original_image_np_for_vis = original_image_np_for_vis_ch_first
                        elif original_image_np_for_vis_ch_first.ndim == 4 and original_image_np_for_vis_ch_first.shape[0] == 1:
                            original_image_np_for_vis = original_image_np_for_vis_ch_first.squeeze(0)
                        elif original_image_np_for_vis_ch_first.ndim == 4 and original_image_np_for_vis_ch_first.shape[0] > 1:
                            logger.warning(f"XAI: 멀티채널({original_image_np_for_vis_ch_first.shape}) 이미지 시각화 시 첫 번째 채널 사용 (PID: {pids_in_batch[i] if i < len(pids_in_batch) else 'N/A'}).")
                            original_image_np_for_vis = original_image_np_for_vis_ch_first[0]
                        else:
                            logger.error(f"XAI: 처리할 수 없는 이미지 차원 {original_image_np_for_vis_ch_first.shape} (PID: {pids_in_batch[i] if i < len(pids_in_batch) else 'N/A'}). 시각화 건너뜁니다.")
                            continue

                        final_trained_model.eval()
                        logits = final_trained_model(single_input_gpu)
                        predicted_class_idx_xai = torch.argmax(logits, dim=1).item()

                        cam_map_tensor = grad_cam_obj(x=single_input_gpu, class_idx=predicted_class_idx_xai)

                        if cam_map_tensor is None:
                            logger.warning(f"Grad-CAM 생성 실패 for PID: {pids_in_batch[i] if i < len(pids_in_batch) else 'N/A'}, class_idx: {predicted_class_idx_xai}. 건너뜁니다.")
                            continue

                        cam_map_np_for_vis = cam_map_tensor.squeeze().cpu().detach().numpy()

                        true_label_idx_xai = labels_for_xai_cpu[i].item()

                        predicted_class_name_str = final_int_to_str_label_map_for_xai.get(predicted_class_idx_xai, f"PredClass_{predicted_class_idx_xai}")
                        true_class_name_str = final_int_to_str_label_map_for_xai.get(true_label_idx_xai, f"TrueClass_{true_label_idx_xai}")
                        patient_id_for_xai_item = str(pids_in_batch[i] if isinstance(pids_in_batch, list) and i < len(pids_in_batch) else f"UnknownPID_B{xai_batch_idx}I{i}")

                        save_gradcam_slices(original_image_np_for_vis, cam_map_np_for_vis, patient_id_for_xai_item,
                                            predicted_class_name_str, true_class_name_str, XAI_OUTPUT_DIR,
                                            filename_prefix=f"xai_sample_b{xai_batch_idx}_i{i}")
                        visualized_count += 1
                    samples_processed_for_xai += inputs_for_xai.size(0)
                logger.info(f"XAI (Grad-CAM) 시각화 {visualized_count}개 완료 (총 {samples_processed_for_xai}개 테스트 샘플 처리 시도). 저장 폴더: {XAI_OUTPUT_DIR}")

            except Exception as e_xai_main:
                logger.error(f"XAI (Grad-CAM) 처리 중 주요 오류 발생: {type(e_xai_main).__name__} - {e_xai_main}")
                logger.error(f"XAI 대상 레이어 경로 '{grad_cam_target_layer_full_name_for_monai}' 또는 Grad-CAM 모듈 자체에 문제가 있을 수 있습니다. "
                               "모델 구조와 레이어 이름을 다시 확인하십시오. MONAI 버전이 오래되어 특정 기능(예: 레이어 경로 해석)을 지원하지 않을 수도 있습니다.")
                logger.debug(traceback.format_exc())

        elif not test_data: logger.info("테스트 데이터가 없어 XAI 시각화를 건너뜁니다.")
        elif XAI_NUM_SAMPLES_TO_VISUALIZE == 0: logger.info("XAI_NUM_SAMPLES_TO_VISUALIZE=0. XAI 시각화 건너뜁니다.")
    elif not final_trained_model: logger.error("훈련된 최종 모델이 없어 테스트 및 XAI를 진행할 수 없습니다.")
    elif not test_loader: logger.warning("테스트 로더가 없어 테스트 및 XAI를 진행할 수 없습니다 (예: 테스트 데이터 없음).")

    logger.info("--- 모든 과정 완료 ---")