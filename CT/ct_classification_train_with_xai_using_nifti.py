# ct_classification_train_with_xai_using_nifti.py
# 전처리된 NIfTI 영상 기반 암 분류 모델 학습 및 XAI 적용

# --- OpenMP 중복 라이브러리 로드 허용 ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import glob # 사용 빈도 낮아져 제거 가능성 있음
import numpy as np
import pandas as pd
# import SimpleITK as sitk # NIfTI 사용으로 직접적 필요성 감소
import monai
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, # OrientationD, SpacingD는 전처리에서 수행
    ScaleIntensityRangePercentilesD, ResizeD, Compose,
    EnsureTypeD, RandFlipd, RandRotate90d
)
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.utils import set_determinism
from monai.visualize import GradCAM
from monai.networks.nets import resnet34

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from tqdm import tqdm
import logging
import traceback
import copy
import matplotlib.pyplot as plt

# --- 설정값 ---
# 입력 데이터 경로 (Manifest 파일 기준)
MANIFEST_FILE_PATH = os.path.join(os.getcwd(), "preprocessed_nifti_data", "preprocessed_manifest.csv") # 전처리 스크립트가 생성한 manifest 파일
# CLINICAL_DATA_FILE은 manifest에 정보가 통합되었으므로 직접 사용 안 함
PATIENT_ID_COL = 'bcr_patient_barcode' # Manifest 파일 내 환자 ID 컬럼명 (일치해야 함)
LABEL_COL = 'ajcc_pathologic_stage' # Manifest 파일 내 원본 레이블 컬럼명 (일치해야 함)
# NIFTI_IMAGE_PATH_COL은 manifest 파일 내 NIfTI 경로를 담은 컬럼명으로, load_and_prepare_data_from_manifest 내부에서 처리

# 출력 설정
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "classification_results_nifti_with_xai_v3") # 결과 저장 폴더명 변경
MODEL_CHECKPOINT_NAME = "best_ct_classification_nifti_model.pth"
XAI_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "xai_gradcam_outputs_nifti")
LOG_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "ct_classification_nifti_log.txt")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(XAI_OUTPUT_DIR, exist_ok=True)

# 전처리 및 모델 관련 설정 (PIXDIM은 전처리에서 사용, 여기서는 Resize만 중요)
RESIZE_SHAPE = (96, 96, 96) # 모델 입력 크기
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
set_determinism(RANDOM_SEED)

PRETRAINED_WEIGHTS_PATH = "G:/내 드라이브/2조/본프로젝트/resnet_34_23dataset.pth" # 실제 경로

# 학습 파라미터
NUM_CLASSES = 5 # get_label_mapping_from_manifest 에서 실제 값으로 업데이트됨
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # GPU 메모리 상황에 따라 조절
NUM_EPOCHS = 50
K_FOLDS = 0 # 0 또는 1 이면 Hold-out, 2 이상이면 K-Fold CV
TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.15 # (1-TEST_SPLIT_RATIO)에 대한 비율로 내부에서 재계산됨
FREEZE_FEATURE_EXTRACTOR_EPOCHS = 5
NUM_WORKERS_DATALOADER = 4 # 데이터 로더 워커 수 (시스템 환경에 맞게 조절)

# XAI 설정
XAI_NUM_SAMPLES_TO_VISUALIZE = 5
XAI_TARGET_LAYER_NAME = "layer4" # ResNet34 기준

# --- 로거 설정 ---
logger = logging.getLogger("train_nifti") # 로거 이름 변경
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s'))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s'))
logger.addHandler(stream_handler)


# --- 데이터 로드 및 전처리 함수 (Manifest 기반) ---
def get_label_mapping_from_manifest(df, label_col_name): # LABEL_COL 사용
    unique_labels = sorted(df[label_col_name].astype(str).unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    logger.info(f"Manifest 기반 레이블 매핑: {label_to_int}")
    
    global NUM_CLASSES # 전역 변수 NUM_CLASSES를 업데이트
    actual_num_classes = len(unique_labels)
    if actual_num_classes != NUM_CLASSES:
        logger.warning(f"설정된 NUM_CLASSES({NUM_CLASSES})와 실제 레이블 종류 수({actual_num_classes})가 다릅니다. NUM_CLASSES를 {actual_num_classes}로 업데이트합니다.")
        NUM_CLASSES = actual_num_classes
    return label_to_int, int_to_label

def load_and_prepare_data_from_manifest(manifest_file_path, patient_id_col_name, original_label_col_name):
    try:
        manifest_df = pd.read_csv(manifest_file_path)
        logger.info(f"Manifest 파일 로드 완료: {manifest_file_path}, 총 {len(manifest_df)}개 항목.")
    except FileNotFoundError:
        logger.error(f"Manifest 파일({manifest_file_path})을 찾을 수 없습니다.")
        return [], None, None

    # Manifest 파일에 필요한 컬럼 확인 (예: 'image_nifti_path', PATIENT_ID_COL, LABEL_COL, 'label_encoded')
    # 'image_nifti_path'와 'label_encoded'는 전처리 스크립트에서 생성한 컬럼명과 일치해야 함
    required_cols = ['image_nifti_path', patient_id_col_name, original_label_col_name, 'label_encoded']
    for col in required_cols:
        if col not in manifest_df.columns:
            logger.error(f"Manifest 파일에 필요한 컬럼 '{col}'이 없습니다.")
            return [], None, None
    
    # 이미 인코딩된 레이블 사용, 레이블 매핑은 int_to_label 생성용으로만
    # 원본 레이블 컬럼(LABEL_COL)을 기준으로 매핑 정보 생성 (XAI 시각화 등에 사용될 수 있음)
    label_to_int_map_ref, int_to_label_map = get_label_mapping_from_manifest(manifest_df, original_label_col_name)

    all_data_dicts = []
    for _, row in manifest_df.iterrows():
        # 파일 존재 여부 확인 (선택적이지만 권장)
        if not os.path.exists(row['image_nifti_path']):
            logger.warning(f"NIfTI 파일 경로를 찾을 수 없습니다: {row['image_nifti_path']}. 이 항목은 건너뜁니다.")
            continue

        data_dict = {
            "image": row['image_nifti_path'], # MONAI LoadImageD가 사용할 키
            "label": torch.tensor(row['label_encoded'], dtype=torch.long), # 이미 인코딩된 레이블 사용
            patient_id_col_name: row[patient_id_col_name],
            "original_label": row[original_label_col_name] # 원본 레이블 (XAI 등에 사용)
            # 필요시 study_id, series_id 등 manifest의 다른 정보도 추가 가능
        }
        all_data_dicts.append(data_dict)
            
    if not all_data_dicts:
        logger.error("Manifest에서 유효한 데이터를 로드하지 못했습니다. 파일 경로 및 내용을 확인하세요.")

    # int_to_label_map은 get_label_mapping_from_manifest에서 업데이트된 NUM_CLASSES 기준으로 생성됨
    return all_data_dicts, label_to_int_map_ref, int_to_label_map


# --- MONAI Transforms 정의 (NIfTI용) ---
# OrientationD, SpacingD는 전처리에서 이미 수행되었다고 가정
train_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0), # D
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=1), # H
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=2), # W
    RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)), # HW 평면 (Axial 기준 Coronal, Sagittal slice) 회전
    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True), # align_corners 주의
    EnsureTypeD(keys=["image"], dtype=torch.float32),
])
val_test_transforms = Compose([
    LoadImageD(keys=["image"], reader="NibabelReader", image_only=True, ensure_channel_first=True),
    ScaleIntensityRangePercentilesD(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ResizeD(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear", align_corners=True),
    EnsureTypeD(keys=["image"], dtype=torch.float32),
])

# --- 모델 정의 (CTClassifier - 이전과 동일) ---
class CTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_arch=resnet34,
                 pretrained_weights_path=None, freeze_feature_extractor=True):
        super().__init__()
        logger.info(f"모델 초기화: {pretrained_model_arch.__name__} 사용, 클래스 수: {num_classes}")
        # MedicalNet ResNet34는 3D 입력을 받음. n_input_channels=1 (흑백 CT)
        self.feature_extractor = pretrained_model_arch(n_input_channels=1, num_classes=1000) # MedicalNet ResNet은 num_classes로 초기화 (이후 fc 대체)

        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                state_dict = torch.load(pretrained_weights_path, map_location=DEVICE, weights_only=True) # weights_only=True 권장
                if 'state_dict' in state_dict: # MedicalNet 가중치 파일 구조 대응
                    state_dict = state_dict['state_dict']
                
                # 'module.' 접두사 제거 (DataParallel 등으로 학습된 모델 로드 시)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # MedicalNet ResNet은 conv1, bn1, layer1..layer4, fc 구조
                # 사전 훈련된 가중치의 fc는 무시하고 특징 추출기 부분만 로드
                # strict=False로 하여 fc 레이어 불일치 오류 방지
                missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(new_state_dict, strict=False)
                logger.info(f"모델({pretrained_model_arch.__name__})에 사전 훈련된 가중치 로드 완료: {pretrained_weights_path}")
                if missing_keys: logger.warning(f"누락된 키: {missing_keys}")
                if unexpected_keys: logger.warning(f"예상치 못한 키: {unexpected_keys} (fc 레이어 등일 수 있음)")

            except Exception as e:
                logger.error(f"사전 훈련된 가중치 로드 실패: {e}. 무작위 초기화된 모델로 진행합니다.")
                logger.error(traceback.format_exc())
        elif pretrained_weights_path:
            logger.warning(f"사전 훈련된 가중치 파일을 찾을 수 없습니다: {pretrained_weights_path}. 무작위 초기화된 모델로 진행합니다.")
        else:
            logger.warning("사전 훈련된 가중치 경로가 제공되지 않았습니다. 무작위 초기화된 모델로 진행합니다. 성능에 영향이 있을 수 있습니다.")

        if freeze_feature_extractor:
            for param_name, param in self.feature_extractor.named_parameters():
                if "fc" not in param_name: # 마지막 fc 레이어를 제외하고 동결
                    param.requires_grad = False
            logger.info("특징 추출기 동결됨 (마지막 fc 레이어 제외).")
        
        try:
            # ResNet의 마지막 fully connected layer (fc)의 입력 특징 수 가져오기
            original_num_ftrs = self.feature_extractor.fc.in_features
            # 기존 fc 레이어를 Identity로 대체하여 특징 벡터만 추출
            self.feature_extractor.fc = nn.Identity()
            num_ftrs_for_head = original_num_ftrs
        except AttributeError: # 'fc' 레이어가 없는 모델 아키텍처의 경우 (예: 직접 특징 추출기 사용)
            logger.warning(f"{pretrained_model_arch.__name__}에 'fc' 속성이 없습니다. 특징 추출기의 출력을 직접 사용한다고 가정합니다.")
            # 이 경우, num_ftrs_for_head를 모델 아키텍처에 맞게 수동으로 설정해야 할 수 있습니다.
            # 예시: ResNet34의 경우 layer4 출력 후 AdaptiveAvgPool3d를 거치면 512가 됨.
            # MedicalNet ResNet은 아키텍처에 따라 다를 수 있으므로 확인 필요.
            # 여기서는 feature_extractor가 (batch, num_features)를 반환한다고 가정.
            # ResNet34의 경우 보통 512.
            num_ftrs_for_head = 512 if pretrained_model_arch == resnet34 else 2048 # 모델에 맞게 조정 필요!
            # MedicalNet ResNet34의 경우 `get_inplanes()` 또는 아키텍처 확인 필요. 보통 512.
            logger.warning(f"분류기 헤드 입력 특징 수를 {num_ftrs_for_head}로 가정합니다. 모델 아키텍처를 확인하세요.")


        # 새로운 분류기 헤드 정의
        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs_for_head, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        logger.info(f"새로운 분류기 헤드 구성: Linear({num_ftrs_for_head}, 256) -> ... -> Linear(256, {num_classes})")

    def forward(self, x):
        features = self.feature_extractor(x) # 특징 추출
        output = self.classifier_head(features) # 분류기 헤드를 통과
        return output

    def unfreeze_feature_extractor(self):
        for param_name, param in self.feature_extractor.named_parameters():
            if "fc" not in param_name: # 마지막 fc 레이어는 계속 학습 가능하게 둠 (Identity 이므로 영향 없음)
                param.requires_grad = True
        logger.info("특징 추출기 전체 동결 해제됨 (마지막 fc 레이어 제외).")


# --- 학습 및 평가 함수 (train_one_epoch, evaluate_model - 이전과 동일) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels_epoch = []
    all_preds_proba_epoch = []
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training")):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted_classes = torch.max(outputs, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)
        all_labels_epoch.extend(labels.cpu().numpy())
        all_preds_proba_epoch.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
    epoch_loss = running_loss / total_predictions
    epoch_acc = correct_predictions / total_predictions
    try:
        if NUM_CLASSES > 2 and len(np.unique(all_labels_epoch)) > 1:
             epoch_auc = roc_auc_score(all_labels_epoch, all_preds_proba_epoch, multi_class='ovr', average='macro')
        elif NUM_CLASSES == 2 and len(np.unique(all_labels_epoch)) == 2: # 이진 분류
            epoch_auc = roc_auc_score(all_labels_epoch, np.array(all_preds_proba_epoch)[:, 1])
        else: # 단일 클래스만 존재하거나 하는 경우
            epoch_auc = 0.0
            if len(np.unique(all_labels_epoch)) <= 1 : logger.debug(f"훈련 중 AUC 계산 불가: 레이블 종류 부족 ({len(np.unique(all_labels_epoch))}개)")
    except ValueError as e:
        logger.warning(f"훈련 중 AUC 계산 오류: {e}. AUC는 0.0으로 설정됩니다.")
        epoch_auc = 0.0
    return epoch_loss, epoch_acc, epoch_auc

def evaluate_model(model, dataloader, criterion, device, phase="Validation"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels_eval = []
    all_predicted_classes_eval = []
    all_preds_proba_eval = [] # AUC 계산용
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=phase):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            all_labels_eval.extend(labels.cpu().numpy())
            all_predicted_classes_eval.extend(predicted_classes.cpu().numpy())
            all_preds_proba_eval.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    eval_loss = running_loss / total_predictions
    eval_acc = correct_predictions / total_predictions
    
    # 추가 평가지표
    f1 = f1_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0)
    precision = precision_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0)
    recall = recall_score(all_labels_eval, all_predicted_classes_eval, average='macro', zero_division=0)
    
    try:
        if NUM_CLASSES > 2 and len(np.unique(all_labels_eval)) > 1 : # 다중 클래스 AUC
            auc_score_val = roc_auc_score(all_labels_eval, all_preds_proba_eval, multi_class='ovr', average='macro')
        elif NUM_CLASSES == 2 and len(np.unique(all_labels_eval)) == 2: # 이진 분류 AUC
            auc_score_val = roc_auc_score(all_labels_eval, np.array(all_preds_proba_eval)[:, 1])
        else: # 단일 클래스만 존재하거나 하는 경우
            auc_score_val = 0.0
            if len(np.unique(all_labels_eval)) <=1 : logger.debug(f"{phase} 중 AUC 계산 불가: 레이블 종류 부족 ({len(np.unique(all_labels_eval))}개)")
    except ValueError as e:
        logger.warning(f"{phase} 중 AUC 계산 오류: {e}. AUC는 0.0으로 설정됩니다.")
        auc_score_val = 0.0

    # Confusion matrix는 모든 클래스 레이블을 포함하도록 수정
    conf_matrix_val = confusion_matrix(all_labels_eval, all_predicted_classes_eval, labels=list(range(NUM_CLASSES)))
    
    return {
        "loss": eval_loss, "accuracy": eval_acc, "f1_score": f1, 
        "precision": precision, "recall": recall, "auc": auc_score_val,
        "confusion_matrix": conf_matrix_val
    }

# --- XAI: Grad-CAM 시각화 함수 (이전과 동일, 단 입력 이미지는 NIfTI 로드 후 변환된 것) ---
def save_gradcam_slices(original_image_np, cam_map_np, patient_id, pred_class_name, true_class_name, output_dir, filename_prefix="gradcam"):
    # original_image_np는 (C, D, H, W) 또는 (D, H, W) 가정, 시각화를 위해 채널 차원 제거
    if original_image_np.ndim == 4 and original_image_np.shape[0] == 1: # (1, D, H, W)
        original_image_np = original_image_np.squeeze(0) # (D, H, W)
    elif original_image_np.ndim != 3: # (D,H,W)가 아니면 오류
        logger.error(f"XAI 시각화 오류: 원본 이미지 차원({original_image_np.shape})이 (D,H,W)가 아님.")
        return

    # cam_map_np도 (D, H, W)로 가정 (GradCAM 출력은 보통 (D,H,W))
    if cam_map_np.ndim == 4 and cam_map_np.shape[0] == 1: # (1, D, H, W) 형태일 수 있음
         cam_map_np = cam_map_np.squeeze(0)
    
    if original_image_np.shape != cam_map_np.shape:
        logger.warning(f"XAI 시각화: 원본({original_image_np.shape})과 CAM({cam_map_np.shape}) 차원이 불일치합니다. CAM 맵을 원본 크기로 리사이징 시도.")
        try:
            # CAM 맵을 원본 이미지 크기로 리사이징 (MONAI Resize 사용)
            cam_map_tensor = torch.tensor(cam_map_np[np.newaxis, np.newaxis, ...]).float() # (1, 1, D, H, W)
            resizer = ResizeD(keys=["img"], spatial_size=original_image_np.shape, mode="trilinear", align_corners=True) # align_corners 주의
            cam_map_resized_dict = resizer({"img": cam_map_tensor})
            cam_map_np = cam_map_resized_dict["img"].squeeze().numpy()
            logger.info(f"CAM 맵 리사이즈 완료: {cam_map_np.shape}")
        except Exception as e_resize:
            logger.error(f"CAM 맵 리사이즈 실패: {e_resize}. 시각화를 건너뜁니다.")
            return
            
    depth, height, width = original_image_np.shape # (D, H, W)

    # 중앙 슬라이스 선택 (D, H, W 순서 기준)
    slices_to_show = {
        "axial": (original_image_np[depth // 2, :, :], cam_map_np[depth // 2, :, :]),       # 가운데 D 슬라이스 (H, W)
        "coronal": (original_image_np[:, height // 2, :], cam_map_np[:, height // 2, :]),   # 가운데 H 슬라이스 (D, W)
        "sagittal": (original_image_np[:, :, width // 2], cam_map_np[:, :, width // 2]) # 가운데 W 슬라이스 (D, H)
    }
    # RAS 방향 기준 Axial: superior-inferior (D), Coronal: anterior-posterior (H), Sagittal: left-right (W)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1x3 subplot
    fig.suptitle(f"Grad-CAM: Patient {patient_id} (Predicted: {pred_class_name}, True: {true_class_name})", fontsize=16)

    for i, (view_name, (img_slice, cam_slice)) in enumerate(slices_to_show.items()):
        # img_slice와 cam_slice가 2D인지 확인
        if img_slice.ndim != 2 or cam_slice.ndim != 2:
            logger.error(f"XAI 시각화 오류: {view_name} 뷰의 슬라이스가 2D가 아님 (Img: {img_slice.shape}, CAM: {cam_slice.shape}).")
            continue
        axes[i].imshow(np.rot90(img_slice), cmap="gray") # 의료 영상 표기 관례상 rot90 적용 가능성 있음 (필요시 조정)
        axes[i].imshow(np.rot90(cam_slice), cmap="jet", alpha=0.5)
        axes[i].set_title(f"{view_name.capitalize()} View")
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # suptitle과의 간격 조정
    
    # 파일명에 포함될 수 없는 문자 처리
    safe_pred_class_name = "".join(c if c.isalnum() else '_' for c in str(pred_class_name))
    safe_true_class_name = "".join(c if c.isalnum() else '_' for c in str(true_class_name))
    safe_patient_id = "".join(c if c.isalnum() else '_' for c in str(patient_id))

    save_path = os.path.join(output_dir, f"{filename_prefix}_pid_{safe_patient_id}_pred_{safe_pred_class_name}_true_{safe_true_class_name}.png")
    try:
        plt.savefig(save_path)
        logger.info(f"Grad-CAM 시각화 저장 완료: {save_path}")
    except Exception as e_save:
        logger.error(f"Grad-CAM 시각화 저장 실패 ({save_path}): {e_save}")
    finally:
        plt.close(fig)


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    logger.info("--- NIfTI 영상 기반 암 분류 모델 학습 및 XAI 적용 시작 ---")
    logger.info(f"사용 디바이스: {DEVICE}")
    logger.info(f"결과 저장 폴더: {BASE_OUTPUT_DIR}")
    logger.info(f"XAI 결과 저장 폴더: {XAI_OUTPUT_DIR}")
    logger.info(f"데이터 로더 워커 수: {NUM_WORKERS_DATALOADER}")
    
    # 1. Manifest 파일로부터 데이터 정보 로드
    all_data_dicts, label_to_int_map_ref, int_to_label_map = load_and_prepare_data_from_manifest(
        MANIFEST_FILE_PATH, PATIENT_ID_COL, LABEL_COL
    )
    if not all_data_dicts:
        logger.error("데이터 준비 실패. Manifest 파일을 확인하거나 전처리 스크립트를 먼저 실행하세요. 시스템을 종료합니다.")
        sys.exit()
    
    logger.info(f"실제 클래스 수 (업데이트됨): {NUM_CLASSES}, 원본 레이블 컬럼: {LABEL_COL}")
    logger.info(f"참고용 레이블 인코딩 맵 (manifest 기준): {label_to_int_map_ref}") # 전처리시 사용된 매핑
    logger.info(f"역 레이블 인코딩 맵 (현재 데이터 기준): {int_to_label_map}") # 학습 및 XAI에 사용될 매핑

    # Stratified K-Fold 또는 Hold-out을 위한 레이블 목록
    labels_for_stratify = [d['label'].item() for d in all_data_dicts] # 이미 인코딩된 레이블 사용

    if len(np.unique(labels_for_stratify)) < 2 and (K_FOLDS > 1 or TEST_SPLIT_RATIO > 0):
        logger.error(f"데이터 분할에 필요한 최소 레이블 종류(2개)를 만족하지 못합니다 (현재: {len(np.unique(labels_for_stratify))}개). K_FOLDS와 TEST_SPLIT_RATIO를 확인하거나 데이터를 점검하세요.")
        if len(all_data_dicts) > 0 : # 데이터가 있는데 레이블 종류가 1개인 경우
             logger.warning("모든 데이터가 동일한 레이블을 가집니다. 학습이 의미 없을 수 있습니다.")
        # sys.exit() # 필요시 종료
    elif not all_data_dicts:
        logger.error("데이터가 없습니다. 종료합니다.")
        sys.exit()

    criterion = nn.CrossEntropyLoss()
    logger.info(f"손실 함수: CrossEntropyLoss")
    final_trained_model = None # 최종적으로 선택/학습된 모델

    # --- K-Fold Cross Validation 또는 Hold-out ---
    if K_FOLDS > 1 and len(np.unique(labels_for_stratify)) >= K_FOLDS : # KFold는 최소 K개의 클래스 샘플이 필요하진 않지만, 각 fold에 모든 클래스가 있도록 하려면 충분해야함.
        logger.info(f"--- {K_FOLDS}-Fold 교차 검증 시작 ---")
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_data_dicts)), labels_for_stratify)):
            logger.info(f"--- Fold {fold + 1}/{K_FOLDS} ---")
            train_data_fold = [all_data_dicts[i] for i in train_idx]
            val_data_fold = [all_data_dicts[i] for i in val_idx]

            train_dataset_fold = Dataset(data=train_data_fold, transform=train_transforms)
            val_dataset_fold = Dataset(data=val_data_fold, transform=val_test_transforms)
            
            train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
            val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)

            model = CTClassifier(num_classes=NUM_CLASSES, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, freeze_feature_extractor=True).to(DEVICE)
            # 동결된 파라미터를 제외하고 옵티마이저 설정
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
            
            best_val_auc_fold = 0.0
            best_model_state_fold = None

            for epoch in range(NUM_EPOCHS):
                logger.info(f"Fold {fold+1}, Epoch {epoch+1}/{NUM_EPOCHS}")
                if epoch == FREEZE_FEATURE_EXTRACTOR_EPOCHS and PRETRAINED_WEIGHTS_PATH: # 지정된 에폭 후 동결 해제
                    logger.info("특징 추출기 동결 해제 및 옵티마이저 재설정.")
                    model.unfreeze_feature_extractor()
                    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10) # 학습률 낮춰서 미세 조정
                
                train_loss, train_acc, train_auc = train_one_epoch(model, train_loader_fold, criterion, optimizer, DEVICE)
                logger.info(f"Fold {fold+1} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

                val_metrics = evaluate_model(model, val_loader_fold, criterion, DEVICE, phase=f"Fold {fold+1} Validation")
                logger.info(f"Fold {fold+1} Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")

                if val_metrics['auc'] > best_val_auc_fold: # AUC를 기준으로 최적 모델 저장
                    best_val_auc_fold = val_metrics['auc']
                    best_model_state_fold = copy.deepcopy(model.state_dict())
                    logger.info(f"Fold {fold+1} - 새로운 최적 검증 AUC: {best_val_auc_fold:.4f} (Epoch {epoch+1})")
            
            fold_results.append({
                "fold": fold + 1,
                "best_val_auc": best_val_auc_fold,
                "best_model_state": best_model_state_fold
            })
            logger.info(f"Fold {fold+1} 완료. 해당 Fold 최고 검증 AUC: {best_val_auc_fold:.4f}")

        # 교차 검증 결과 요약 및 최적 모델 선택
        if fold_results:
            best_overall_fold_result = max(fold_results, key=lambda x: x['best_val_auc'])
            logger.info("\n--- 교차 검증 결과 요약 ---")
            for res in fold_results:
                logger.info(f"Fold {res['fold']}: Best Validation AUC = {res['best_val_auc']:.4f}")
            logger.info(f"전체 Fold 중 최적 성능 Fold: {best_overall_fold_result['fold']} (Val AUC: {best_overall_fold_result['best_val_auc']:.4f})")

            if best_overall_fold_result['best_model_state']:
                final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME)
                torch.save(best_overall_fold_result['best_model_state'], final_model_path)
                logger.info(f"최종 최적 모델(K-Fold) 저장 완료: {final_model_path}")
                
                # 최종 모델 로드 (테스트 및 XAI용)
                final_trained_model = CTClassifier(num_classes=NUM_CLASSES, pretrained_weights_path=None, freeze_feature_extractor=False).to(DEVICE) # 가중치 없이 구조만
                final_trained_model.load_state_dict(best_overall_fold_result['best_model_state'])
            else:
                logger.error("교차 검증 후 최적 모델 상태를 찾지 못했습니다.")
        else:
            logger.error("교차 검증 결과가 없습니다.")

    else: # Hold-out 방식
        logger.info("--- Hold-out 검증 시작 ---")
        
        # 1. Train+Validation 세트와 Test 세트 분리
        # VAL_SPLIT_RATIO가 0이고 TEST_SPLIT_RATIO도 0이면 전체 데이터로 학습 (K_FOLDS=0 일때)
        if TEST_SPLIT_RATIO == 0 and VAL_SPLIT_RATIO == 0 and K_FOLDS <=1:
             logger.info("TEST_SPLIT_RATIO와 VAL_SPLIT_RATIO가 모두 0입니다. 전체 데이터를 훈련 데이터로 사용합니다.")
             train_indices = list(range(len(all_data_dicts)))
             val_indices = []
             test_indices = []
        elif TEST_SPLIT_RATIO == 0 and VAL_SPLIT_RATIO > 0 and K_FOLDS <=1 :
             logger.info("TEST_SPLIT_RATIO가 0입니다. 전체 데이터를 Train/Validation으로 분할합니다.")
             train_indices, val_indices = train_test_split(
                list(range(len(all_data_dicts))),
                test_size=VAL_SPLIT_RATIO, # 이 경우 VAL_SPLIT_RATIO가 val_size가 됨
                stratify=labels_for_stratify,
                random_state=RANDOM_SEED
             )
             test_indices = [] # 테스트셋 없음
        elif TEST_SPLIT_RATIO > 0 :
            train_val_indices, test_indices = train_test_split(
                list(range(len(all_data_dicts))),
                test_size=TEST_SPLIT_RATIO,
                stratify=labels_for_stratify,
                random_state=RANDOM_SEED
            )
            # 2. Train 세트와 Validation 세트 분리
            if VAL_SPLIT_RATIO > 0 and len(train_val_indices) > 0 :
                # stratify를 위해 train_val_indices에 해당하는 레이블 사용
                train_val_labels = [labels_for_stratify[i] for i in train_val_indices]
                # VAL_SPLIT_RATIO는 전체 데이터에 대한 비율이므로, train_val_set 내에서의 비율로 재계산
                # (전체 데이터 수 * VAL_SPLIT_RATIO) / train_val 데이터 수
                relative_val_split_ratio = VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO) if (1.0 - TEST_SPLIT_RATIO) > 0 else 0
                
                if relative_val_split_ratio > 0 and relative_val_split_ratio < 1 and len(np.unique(train_val_labels)) > 1 :
                    train_indices, val_indices = train_test_split(
                        train_val_indices,
                        test_size=relative_val_split_ratio,
                        stratify=train_val_labels,
                        random_state=RANDOM_SEED
                    )
                else: # val_split_ratio가 부적절하거나 val set 만들 수 없는 경우
                    logger.warning(f"Validation set을 만들 수 없거나 비율이 부적절합니다 (relative_val_split_ratio: {relative_val_split_ratio}). Train 데이터만 사용합니다.")
                    train_indices = train_val_indices
                    val_indices = []
            else: # VAL_SPLIT_RATIO가 0이거나 train_val_indices가 없는 경우
                train_indices = train_val_indices
                val_indices = []
        else: # 위 모든 조건에 해당 안되는 경우 (논리적으로는 거의 없음)
            logger.error("데이터 분할 로직 오류. 기본값으로 전체 데이터를 훈련 데이터로 사용합니다.")
            train_indices = list(range(len(all_data_dicts)))
            val_indices = []
            test_indices = []


        train_data = [all_data_dicts[i] for i in train_indices]
        val_data = [all_data_dicts[i] for i in val_indices] if val_indices else []
        test_data = [all_data_dicts[i] for i in test_indices] if test_indices else []

        logger.info(f"데이터 분할: 훈련 {len(train_data)}개, 검증 {len(val_data)}개, 테스트 {len(test_data)}개")

        if not train_data:
            logger.error("훈련 데이터가 없습니다. 종료합니다.")
            sys.exit()

        train_dataset = Dataset(data=train_data, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
        
        val_loader = None
        if val_data:
            val_dataset = Dataset(data=val_data, transform=val_test_transforms)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)

        test_loader = None
        if test_data:
            test_dataset = Dataset(data=test_data, transform=val_test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS_DATALOADER, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)

        model = CTClassifier(num_classes=NUM_CLASSES, pretrained_weights_path=PRETRAINED_WEIGHTS_PATH, freeze_feature_extractor=True).to(DEVICE)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        
        best_val_metric_holdout = 0.0 # 검증 세트 성능 (예: AUC)
        best_model_state_holdout = None

        for epoch in range(NUM_EPOCHS):
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            if epoch == FREEZE_FEATURE_EXTRACTOR_EPOCHS and PRETRAINED_WEIGHTS_PATH:
                logger.info("특징 추출기 동결 해제 및 옵티마이저 재설정.")
                model.unfreeze_feature_extractor()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
            
            train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

            current_epoch_val_metric = train_auc # 검증셋 없으면 훈련 AUC 기준
            if val_loader:
                val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, phase="Validation")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1_score']:.4f}")
                current_epoch_val_metric = val_metrics['auc'] # 검증 AUC 기준 모델 저장
            else:
                logger.info("검증 데이터로더(val_loader)가 없습니다. 훈련 AUC를 기준으로 최적 모델을 저장합니다.")

            if current_epoch_val_metric > best_val_metric_holdout:
                best_val_metric_holdout = current_epoch_val_metric
                best_model_state_holdout = copy.deepcopy(model.state_dict())
                logger.info(f"새로운 최적 검증 지표(AUC): {best_val_metric_holdout:.4f} (Epoch {epoch+1})")
        
        if best_model_state_holdout:
            final_model_path = os.path.join(BASE_OUTPUT_DIR, MODEL_CHECKPOINT_NAME)
            torch.save(best_model_state_holdout, final_model_path)
            logger.info(f"최종 최적 모델(Hold-out) 저장 완료: {final_model_path}")
            final_trained_model = CTClassifier(num_classes=NUM_CLASSES, pretrained_weights_path=None, freeze_feature_extractor=False).to(DEVICE)
            final_trained_model.load_state_dict(best_model_state_holdout)
        elif model : # 저장된 최적 모델 없고, 현재 모델이라도 있으면 사용
             final_trained_model = model 
             logger.warning("저장된 최적 모델 상태(best_model_state_holdout)가 없습니다. 마지막 에폭의 모델을 사용합니다.")
        else:
            logger.error("훈련된 모델을 찾을 수 없습니다.")
            final_trained_model = None


    # --- 최종 모델 테스트 및 XAI ---
    if final_trained_model and test_loader:
        logger.info("\n--- 테스트셋 최종 평가 ---")
        final_trained_model.eval() # 평가 모드
        test_metrics = evaluate_model(final_trained_model, test_loader, criterion, DEVICE, phase="Test")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

        # --- XAI (Grad-CAM) 결과 생성 ---
        if XAI_NUM_SAMPLES_TO_VISUALIZE > 0 and test_data: # test_data는 원본 dict 리스트
            logger.info("\n--- XAI (Grad-CAM) 결과 생성 시작 ---")
            try:
                # GradCAM 대상 레이어 설정 (모델 구조에 따라 확인 필요)
                # 예: final_trained_model.feature_extractor.layer4 (ResNet의 경우)
                # 문자열로 지정 시 MONAI가 내부적으로 레이어를 찾음
                grad_cam_target_layer_str = f"feature_extractor.{XAI_TARGET_LAYER_NAME}"
                grad_cam_obj = GradCAM(nn_module=final_trained_model, target_layers=grad_cam_target_layer_str)
                logger.info(f"Grad-CAM 대상 레이어 설정: {grad_cam_target_layer_str}")

                visualized_count = 0
                # test_data는 전처리 전의 NIfTI 경로와 레이블 정보를 담은 dict 리스트
                # XAI 시각화를 위해 test_data에서 샘플을 가져와 val_test_transforms를 적용해야 함
                for i, item_dict_from_test_data_list in enumerate(test_data):
                    if visualized_count >= XAI_NUM_SAMPLES_TO_VISUALIZE:
                        break
                    try:
                        # XAI를 위한 단일 데이터 로드 및 전처리
                        # item_dict_from_test_data_list는 'image' (NIfTI 경로), 'label' (인코딩된 값), 'original_label', PATIENT_ID_COL 포함
                        
                        # val_test_transforms를 사용하여 모델 입력 형태로 변환
                        # 주의: LoadImageD는 파일 경로를 받으므로, item_dict_from_test_data_list의 "image" 키 사용
                        data_for_xai_model = val_test_transforms(item_dict_from_test_data_list) # 단일 dict 전달
                        input_tensor_gpu = data_for_xai_model["image"].unsqueeze(0).to(DEVICE) # 배치 차원 추가 및 GPU로 이동
                        
                        # 시각화를 위한 원본 (리사이즈된) 이미지 준비 (채널 차원 제거)
                        # input_tensor_gpu는 (1, C, D, H, W) 형태. 시각화 함수는 (D,H,W) 또는 (C,D,H,W)를 받을 수 있게 수정됨
                        original_image_for_vis_np = input_tensor_gpu.squeeze(0).cpu().numpy() # (C, D, H, W)

                        final_trained_model.eval() # 모델을 평가 모드로
                        with torch.no_grad(): # XAI 예측 시에는 그래디언트 계산 불필요
                            logits = final_trained_model(input_tensor_gpu)
                            predicted_class_idx = torch.argmax(logits, dim=1).item()
                        
                        # GradCAM 계산 (class_idx는 필수가 아님, 지정 안하면 가장 높은 확률 클래스 사용)
                        # 여기서는 명시적으로 predicted_class_idx 사용
                        cam_map = grad_cam_obj(x=input_tensor_gpu, class_idx=predicted_class_idx) # (1, D, H, W) 또는 (D,H,W)
                        cam_map_np = cam_map.squeeze().cpu().detach().numpy() # (D, H, W)

                        # 클래스 이름 및 환자 ID 가져오기
                        predicted_class_name = int_to_label_map.get(predicted_class_idx, f"Class_{predicted_class_idx}")
                        true_class_name = item_dict_from_test_data_list["original_label"] # manifest에서 읽은 원본 레이블
                        patient_id_for_xai = item_dict_from_test_data_list[PATIENT_ID_COL]
                        
                        logger.info(f"XAI 샘플 {i+1}: PID {patient_id_for_xai}, True {true_class_name}, Pred {predicted_class_name} (idx {predicted_class_idx})")

                        save_gradcam_slices(
                            original_image_for_vis_np, cam_map_np, patient_id_for_xai,
                            predicted_class_name, true_class_name, XAI_OUTPUT_DIR,
                            filename_prefix=f"xai_test_sample_{i}"
                        )
                        visualized_count += 1
                    except Exception as e_xai_sample:
                        logger.error(f"XAI 샘플 {i} (PID: {item_dict_from_test_data_list.get(PATIENT_ID_COL,'N/A')}) 처리 중 오류: {e_xai_sample}")
                        logger.debug(traceback.format_exc()) # 디버그용 상세 오류
                
                logger.info(f"XAI (Grad-CAM) 시각화 {visualized_count}개 완료. 저장 폴더: {XAI_OUTPUT_DIR}")

            except Exception as e_gradcam_setup:
                logger.error(f"Grad-CAM 설정 또는 실행 중 오류 발생: {e_gradcam_setup}")
                logger.error(traceback.format_exc())
        elif not test_data:
            logger.info("테스트 데이터가 없어 XAI 시각화를 건너뜁니다.")
        elif XAI_NUM_SAMPLES_TO_VISUALIZE == 0:
             logger.info("XAI_NUM_SAMPLES_TO_VISUALIZE가 0으로 설정되어 XAI 시각화를 건너뜁니다.")

    elif not final_trained_model:
        logger.error("훈련된 최종 모델이 없어 테스트 및 XAI를 진행할 수 없습니다.")
    elif not test_loader:
        logger.warning("테스트 데이터로더가 없어 테스트 및 XAI를 진행할 수 없습니다 (훈련 데이터는 있었을 수 있음).")
        
    logger.info("--- 모든 과정 완료 ---")