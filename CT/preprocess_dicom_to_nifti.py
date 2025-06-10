# preprocess_dicom_to_nifti.py
# DICOM 시리즈를 NIfTI로 변환하고, 학습에 사용할 manifest 파일을 생성합니다.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk # DICOM 로딩 및 유효성 검사에 사용될 수 있음
import monai
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, OrientationD, SpacingD,
    Compose
)
from monai.data import NibabelWriter # NIfTI 저장을 위해 사용
from monai.utils import set_determinism

from tqdm import tqdm
import logging
import traceback
import torch # monai.transforms에서 내부적으로 사용될 수 있음

# --- 설정값 ---
# 입력 데이터 경로
CT_ROOT_DIR = r"G:\내 드라이브\2조\데이터\병합\CT\CT"  # 원본 CT DICOM 데이터 루트 폴더 경로
CLINICAL_DATA_FILE = r"G:\내 드라이브\2조\데이터\병합\merged_clinical_data_final_preprocessed.csv" # 환자 ID 및 레이블 포함 파일
PATIENT_ID_COL = 'bcr_patient_barcode' # 환자 ID 컬럼명
LABEL_COL = 'ajcc_pathologic_stage' # 예측 목표 레이블 컬럼명

# 출력 설정
PREPROCESSED_NIFTI_DIR = os.path.join(os.getcwd(), "preprocessed_nifti_data") # 전처리된 NIfTI 파일 저장 기본 폴더
MANIFEST_FILE_PATH = os.path.join(PREPROCESSED_NIFTI_DIR, "preprocessed_manifest.csv") # 생성될 manifest CSV 파일 경로
LOG_FILE_PATH_PREPROCESS = os.path.join(PREPROCESSED_NIFTI_DIR, "preprocess_log.txt")
os.makedirs(PREPROCESSED_NIFTI_DIR, exist_ok=True)

# 전처리 관련 설정
PIXDIM = (1.5, 1.5, 2.0) # MONAI SpacingD에서 사용할 목표 해상도 (mm)
RANDOM_SEED = 42 # 결과 재현성용 (여기서는 크게 중요하지 않지만 일관성 유지)
set_determinism(RANDOM_SEED)

# --- 로거 설정 ---
logger = logging.getLogger("preprocess")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE_PATH_PREPROCESS, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(module)s - %(message)s'))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s'))
logger.addHandler(stream_handler)

# --- DICOM 시리즈 탐색 함수 (원본 스크립트에서 가져옴) ---
def validate_dicom_series(series_path):
    try:
        dicom_files = [f for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
        if len(dicom_files) < 10: # 최소 10개 DICOM 파일 기준
            return False
        # SimpleITK를 사용하여 시리즈 ID 존재 여부 확인 (선택적, 시간 소요 가능)
        # reader = sitk.ImageSeriesReader()
        # series_ids = reader.GetGDCMSeriesIDs(series_path)
        # if not series_ids:
        #     return False
        return True
    except Exception as e:
        logger.error(f"시리즈 검증 중 오류 발생: {series_path} - {str(e)}")
        return False

def find_dicom_series_optimized(ct_root_dir, patient_id_col_name_placeholder): # patient_id_col_name_placeholder는 사용되지 않으나 호환성 위해 유지
    all_series_data_list = []
    logger.info(f"최적화된 DICOM 시리즈 탐색 시작 (대상 루트: {ct_root_dir})")
    if not os.path.isdir(ct_root_dir):
        logger.error(f"오류: CT 루트 디렉토리를 찾을 수 없습니다 - {ct_root_dir}")
        return []

    patient_id_folders = [pid for pid in os.listdir(ct_root_dir) if os.path.isdir(os.path.join(ct_root_dir, pid))]
    
    for patient_id in tqdm(patient_id_folders, desc="환자 폴더 탐색 중"):
        patient_path = os.path.join(ct_root_dir, patient_id)
        study_folders = [study for study in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, study))]
        if not study_folders: continue

        for study_id in study_folders:
            study_path = os.path.join(patient_path, study_id)
            series_folders = [series for series in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, series))]
            if not series_folders: continue

            for series_id_from_folder_name in series_folders:
                series_path = os.path.join(study_path, series_id_from_folder_name)
                if not any(f.lower().endswith('.dcm') for f in os.listdir(series_path)):
                    continue
                if validate_dicom_series(series_path):
                    all_series_data_list.append({
                        PATIENT_ID_COL: patient_id, # 실제 환자 ID 컬럼명 사용
                        "dicom_image_path": series_path, # 원본 DICOM 시리즈 경로
                        "study_id": study_id,
                        "series_id": series_id_from_folder_name
                    })
    num_total_series = len(all_series_data_list)
    if num_total_series > 0:
        logger.info(f"DICOM 시리즈 탐색 완료. 총 {num_total_series}개의 유효 시리즈 정보 찾음.")
    else:
        logger.warning("탐색된 유효 DICOM 시리즈가 없습니다.")
    return all_series_data_list

# --- 레이블 매핑 함수 (원본 스크립트에서 가져옴, manifest 생성 시 사용) ---
def get_label_mapping(df, label_col_name):
    unique_labels = sorted(df[label_col_name].astype(str).unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()} # 사용되진 않지만 참고용
    logger.info(f"레이블 매핑 생성됨: {label_to_int}")
    return label_to_int, int_to_label, len(unique_labels)

# --- 메인 전처리 실행 로직 ---
def main_preprocess():
    logger.info("--- DICOM to NIfTI 전처리 및 manifest 생성 시작 ---")
    logger.info(f"원본 DICOM 루트: {CT_ROOT_DIR}")
    logger.info(f"전처리된 NIfTI 저장 위치: {PREPROCESSED_NIFTI_DIR}")
    logger.info(f"Manifest 파일 저장 위치: {MANIFEST_FILE_PATH}")

    # 1. 임상 데이터 로드 및 레이블 매핑 준비
    try:
        clinical_df = pd.read_csv(CLINICAL_DATA_FILE)
        logger.info(f"임상 데이터 로드 완료. 총 {len(clinical_df)}개 행.")
    except FileNotFoundError:
        logger.error(f"임상 데이터 파일({CLINICAL_DATA_FILE})을 찾을 수 없습니다. 종료합니다.")
        return
    
    if PATIENT_ID_COL not in clinical_df.columns or LABEL_COL not in clinical_df.columns:
        logger.error(f"임상 데이터에 필요한 컬럼({PATIENT_ID_COL} 또는 {LABEL_COL})이 없습니다. 종료합니다.")
        return

    clinical_df = clinical_df.dropna(subset=[LABEL_COL])
    clinical_df[LABEL_COL] = clinical_df[LABEL_COL].astype(str)
    label_to_int_map, _, num_unique_classes = get_label_mapping(clinical_df, LABEL_COL)
    logger.info(f"총 {num_unique_classes}개의 고유 레이블 발견됨.")

    # 2. DICOM 시리즈 탐색
    dicom_series_infos = find_dicom_series_optimized(CT_ROOT_DIR, PATIENT_ID_COL)
    if not dicom_series_infos:
        logger.error("처리할 DICOM 시리즈를 찾지 못했습니다. 종료합니다.")
        return

    # 3. MONAI 전처리 변환 정의 (NIfTI 저장 전까지 적용할 변환)
    # image_only=False로 설정하여 메타데이터(affine 등)를 함께 로드
    # ensure_channel_first=True는 LoadImageD 내에서 처리 가능
    preprocessing_transforms = Compose([
        LoadImageD(keys=["image"], reader="ITKReader", image_only=False, ensure_channel_first=True),
        OrientationD(keys=["image"], axcodes="RAS"), # RAS 방향으로 재정렬
        SpacingD(keys=["image"], pixdim=PIXDIM, mode="bilinear"), # 지정된 해상도로 재샘플링
        # ResizeD나 ScaleIntensityRangePercentilesD 등은 학습 시점에 적용하는 것이 일반적이나,
        # 필요에 따라 여기서 추가할 수도 있음. 여기서는 기본적인 공간 정규화만 수행.
    ])

    # 4. 각 DICOM 시리즈 전처리 및 NIfTI 저장, manifest 정보 수집
    successfully_preprocessed_list = []
    writer = NibabelWriter() # NIfTI 저장을 위한 writer

    for series_info in tqdm(dicom_series_infos, desc="DICOM 시리즈 전처리 및 NIfTI 변환 중"):
        patient_id = series_info[PATIENT_ID_COL]
        original_dicom_path = series_info["dicom_image_path"]
        study_id = series_info["study_id"]
        series_id = series_info["series_id"]

        # 출력 NIfTI 파일 경로 생성
        # 예: PREPROCESSED_NIFTI_DIR / TCGA-XX-YYYY / study_id_series_id.nii.gz
        patient_nifti_dir = os.path.join(PREPROCESSED_NIFTI_DIR, patient_id)
        os.makedirs(patient_nifti_dir, exist_ok=True)
        
        # 파일명에 포함될 수 없는 문자 등을 안전하게 처리
        safe_study_id = "".join(c if c.isalnum() else '_' for c in study_id)
        safe_series_id = "".join(c if c.isalnum() else '_' for c in series_id)
        nifti_filename = f"{safe_study_id}_{safe_series_id}.nii.gz"
        output_nifti_path = os.path.join(patient_nifti_dir, nifti_filename)

        # 이미 처리된 파일이면 건너뛰기 (선택적)
        if os.path.exists(output_nifti_path):
            # logger.info(f"이미 처리된 파일 (건너뛰기): {output_nifti_path}")
            # Manifest에 추가하기 위해 레이블 정보 가져오기
            patient_clinical_data = clinical_df[clinical_df[PATIENT_ID_COL] == patient_id]
            if not patient_clinical_data.empty:
                original_label = patient_clinical_data[LABEL_COL].iloc[0]
                label_encoded = label_to_int_map.get(original_label, -1) # 없는 경우 -1 또는 다른 값
                if label_encoded != -1:
                     successfully_preprocessed_list.append({
                        PATIENT_ID_COL: patient_id,
                        "image_nifti_path": output_nifti_path,
                        "original_label": original_label,
                        "label_encoded": label_encoded,
                        "study_id": study_id,
                        "series_id": series_id,
                        "original_dicom_path": original_dicom_path
                    })
            continue

        # 데이터 로딩 및 전처리 시도
        data_to_transform = {"image": original_dicom_path}
        try:
            transformed_data = preprocessing_transforms(data_to_transform)
            image_tensor = transformed_data["image"] # (C, H, W, D) 형태의 텐서
            
            # NIfTI로 저장
            # image_tensor는 채널 차원(C)을 포함할 수 있음. NibabelWriter는 보통 (H,W,D) 또는 (H,W,D,C)를 기대.
            # 단일 채널(흑백) 의료 영상이므로 squeeze(0) 또는 [0]으로 채널 제거 가능
            if image_tensor.ndim == 4 and image_tensor.shape[0] == 1: # 예상되는 단일 채널 (1, D, H, W)
                image_data_to_save = image_tensor.squeeze(0).cpu().numpy()
            elif image_tensor.ndim == 3: # 이미 채널이 없는 3D 데이터 (D, H, W)
                image_data_to_save = image_tensor.cpu().numpy()
            else: # 채널이 여러 개이거나 (예: [3, D, H, W]), 다른 예상치 못한 차원의 경우
                logger.warning(f"단일 채널 이미지가 아니거나 예상치 못한 차원 ({image_tensor.shape}) 파일: {original_dicom_path}. 이 파일은 건너뜁니다.")
                continue # 다음 파일로 넘어감 (manifest에 추가되지 않음)

            # SpacingD, OrientationD 등을 거치면 transformed_data['image_meta_dict']['affine']에 업데이트된 affine 정보가 있음
            affine_matrix = transformed_data.get('image_meta_dict', {}).get('spatial_shape_original_affine', # LoadImageD ITKReader
                                                                            transformed_data.get('image_meta_dict', {}).get('original_affine', torch.eye(4)))

            if isinstance(affine_matrix, torch.Tensor):
                affine_matrix = affine_matrix.cpu().numpy()

            if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4,4) :
                 logger.warning(f"유효한 Affine 행렬을 얻지 못했습니다 ({type(affine_matrix)}). 기본 Affine 사용: {original_dicom_path}")
                 affine_matrix = np.eye(4) # 기본 단위 행렬


            writer.set_data_array(image_data_to_save, channel_dim=None) # channel_dim=None은 (H,W,D) 가정
            writer.set_metadata({"affine": affine_matrix, "original_affine": affine_matrix, "dtype": image_data_to_save.dtype.name})
            writer.write(output_nifti_path, verbose=False)
            
            # logger.info(f"성공: {original_dicom_path} -> {output_nifti_path}")

            # Manifest에 추가할 정보 구성
            patient_clinical_data = clinical_df[clinical_df[PATIENT_ID_COL] == patient_id]
            if not patient_clinical_data.empty:
                original_label = patient_clinical_data[LABEL_COL].iloc[0]
                label_encoded = label_to_int_map.get(original_label, -1)

                if label_encoded != -1: # 유효한 레이블이 있는 경우에만 추가
                    successfully_preprocessed_list.append({
                        PATIENT_ID_COL: patient_id,
                        "image_nifti_path": output_nifti_path, # NIfTI 파일 경로
                        "original_label": original_label,
                        "label_encoded": label_encoded,
                        "study_id": study_id,
                        "series_id": series_id,
                        "original_dicom_path": original_dicom_path # 원본 경로도 추적용으로 저장
                    })
                else:
                    logger.warning(f"환자 {patient_id}의 레이블 '{original_label}'을 매핑할 수 없습니다. (NIfTI: {output_nifti_path})")
            else:
                logger.warning(f"환자 {patient_id}에 대한 임상 데이터를 찾을 수 없습니다. (NIfTI: {output_nifti_path})")

        except Exception as e:
            logger.error(f"오류 발생 ({original_dicom_path}): {e}")
            # logger.debug(traceback.format_exc()) # 상세 오류 로그 (필요시 활성화)
            # 오류 발생 시, 이 시리즈는 manifest에 포함되지 않음

    # 5. Manifest 파일 저장
    if successfully_preprocessed_list:
        manifest_df = pd.DataFrame(successfully_preprocessed_list)
        manifest_df.to_csv(MANIFEST_FILE_PATH, index=False, encoding='utf-8-sig')
        logger.info(f"Manifest 파일 저장 완료: {MANIFEST_FILE_PATH} (총 {len(manifest_df)}개 항목)")
    else:
        logger.warning("성공적으로 전처리된 파일이 없어 manifest 파일을 생성하지 않았습니다.")

    logger.info("--- DICOM to NIfTI 전처리 및 manifest 생성 완료 ---")

if __name__ == "__main__":
    try:
        main_preprocess()
    except Exception as e:
        logger.critical(f"전처리 스크립트 실행 중 치명적 오류 발생: {e}")
        logger.critical(traceback.format_exc())