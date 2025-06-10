import pandas as pd
import numpy as np
import os

# --- 0. 전역 상수 및 설정 ---
PATIENT_ID_COL = 'bcr_patient_barcode' # 환자 ID로 사용될 컬럼명

# --- 1. 임상 데이터 처리 함수들 ---

# 원본 임상 데이터에서 레이블 생성에 필요한 최소한의 컬럼 (환자 ID + 원본 병기 정보)
# 실제 원본 파일에 따라 'ajcc_pathologic_stage' 또는 유사한 컬럼명을 사용해야 합니다.
# 이 예제에서는 'stage_event_pathologic_stage'를 원본 병기 컬럼으로 가정합니다.
# 만약 다른 컬럼이라면 이 부분을 수정해주세요.
RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA = 'stage_event_pathologic_stage' # ★ 실제 원본 병기 컬럼명으로 변경!

# 최종적으로 저장될 컬럼명 (CT 분류 스크립트에서 사용할 이름)
FINAL_TARGET_LABEL_COL = 'ajcc_pathologic_stage' # ★ CT 분류 스크립트의 LABEL_COL과 일치시켜야 함

def load_clinical_data(file_path):
    """지정된 경로에서 임상 데이터 CSV 파일을 로드합니다."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"임상 데이터 로드 완료: {file_path}, 형상: {df.shape}")
        
        # 원본 파일에 'bcr_patient_barcode' 컬럼이 있다면 PATIENT_ID_COL에서 정의한 이름으로 통일
        if 'bcr_patient_barcode' in df.columns and PATIENT_ID_COL != 'bcr_patient_barcode':
            df.rename(columns={'bcr_patient_barcode': PATIENT_ID_COL}, inplace=True)
        
        if PATIENT_ID_COL not in df.columns:
            raise ValueError(f"'{PATIENT_ID_COL}' 컬럼이 임상 데이터에 없습니다. 컬럼명을 확인하세요.")
        return df
    except FileNotFoundError:
        print(f"오류: 임상 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다."); return pd.DataFrame()
    except Exception as e:
        print(f"임상 데이터 로드 중 오류 발생: {e}"); return pd.DataFrame()

def select_and_clean_target_label_data(df_full, patient_id_col, raw_label_col):
    """
    로드된 전체 임상 데이터에서 환자 ID와 원본 레이블 컬럼만 선택하고 기본적인 NA 값들을 np.nan으로 변환합니다.
    """
    if df_full.empty: 
        print("경고: 입력된 데이터프레임이 비어있습니다.")
        return pd.DataFrame()
        
    # 필수 컬럼 존재 확인
    if patient_id_col not in df_full.columns:
        print(f"경고: 환자 ID 컬럼 '{patient_id_col}'이 원본에 없어 빈 데이터프레임을 반환합니다.")
        return pd.DataFrame()
    if raw_label_col not in df_full.columns:
        print(f"경고: 원본 레이블 컬럼 '{raw_label_col}'이 원본에 없어 빈 데이터프레임을 반환합니다.")
        print(f"사용 가능한 컬럼: {df_full.columns.tolist()}")
        return pd.DataFrame()
        
    df = df_full[[patient_id_col, raw_label_col]].copy()
    
    common_na_values = [
        'Not Available', 'Unknown', '[Not Applicable]', '[Not Available]', '[Unknown]',
        '[Not Evaluated]', 'Not Reported', '[Not Reported]', '', 'NaN', 'NA', None, 
        '[Discrepancy]'
    ]
    # 레이블 컬럼에 대해서만 NA 값 처리 및 문자열 정리
    if df[raw_label_col].dtype == 'object':
        df[raw_label_col] = df[raw_label_col].replace(common_na_values, np.nan).str.strip()
    
    print(f"환자 ID 및 원본 레이블 선택, 기본 클리닝 후 형상: {df.shape}")
    return df

def clean_and_standardize_stage_for_ct_labels(stage_series, final_label_col_name):
    """
    CT 영상 분류 모델의 타겟 레이블로 사용할 병기 정보를 정리하고 표준화하는 함수입니다.
    ⚠️ 이 함수는 예시이며, 실제 데이터의 값 분포를 보고 직접 수정해야 합니다.
    결과 컬럼명은 final_label_col_name으로 지정됩니다.
    """
    # 데이터를 문자열로 바꾸고, 앞뒤 공백 제거, 모두 대문자로 변환합니다.
    # .astype(str) 전에 .fillna('')를 하여 NaN 값을 빈 문자열로 먼저 처리 (오류 방지)
    stage_series_cleaned = stage_series.fillna('').astype(str).str.strip().str.upper()

    # 병기 값들을 표준화된 형태로 매핑(mapping)합니다.
    # CT 분류 모델에서 사용할 최종 레이블 형태를 정의합니다.
    stage_mapping = {
        # 예시 매핑 (실제 데이터에 맞게 추가/수정 필요)
        'STAGE I': 'Stage I', 'I': 'Stage I',
        'STAGE IA': 'Stage IA', 'IA': 'Stage IA',
        'STAGE IB': 'Stage IB', 'IB': 'Stage IB',
        'STAGE II': 'Stage II', 'II': 'Stage II',
        'STAGE IIA': 'Stage IIA', 'IIA': 'Stage IIA',
        'STAGE IIB': 'Stage IIB', 'IIB': 'Stage IIB',
        'STAGE III': 'Stage III', 'III': 'Stage III',
        'STAGE IIIA': 'Stage IIIA', 'IIIA': 'Stage IIIA',
        'STAGE IIIB': 'Stage IIIB', 'IIIB': 'Stage IIIB',
        'STAGE IIIC': 'Stage IIIC', 'IIIC': 'Stage IIIC',
        'STAGE IV': 'Stage IV', 'IV': 'Stage IV',
        'STAGE 0': 'Stage 0', '0': 'Stage 0', # 제자리암 등
        'X': np.nan, 'STAGE X': np.nan, # 평가 불가
        '[NOT AVAILABLE]': np.nan,
        '[DISCREPANCY]': np.nan,
        'NOT REPORTED': np.nan,
        'UNKNOWN': np.nan,
        'NAN': np.nan,
        '': np.nan # 빈 문자열도 NaN으로 처리
    }
    
    standardized_stages = stage_series_cleaned.replace(stage_mapping)
    
    # 데이터프레임으로 변환하여 반환
    return pd.DataFrame({final_label_col_name: standardized_stages})


# --- 메인 실행 함수 ---
def create_ct_label_file():
    print("===== CT 영상 분류용 레이블 파일 생성 시작 =====")
    
    # ★★★ 사용자 설정 필요: 원본 임상 데이터 파일 경로 ★★★
    # 예시: clinical_data_for_ct_path = r'C:\Users\YourName\Desktop\TCGA_BRCA_clinical_data.csv'
    # 제공해주신 코드의 경로를 사용하거나, 실제 TCGA 등에서 다운받은 원본 임상 정보 파일 경로를 지정합니다.
    clinical_data_for_ct_path = r'C:\Users\21\Desktop\modeling\clinical_data.csv' # ★ 사용자의 실제 원본 임상 파일 경로로 수정!
    
    # 최종 저장될 파일명
    output_ct_label_file = "merged_clinical_data_final_preprocessed.csv"

    # 1. 원본 임상 데이터 로드
    df_clinical_raw = load_clinical_data(clinical_data_for_ct_path)
    
    if df_clinical_raw.empty:
        print(f"원본 임상 데이터를 로드할 수 없어 레이블 파일 생성을 중단합니다.")
        return

    # 2. 환자 ID 및 원본 병기 정보 컬럼 선택 및 기본 클리닝
    df_selected_labels = select_and_clean_target_label_data(df_clinical_raw, PATIENT_ID_COL, RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA)

    if df_selected_labels.empty:
        print(f"환자 ID 또는 원본 병기 정보를 선택할 수 없어 레이블 파일 생성을 중단합니다.")
        return

    # 3. 병기 정보 표준화
    print(f"\n레이블 컬럼 '{RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA}' 표준화를 시작합니다...")
    # 원본 병기 값 분포 확인 (표준화 규칙 수립에 도움)
    if not df_selected_labels[RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA].empty:
        print(f"'{RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA}' 컬럼의 원본 값 분포 (상위 20개):")
        print(df_selected_labels[RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA].value_counts(dropna=False).nlargest(20))
        print(f"총 고유값 개수 (NaN 포함): {df_selected_labels[RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA].nunique(dropna=False)}")
    
    df_standardized_labels = clean_and_standardize_stage_for_ct_labels(
        df_selected_labels[RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA],
        FINAL_TARGET_LABEL_COL # 최종 컬럼명 지정
    )

    # 4. 표준화된 레이블을 원본 데이터프레임(환자 ID 포함)과 병합
    # df_selected_labels에는 PATIENT_ID_COL과 RAW_LABEL_COL_FROM_ORIGINAL_CLINICAL_DATA가 있음
    # 인덱스를 기준으로 합치거나, PATIENT_ID_COL을 사용
    df_final_labels = pd.concat([
        df_selected_labels[[PATIENT_ID_COL]].reset_index(drop=True), # PATIENT_ID_COL만 가져오고 인덱스 리셋
        df_standardized_labels.reset_index(drop=True) # 표준화된 레이블도 인덱스 리셋하여 concat
    ], axis=1)
    
    # 5. 최종 레이블 컬럼에서 결측치(NaN)가 있는 행 제거
    print(f"\n'{FINAL_TARGET_LABEL_COL}' 컬럼의 결측치 수 (명시적 제거 전): {df_final_labels[FINAL_TARGET_LABEL_COL].isnull().sum()}")
    df_final_labels.dropna(subset=[FINAL_TARGET_LABEL_COL], inplace=True)
    print(f"'{FINAL_TARGET_LABEL_COL}' 컬럼의 결측치 수 (명시적 제거 후): {df_final_labels[FINAL_TARGET_LABEL_COL].isnull().sum()}")
    print(f"결측 레이블 제거 후 데이터 형상: {df_final_labels.shape}")

    # 6. 최종 레이블 분포 확인
    if not df_final_labels[FINAL_TARGET_LABEL_COL].empty:
        print(f"\n'{FINAL_TARGET_LABEL_COL}' 컬럼의 최종 값 분포:")
        print(df_final_labels[FINAL_TARGET_LABEL_COL].value_counts())
    else:
        print(f"경고: 최종 레이블 컬럼 '{FINAL_TARGET_LABEL_COL}'이 모든 값 제거 후 비어있습니다.")
        print("      `clean_and_standardize_stage_for_ct_labels` 함수의 `stage_mapping`을 확인하거나 원본 데이터를 확인해주세요.")

    # 7. (선택적) 환자 ID 중복 처리: 한 환자에게 여러 유효한 병기가 있는 경우 처리
    #    이 예제에서는 가장 처음 발견된 유효한 병기를 사용 (이미 dropna 등으로 처리되었을 수 있음)
    #    만약 그래도 중복이 있다면, drop_duplicates 사용
    num_duplicates = df_final_labels.duplicated(subset=[PATIENT_ID_COL]).sum()
    if num_duplicates > 0:
        print(f"경고: 환자 ID '{PATIENT_ID_COL}'에 중복된 항목 {num_duplicates}개 발견. 첫 번째 항목만 유지합니다.")
        df_final_labels.drop_duplicates(subset=[PATIENT_ID_COL], keep='first', inplace=True)
        print(f"환자 ID 중복 제거 후 데이터 형상: {df_final_labels.shape}")

    # 8. 결과 저장
    if not df_final_labels.empty:
        try:
            # PATIENT_ID_COL과 FINAL_TARGET_LABEL_COL만 저장
            df_to_save = df_final_labels[[PATIENT_ID_COL, FINAL_TARGET_LABEL_COL]]
            df_to_save.to_csv(output_ct_label_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ CT 영상 분류용 레이블 파일 생성 완료: {output_ct_label_file}")
            print(f"   최종 저장된 데이터 형상: {df_to_save.shape}")
        except Exception as e:
            print(f"오류: 파일 저장 중 문제 발생 - {e}")
    else:
        print("\n저장할 데이터가 없습니다. (모든 레이블이 유효하지 않거나 필터링됨)")

    print("\n===== CT 영상 분류용 레이블 파일 생성 완료 =====")

if __name__ == '__main__':
    create_ct_label_file()