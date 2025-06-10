import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 임상 데이터 처리에 필요
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer # 임상 데이터 처리에 필요
from sklearn.pipeline import Pipeline # 임상 데이터 처리에 필요
from sklearn.decomposition import PCA # 오믹스 데이터 처리에 필요
import re
import joblib
import os

# --- 0. 전역 상수 및 설정 ---
PATIENT_ID_COL = 'bcr_patient_barcode'
PCA_MODELS_DIR = "pca_models_and_features"

if not os.path.exists(PCA_MODELS_DIR):
    os.makedirs(PCA_MODELS_DIR)
    print(f"디렉토리 생성됨: {PCA_MODELS_DIR}")

# --- 1. 임상 데이터 처리 함수들 (복원 및 사용) ---
RELEVANT_CLINICAL_COLS = [
    PATIENT_ID_COL, 'age_at_initial_pathologic_diagnosis', 'gender', 'ethnicity', 'race_list_race',
    'menopause_status', 'histological_type', 'stage_event_pathologic_stage',
    'stage_event_tnm_categories_pathologic_categories_pathologic_T',
    'stage_event_tnm_categories_pathologic_categories_pathologic_N',
    'stage_event_tnm_categories_pathologic_categories_pathologic_M',
    'breast_carcinoma_estrogen_receptor_status', 'breast_carcinoma_progesterone_receptor_status',
    'lab_proc_her2_neu_immunohistochemistry_receptor_status', 'number_of_lymphnodes_positive_by_he',
    'margin_status', 'history_of_neoadjuvant_treatment'
    # 생존 관련 컬럼이나 이전 타겟 생성용 컬럼은 특징으로 미포함
]

def load_clinical_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"임상 데이터 로드 완료: {file_path}, 형상: {df.shape}")
        # bcr_patient_barcode 컬럼이 있다면 PATIENT_ID_COL (전역변수)로 이름 변경
        if 'bcr_patient_barcode' in df.columns and PATIENT_ID_COL != 'bcr_patient_barcode':
            df.rename(columns={'bcr_patient_barcode': PATIENT_ID_COL}, inplace=True)
        if PATIENT_ID_COL not in df.columns:
            # 파일 내에 PATIENT_ID_COL과 정확히 일치하는 컬럼이 없는 경우, 대소문자 변환 또는 부분 일치 등으로 찾아볼 수 있음
            # 예시: df.columns = [col.lower() for col in df.columns] 후 다시 PATIENT_ID_COL 찾기
            # 여기서는 정확히 일치하는 컬럼이 있다고 가정
            raise ValueError(f"'{PATIENT_ID_COL}' 컬럼이 임상 데이터 파일에 없습니다. 컬럼명을 확인하세요.")
        # 환자 ID를 대문자로 통일 (TCGA 데이터 형식 고려)
        if PATIENT_ID_COL in df.columns:
            df[PATIENT_ID_COL] = df[PATIENT_ID_COL].astype(str).str.upper()
        return df
    except FileNotFoundError:
        print(f"오류: 임상 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다."); return pd.DataFrame()
    except Exception as e:
        print(f"임상 데이터 로드 중 오류 발생: {e}"); return pd.DataFrame()

def select_and_clean_clinical_data(df_full, selected_columns):
    if df_full.empty: return pd.DataFrame()
    
    # 선택된 컬럼 중 실제 데이터에 있는 컬럼만 선택
    cols_to_select = [col for col in selected_columns if col in df_full.columns]
    missing_cols = [col for col in selected_columns if col not in df_full.columns]
    if missing_cols:
        print(f"경고: 다음 요청된 임상 컬럼이 원본 데이터에 없어 제외됩니다: {missing_cols}")
    
    df = df_full[cols_to_select].copy() # .copy()로 SettingWithCopyWarning 방지

    common_na_values = ['Not Available', 'Unknown', '[Not Applicable]', '[Not Available]', '[Unknown]',
                        '[Not Evaluated]', 'Not Reported', '[Not Reported]', '', 'NaN', 'NA', None, '[Discrepancy]']
    for col in df.columns:
        if df[col].dtype == 'object': # 문자열 타입 컬럼에 대해서만 NA 처리 및 strip 적용
            df[col] = df[col].replace(common_na_values, np.nan).str.strip()

    # 특정 수치형 컬럼을 숫자 타입으로 변환 (오류 시 NaN)
    numerical_cols_to_convert = ['age_at_initial_pathologic_diagnosis', 'number_of_lymphnodes_positive_by_he']
    for col in numerical_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def preprocess_clinical_features_for_merging(df_clinical_selected):
    if df_clinical_selected.empty or PATIENT_ID_COL not in df_clinical_selected.columns:
        print("전처리할 임상 데이터가 없거나 환자 ID 컬럼이 없습니다."); return pd.DataFrame()

    # PATIENT_ID_COL을 인덱스로 설정 (이미 인덱스라면 그대로 사용)
    if PATIENT_ID_COL in df_clinical_selected.columns:
        df_features_raw = df_clinical_selected.set_index(PATIENT_ID_COL)
    else: # PATIENT_ID_COL이 컬럼에 없고 인덱스일 경우
        df_features_raw = df_clinical_selected.copy()


    # 수치형, 범주형 컬럼 분류
    numerical_cols = df_features_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_features_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    # 전처리 파이프라인 정의
    # 수치형: 중앙값으로 결측치 대치 후 표준화
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # 범주형: 최빈값으로 결측치 대치 후 원핫인코딩 (희소 행렬 반환 안 함)
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # 누락된 값 채우기
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False 대신 sparse_output=False
    ])

    transformers_list = []
    if numerical_cols:
        transformers_list.append(('num', numerical_pipeline, numerical_cols))
    if categorical_cols:
        transformers_list.append(('cat', categorical_pipeline, categorical_cols))

    if not transformers_list:
        print("처리할 유효한 임상 특징(수치형 또는 범주형)이 없습니다."); return pd.DataFrame(index=df_features_raw.index)

    # ColumnTransformer로 전처리 적용
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop') # 나머지 컬럼은 제외

    try:
        X_processed = preprocessor.fit_transform(df_features_raw)
    except Exception as e_ct:
        print(f"ColumnTransformer 처리 중 오류: {e_ct}")
        print("df_features_raw 내용 일부:")
        print(df_features_raw.head())
        print("수치형 컬럼:", numerical_cols)
        print("범주형 컬럼:", categorical_cols)
        for col in numerical_cols:
            print(f"수치형 컬럼 '{col}'의 dtype: {df_features_raw[col].dtype}, NaN 수: {df_features_raw[col].isnull().sum()}")
        for col in categorical_cols:
            print(f"범주형 컬럼 '{col}'의 dtype: {df_features_raw[col].dtype}, NaN 수: {df_features_raw[col].isnull().sum()}, 고유값 수: {df_features_raw[col].nunique()}")

        return pd.DataFrame(index=df_features_raw.index)


    # get_feature_names_out()으로 변환된 컬럼명 가져오기
    try:
        processed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError: # scikit-learn 구버전 호환
        processed_feature_names = []
        # 수동으로 컬럼명 생성 (get_feature_names_out 없을 시)
        ct_transformers = preprocessor.transformers_
        for name, trans, Lcols in ct_transformers:
            if trans == 'drop' or trans == 'passthrough': continue
            if hasattr(trans, 'get_feature_names_out'): # 파이프라인 내부 스텝에서 지원하는 경우
                # 파이프라인의 마지막 스텝(예: onehot 또는 scaler)에서 컬럼명 가져오기 시도
                if isinstance(trans, Pipeline):
                    last_step = trans.steps[-1][1]
                    if hasattr(last_step, 'get_feature_names_out'):
                         # onehot의 경우 input_features=Lcols 필요할 수 있음
                        try:
                            if isinstance(last_step, OneHotEncoder):
                                processed_feature_names.extend(last_step.get_feature_names_out(Lcols))
                            else: # StandardScaler 등
                                processed_feature_names.extend(Lcols) # 이름 변경 없음
                        except TypeError: # get_feature_names_out이 인자를 다르게 받는 경우
                             processed_feature_names.extend([f"{name}__{c}" for c in Lcols]) # 기본 이름
                    else: # 마지막 스텝에서 이름 못가져오면, 원래 컬럼명 사용
                        processed_feature_names.extend(Lcols)

                else: # 파이프라인이 아닌 단일 변환기
                     processed_feature_names.extend(trans.get_feature_names_out(Lcols))

            else: # get_feature_names_out이 없는 아주 구버전의 경우
                if name == 'num': # StandardScaler는 이름 변경 없음
                    processed_feature_names.extend(Lcols)
                elif name == 'cat': # OneHotEncoder 이름 수동 생성
                    oh_encoder = trans.named_steps['onehot'] # 파이프라인 가정
                    for i, col_name in enumerate(Lcols):
                        categories = oh_encoder.categories_[i]
                        processed_feature_names.extend([f"{col_name}_{cat}" for cat in categories])
        if not processed_feature_names and X_processed.shape[1] > 0 : # 그래도 이름이 없다면 임시 이름 부여
             processed_feature_names = [f"clinical_feat_{i}" for i in range(X_processed.shape[1])]


    # 전처리된 데이터를 DataFrame으로 변환
    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=df_features_raw.index)
    print(f"임상 특징 전처리 완료 (병합용). 최종 형상: {X_processed_df.shape}")
    return X_processed_df


# --- 오믹스 데이터 처리 함수 (이전과 동일, PCA 로직 없음) ---
def process_single_omics_file(file_path, patient_id_col_name, id_col_for_pivot, value_col,
                              omics_name, log_transform=False, filter_low_variance_quantile=0.05,
                              apply_scaling=True, aggfunc='mean',
                              omics_file_type='expression',
                              variant_class_col=None, min_mutation_freq_percent=1.0):
    print(f"  > {omics_name} ({omics_file_type}) 파일 기본 처리 시작: {file_path}")
    if not file_path or not os.path.exists(file_path):
        print(f"    경고: 파일 없음 또는 경로 오류 - {file_path}"); return pd.DataFrame()
    df_processed = pd.DataFrame()
    try:
        df_raw = pd.read_csv(file_path, low_memory=False)
        # 환자 ID 컬럼 처리 및 인덱스 설정
        original_patient_ids_in_file = None
        if patient_id_col_name in df_raw.columns:
            df_raw.rename(columns={patient_id_col_name: PATIENT_ID_COL}, inplace=True)
            df_raw[PATIENT_ID_COL] = df_raw[PATIENT_ID_COL].astype(str).str.upper() # 대문자 통일
            original_patient_ids_in_file = df_raw[PATIENT_ID_COL].unique()
            df_raw.set_index(PATIENT_ID_COL, inplace=True)
        elif PATIENT_ID_COL == df_raw.index.name: # 이미 인덱스에 PATIENT_ID_COL이 설정된 경우
            df_raw.index = df_raw.index.astype(str).str.upper()
            original_patient_ids_in_file = df_raw.index.unique()
        else: # PATIENT_ID_COL이 명시적으로 없는 경우 (예: 파일의 첫번째 열이 ID일 때)
            # 이 경우, 파일 형식에 대한 가정이 필요함. 여기서는 오류로 처리하거나,
            # 파일의 첫번째 컬럼을 patient_id로 간주하고 처리하는 로직 추가 가능.
            # 예: df_raw[PATIENT_ID_COL] = df_raw.iloc[:, 0].astype(str).str.upper()
            #     df_raw.set_index(PATIENT_ID_COL, inplace=True)
            #     original_patient_ids_in_file = df_raw.index.unique()
            print(f"    경고: {PATIENT_ID_COL} 컬럼이 {file_path}에 명시적으로 없습니다. 파일 형식을 확인하세요."); return pd.DataFrame()

        # 각 오믹스 타입별 초기 처리
        if omics_file_type == 'mutation':
            # PATIENT_ID_COL은 이미 인덱스. id_col_for_pivot, variant_class_col은 컬럼에 있어야 함.
            df_raw_reset = df_raw.reset_index() # 인덱스를 다시 컬럼으로 (crosstab용)
            required_cols_in_df = [PATIENT_ID_COL, id_col_for_pivot, variant_class_col]
            if not all(col in df_raw_reset.columns for col in required_cols_in_df):
                missing = [col for col in required_cols_in_df if col not in df_raw_reset.columns]
                print(f"    오류: 변이 데이터 필요 컬럼 부족 ({', '.join(missing)})."); return pd.DataFrame(index=original_patient_ids_in_file)

            meaningful_variants = ['Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Missense_Mutation', 'Nonsense_Mutation', 'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation']
            df_filtered = df_raw_reset[df_raw_reset[variant_class_col].isin(meaningful_variants)][required_cols_in_df].copy()
            df_filtered.dropna(subset=[PATIENT_ID_COL, id_col_for_pivot], inplace=True) # id_col_for_pivot도 NaN이면 안됨
            df_filtered[PATIENT_ID_COL] = df_filtered[PATIENT_ID_COL].astype(str) # 이미 대문자 처리됨
            df_filtered[id_col_for_pivot] = df_filtered[id_col_for_pivot].astype(str)
            df_filtered = df_filtered.drop_duplicates(subset=[PATIENT_ID_COL, id_col_for_pivot])

            if df_filtered.empty: print(f"    {file_path}: 필터링 후 변이 데이터 없음"); return pd.DataFrame(index=original_patient_ids_in_file)
            df_wide = pd.crosstab(df_filtered[PATIENT_ID_COL], df_filtered[id_col_for_pivot])
            df_wide = (df_wide > 0).astype(int)
            if df_wide.empty: return pd.DataFrame(index=df_filtered[PATIENT_ID_COL].unique()) # 여기서 unique ID 사용

            if min_mutation_freq_percent > 0 and df_wide.shape[0] > 0:
                min_count = int(np.ceil(len(df_wide) * (min_mutation_freq_percent / 100.0))); min_count = max(1, min_count)
                df_wide = df_wide.loc[:, df_wide.sum(axis=0) >= min_count]
            df_processed = df_wide

        elif omics_file_type == 'methylation':
            df_wide = df_raw.select_dtypes(include=np.number)
            if df_wide.empty: print(f"    {file_path}: 수치형 메틸레이션 데이터 없음"); return pd.DataFrame(index=original_patient_ids_in_file)
            df_wide = df_wide.apply(lambda x: x.fillna(x.median()), axis=0)
            df_processed = df_wide

        elif omics_file_type in ['expression', 'mirna', 'cnv']:
            df_raw_reset = df_raw.reset_index() # 인덱스를 다시 컬럼으로 (pivot_table용)
            required_cols_in_df = [PATIENT_ID_COL, id_col_for_pivot, value_col]
            if not all(col in df_raw_reset.columns for col in required_cols_in_df):
                missing = [col for col in required_cols_in_df if col not in df_raw_reset.columns]
                print(f"    오류: {omics_name} 데이터 필요 컬럼 부족 ({', '.join(missing)})."); return pd.DataFrame(index=original_patient_ids_in_file)

            df_selected = df_raw_reset[required_cols_in_df].copy()
            df_selected.dropna(subset=[value_col, id_col_for_pivot], inplace=True)
            df_selected[id_col_for_pivot] = df_selected[id_col_for_pivot].astype(str)
            # PATIENT_ID_COL은 이미 문자열 및 대문자 처리됨

            df_wide = df_selected.pivot_table(index=PATIENT_ID_COL, columns=id_col_for_pivot, values=value_col, aggfunc=aggfunc)
            df_wide.fillna(0, inplace=True) # 데이터 특성에 따라 처리
            df_processed = df_wide
        else:
            print(f"    오류: 알 수 없는 오믹스 파일 타입 - {omics_file_type}"); return pd.DataFrame(index=original_patient_ids_in_file)

        if df_processed.empty:
            print(f"    {omics_name} 초기 처리 후 데이터 없음."); return pd.DataFrame(index=original_patient_ids_in_file)

        print(f"    초기 전처리 후 형상 ({omics_name}): {df_processed.shape}")

        if omics_file_type != 'mutation':
            if log_transform: df_processed = np.log2(df_processed + 1)
            if filter_low_variance_quantile is not None and 0 < filter_low_variance_quantile < 1 and df_processed.shape[1] > 1:
                variances = df_processed.var(axis=0)
                if not variances.empty and variances.nunique() > 1:
                    variance_threshold = variances.quantile(filter_low_variance_quantile)
                    df_processed = df_processed.loc[:, variances > variance_threshold]
                    print(f"    분산 필터링 후 형상 ({omics_name}): {df_processed.shape}")
                elif not variances.empty:
                    df_processed = df_processed.loc[:, variances > 0] # 분산 0인 것만 제거
                    print(f"    분산 0인 특징 제거 후 형상 ({omics_name}): {df_processed.shape}")

            if df_processed.empty or df_processed.shape[1] == 0:
                print(f"    {omics_name} 분산 필터링 후 데이터 없음."); return pd.DataFrame(index=df_processed.index)

            if apply_scaling:
                scaler = StandardScaler()
                df_processed_scaled_values = scaler.fit_transform(df_processed)
                df_processed = pd.DataFrame(df_processed_scaled_values, columns=df_processed.columns, index=df_processed.index)
        
        # 최종 컬럼명에 omics_name 접두사 추가 (예: 'TP53' -> 'GeneExp_TP53')
        df_processed = df_processed.add_prefix(f"{omics_name}_")

    except FileNotFoundError: print(f"    경고: 파일 없음 - {file_path}"); return pd.DataFrame()
    except Exception as e:
        print(f"    오류 ({file_path} 처리 중): {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

    print(f"  > {omics_name} ({omics_file_type}) 파일 기본 처리 완료. 최종 형상: {df_processed.shape}")
    # 모든 환자 ID가 인덱스에 있는지 확인하고 반환 (original_patient_ids_in_file과 비교)
    # reindex를 통해 원래 파일에 있던 모든 환자 ID가 결과에 포함되도록 하고, 없는 데이터는 NaN으로 채움
    if original_patient_ids_in_file is not None:
        df_processed = df_processed.reindex(original_patient_ids_in_file.astype(str))

    return df_processed


# --- 3. 메인 전처리 실행 함수 (임상 데이터 포함 및 수정된 PCA 로직) ---
def main_preprocessing_revised():
    print("===== 샘플 기반 '암 vs 정상' 전처리 파이프라인 시작 (임상 데이터 포함, 특징 유출 방지 버전) =====")

    CLINICAL_DATA_FILE = r'C:\Users\21\Desktop\modeling\clinical_data.csv' # 임상 데이터 파일 경로

    omics_files_info = {
        'GeneExp': {'normal': r'C:\Users\21\Desktop\modeling\merged_gene_expression_normal.csv', 'cancer': r'C:\Users\21\Desktop\modeling\merged_gene_expression_cancer.csv', 'id_col': 'gene_name', 'value_col': 'tpm_unstranded', 'log': True, 'pca': 100, 'type': 'expression', 'var_filter': 0.05, 'scale': True},
        'miRNA': {'normal': r'C:\Users\21\Desktop\modeling\merged_breast_mirna_normal.csv', 'cancer': r'C:\Users\21\Desktop\modeling\merged_breast_mirna_cancer.csv', 'id_col': 'miRNA_ID', 'value_col': 'reads_per_million_miRNA_mapped', 'log': True, 'pca': 50, 'type': 'mirna', 'var_filter': 0.05, 'scale': True},
        'CNV': {'normal': r'C:\Users\21\Desktop\modeling\merged_breast_cnv_normal(유전자별).csv', 'cancer': r'C:\Users\21\Desktop\modeling\merged_breast_cnv_cancer(유전자별).csv', 'id_col': 'gene_name', 'value_col': 'copy_number', 'log': False, 'pca': 50, 'type': 'cnv', 'var_filter': 0.05, 'scale': True},
        'Meth': {'normal': r'C:\Users\21\Desktop\modeling\merged_breast_methylation_normal.csv', 'cancer': r'C:\Users\21\Desktop\modeling\merged_breast_methylation_cancer.csv', 'id_col': None, 'value_col': None, 'log': False, 'pca': 100, 'type': 'methylation', 'var_filter': 0.05, 'scale': True}, # id_col, value_col은 methylation 타입에서 내부적으로 사용 안함
        'Mut': {'normal': r'C:\Users\21\Desktop\modeling\merged_breast_mutation_normal.csv', 'cancer': r'C:\Users\21\Desktop\modeling\merged_breast_mutation_cancer.csv', 'id_col': 'Hugo_Symbol', 'value_col': None, 'variant_class_col': 'Variant_Classification', 'log': False, 'pca': None, 'type': 'mutation', 'var_filter': None, 'scale': False}
    }
    omics_pid_col_name = 'patient_barcode' # 오믹스 파일 내 환자 ID 컬럼명

    # 1. 임상 데이터 로드 및 전처리 (복원)
    X_clinical_processed_patient_indexed = pd.DataFrame() # 초기화
    clinical_df_full = load_clinical_data(CLINICAL_DATA_FILE)
    if not clinical_df_full.empty:
        clinical_df_selected = select_and_clean_clinical_data(clinical_df_full, RELEVANT_CLINICAL_COLS)
        if not clinical_df_selected.empty:
            X_clinical_processed_patient_indexed = preprocess_clinical_features_for_merging(clinical_df_selected)
            if X_clinical_processed_patient_indexed.empty:
                print("경고: 전처리된 임상 특징 데이터가 없습니다 (preprocess_clinical_features_for_merging 결과 비어있음).")
            else:
                print(f"전처리된 임상 데이터 형상: {X_clinical_processed_patient_indexed.shape}")
        else:
            print("경고: 선택 및 정제 후 임상 데이터가 없습니다.")
    else:
        print("경고: 원본 임상 데이터 로드 실패 또는 비어있습니다.")


    # 2. 각 오믹스 데이터 파일별 기본 전처리 (PCA 이전)
    processed_omics_data_no_pca = {}
    for omics_name, info in omics_files_info.items():
        processed_omics_data_no_pca[omics_name] = {}
        for sample_category in ['normal', 'cancer']:
            file_path = info.get(sample_category)
            if file_path: # 파일 경로가 제공된 경우에만 처리 시도
                df_temp = process_single_omics_file(
                    file_path=file_path, patient_id_col_name=omics_pid_col_name,
                    id_col_for_pivot=info['id_col'], value_col=info['value_col'],
                    omics_name=omics_name, log_transform=info['log'],
                    filter_low_variance_quantile=info.get('var_filter'),
                    apply_scaling=info.get('scale', True),
                    omics_file_type=info['type'], variant_class_col=info.get('variant_class_col'),
                    min_mutation_freq_percent=1.0 if info['type'] == 'mutation' else None
                )
                if df_temp is not None and not df_temp.empty: # None 체크 추가
                    processed_omics_data_no_pca[omics_name][sample_category] = df_temp
                elif df_temp is None: # process_single_omics_file이 None을 반환한 경우 (내부 오류)
                    print(f"경고: {omics_name} {sample_category} 처리 중 오류로 None 반환됨.")

            # else: 파일 경로가 info에 없는 경우는 이미 process_single_omics_file 내부에서 처리 (또는 여기서도 로그 가능)


    # 3. PCA 적용 (필요한 경우, normal/cancer 결합 데이터 사용)
    final_processed_omics_data = {}
    for omics_name, info in omics_files_info.items():
        final_processed_omics_data[omics_name] = {} # 각 오믹스 타입별 딕셔너리 초기화
        df_normal_no_pca = processed_omics_data_no_pca.get(omics_name, {}).get('normal')
        df_cancer_no_pca = processed_omics_data_no_pca.get(omics_name, {}).get('cancer')
        n_pca_components = info.get('pca')

        # PCA 적용 조건: n_pca_components가 지정되어 있고, normal 또는 cancer 데이터 중 하나라도 존재하며 비어있지 않을 때
        apply_pca_condition = n_pca_components is not None and \
                              ( (df_normal_no_pca is not None and not df_normal_no_pca.empty) or \
                                (df_cancer_no_pca is not None and not df_cancer_no_pca.empty) )

        if apply_pca_condition:
            print(f"\n--- {omics_name}: PCA 적용 시작 (N_components={n_pca_components}) ---")
            dfs_to_concat_for_pca_fit = []
            if df_normal_no_pca is not None and not df_normal_no_pca.empty: dfs_to_concat_for_pca_fit.append(df_normal_no_pca)
            if df_cancer_no_pca is not None and not df_cancer_no_pca.empty: dfs_to_concat_for_pca_fit.append(df_cancer_no_pca)

            if not dfs_to_concat_for_pca_fit: # 이론상 apply_pca_condition에서 걸러짐
                print(f"    정보: {omics_name}에 PCA를 적용할 데이터가 없습니다 (dfs_to_concat_for_pca_fit 비어있음). 건너<0xEB>니다.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                continue
            
            # outer join으로 모든 컬럼 포함 후, reindex로 컬럼 순서 맞추고 NaN 처리
            # 먼저 모든 데이터프레임의 컬럼 합집합 구하기
            all_cols = set()
            for df in dfs_to_concat_for_pca_fit:
                all_cols.update(df.columns)
            all_cols = sorted(list(all_cols)) # 일관된 순서 유지

            aligned_dfs_for_concat = []
            for df in dfs_to_concat_for_pca_fit:
                aligned_dfs_for_concat.append(df.reindex(columns=all_cols)) # 존재하지 않는 컬럼은 NaN으로 채워짐
            
            df_combined_for_pca_fit = pd.concat(aligned_dfs_for_concat, axis=0)

            # PCA는 NaN 값을 처리할 수 없으므로, 결측치 대치 (컬럼 중앙값)
            imputer_pca = SimpleImputer(strategy='median')
            # 숫자형 컬럼에 대해서만 imputation 수행
            num_cols_for_pca_imputation = df_combined_for_pca_fit.select_dtypes(include=np.number).columns
            if not num_cols_for_pca_imputation.empty:
                 df_combined_for_pca_fit[num_cols_for_pca_imputation] = imputer_pca.fit_transform(df_combined_for_pca_fit[num_cols_for_pca_imputation])
            else: # 숫자형 컬럼이 없는 경우 (거의 발생 안함)
                 print(f"    경고: {omics_name} PCA용 결합 데이터에 숫자형 컬럼이 없어 imputation 생략.")


            if df_combined_for_pca_fit.empty or df_combined_for_pca_fit.shape[1] == 0:
                 print(f"    경고: {omics_name} PCA 학습용 데이터(결합 및 imputation 후)가 비었거나 특징이 없습니다. 건너<0xEB>니다.")
                 if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                 if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                 continue

            current_n_samples, current_n_features = df_combined_for_pca_fit.shape
            max_possible_components = min(current_n_samples, current_n_features)
            adjusted_n_components = int(n_pca_components)
            if adjusted_n_components > max_possible_components:
                print(f"    경고: ({omics_name}) PCA 컴포넌트 수({adjusted_n_components}) > 최대({max_possible_components}), {max_possible_components}로 조정.")
                adjusted_n_components = max_possible_components
            
            if adjusted_n_components <= 0:
                print(f"    경고: ({omics_name}) 조정된 PCA 컴포넌트 수가 0 이하. PCA 건너<0xEB>니다.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
                continue

            pca = PCA(n_components=adjusted_n_components)
            try:
                pca.fit(df_combined_for_pca_fit)
                print(f"    {omics_name}: PCA 모델 학습 완료. 설명된 분산 비율 합: {np.sum(pca.explained_variance_ratio_):.4f}")
                pca_model_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_pca_model.joblib")
                original_features_filename = os.path.join(PCA_MODELS_DIR, f"{omics_name}_pre_pca_features.csv")
                joblib.dump(pca, pca_model_filename)
                pd.DataFrame({'feature': df_combined_for_pca_fit.columns.tolist()}).to_csv(original_features_filename, index=False)
                print(f"    PCA 모델 저장: {pca_model_filename}, 원본 특징명 저장: {original_features_filename}")

                # 개별 데이터셋 (normal, cancer)을 학습된 PCA 모델로 변환
                for sample_cat, df_orig_no_pca in [('normal', df_normal_no_pca), ('cancer', df_cancer_no_pca)]:
                    if df_orig_no_pca is not None and not df_orig_no_pca.empty:
                        df_aligned_for_transform = df_orig_no_pca.reindex(columns=all_cols) # 학습 시 사용된 컬럼 순서로 맞춤
                        # 변환 전에도 imputation 필요 (학습 시와 동일한 imputer 사용)
                        if not num_cols_for_pca_imputation.empty: # 숫자형 컬럼이 있었다면
                            df_aligned_for_transform[num_cols_for_pca_imputation] = imputer_pca.transform(df_aligned_for_transform[num_cols_for_pca_imputation])
                        
                        df_pca_transformed_values = pca.transform(df_aligned_for_transform)
                        df_pca_transformed = pd.DataFrame(df_pca_transformed_values,
                                                         index=df_orig_no_pca.index,
                                                         columns=[f"{omics_name}_PCA_PC{i+1}" for i in range(adjusted_n_components)])
                        final_processed_omics_data[omics_name][sample_cat] = df_pca_transformed
                        print(f"    {omics_name} {sample_cat} 데이터 PCA 변환 후 형상: {df_pca_transformed.shape}")
            except Exception as e_pca_fit_transform:
                print(f"    오류: {omics_name} PCA 학습 또는 변환 중 오류: {e_pca_fit_transform}. 이 오믹스는 PCA 없이 사용.")
                if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
                if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
        else: # PCA 적용 안 하는 경우
            if df_normal_no_pca is not None: final_processed_omics_data[omics_name]['normal'] = df_normal_no_pca
            if df_cancer_no_pca is not None: final_processed_omics_data[omics_name]['cancer'] = df_cancer_no_pca
            if n_pca_components is not None: # PCA 설정은 있었으나 데이터가 없어 적용 못한 경우
                 print(f"--- {omics_name}: PCA 적용 대상 데이터 부족 또는 설정 오류로 PCA 적용 안 함 ---")
            else: # PCA 설정 자체가 없는 경우
                 print(f"--- {omics_name}: PCA 적용 설정 없음 ---")


    # 4. 최종 샘플 기반 데이터셋 구축 (환자별로 특징 취합 + 임상 데이터 병합)
    processed_data_by_patient_for_final_merge = {}
    all_patient_ids_from_omics = set()
    for omics_name_key in final_processed_omics_data:
        for sample_category_key in ['normal', 'cancer']:
            df_current = final_processed_omics_data[omics_name_key].get(sample_category_key)
            if df_current is not None and not df_current.empty:
                all_patient_ids_from_omics.update(df_current.index.tolist())
    
    # 임상 데이터에만 있는 환자 ID도 포함 (선택적, 여기서는 오믹스 데이터 기준으로만 진행)
    # if not X_clinical_processed_patient_indexed.empty:
    #     all_patient_ids_from_omics.update(X_clinical_processed_patient_indexed.index.tolist())

    print(f"\n--- 총 {len(all_patient_ids_from_omics)} 명의 환자(오믹스 기준)에 대한 최종 특징 및 임상데이터 취합 시작 ---")

    for patient_id_str in all_patient_ids_from_omics: # patient_id는 문자열로 통일되어 있어야 함
        processed_data_by_patient_for_final_merge[patient_id_str] = {'normal': {}, 'cancer': {}}
        # 오믹스 데이터 취합
        for omics_name_key in final_processed_omics_data:
            df_normal_omics = final_processed_omics_data[omics_name_key].get('normal')
            if df_normal_omics is not None and patient_id_str in df_normal_omics.index:
                processed_data_by_patient_for_final_merge[patient_id_str]['normal'].update(df_normal_omics.loc[patient_id_str].to_dict())

            df_cancer_omics = final_processed_omics_data[omics_name_key].get('cancer')
            if df_cancer_omics is not None and patient_id_str in df_cancer_omics.index:
                processed_data_by_patient_for_final_merge[patient_id_str]['cancer'].update(df_cancer_omics.loc[patient_id_str].to_dict())
        
        # 임상 데이터 추가 (normal, cancer 샘플 모두에 동일한 환자의 임상 정보 적용)
        if not X_clinical_processed_patient_indexed.empty and patient_id_str in X_clinical_processed_patient_indexed.index:
            clinical_features_for_patient = X_clinical_processed_patient_indexed.loc[patient_id_str].to_dict()
            if processed_data_by_patient_for_final_merge[patient_id_str]['normal']: # normal 오믹스 데이터가 있을 때만 임상 추가
                processed_data_by_patient_for_final_merge[patient_id_str]['normal'].update(clinical_features_for_patient)
            if processed_data_by_patient_for_final_merge[patient_id_str]['cancer']: # cancer 오믹스 데이터가 있을 때만 임상 추가
                processed_data_by_patient_for_final_merge[patient_id_str]['cancer'].update(clinical_features_for_patient)


    all_samples_list = []
    for patient_id_str, data_by_type in processed_data_by_patient_for_final_merge.items():
        # Normal 샘플 생성: 오믹스 특징이 하나라도 있어야 함
        if data_by_type['normal'] and any(k not in RELEVANT_CLINICAL_COLS for k in data_by_type['normal'].keys()): # 오믹스 특징 존재 확인
            sample_dict_normal = {PATIENT_ID_COL: patient_id_str, 'target': 0, 'sample_id': f"{patient_id_str}_NORMAL"}
            sample_dict_normal.update(data_by_type['normal'])
            all_samples_list.append(sample_dict_normal)
        
        # Cancer 샘플 생성: 오믹스 특징이 하나라도 있어야 함
        if data_by_type['cancer'] and any(k not in RELEVANT_CLINICAL_COLS for k in data_by_type['cancer'].keys()): # 오믹스 특징 존재 확인
            sample_dict_cancer = {PATIENT_ID_COL: patient_id_str, 'target': 1, 'sample_id': f"{patient_id_str}_CANCER"}
            sample_dict_cancer.update(data_by_type['cancer'])
            all_samples_list.append(sample_dict_cancer)

    if not all_samples_list:
        print("오류: 최종 샘플 데이터가 없습니다. 전처리 과정 및 입력 파일을 확인하십시오."); return

    final_samples_df = pd.DataFrame(all_samples_list)
    print(f"\n최종 병합된 샘플 데이터 형상 (임상 데이터 포함): {final_samples_df.shape}")

    if final_samples_df.empty: print("오류: 병합 후 데이터프레임이 비어있습니다."); return
    if 'sample_id' in final_samples_df.columns:
        final_samples_df = final_samples_df.set_index('sample_id')
    else: print("치명적 오류: 'sample_id' 컬럼이 생성되지 않았습니다.") ; return
    if 'target' not in final_samples_df.columns: print("오류: 타겟 변수('target')가 최종 DF에 없습니다."); return

    y = final_samples_df['target']
    cols_to_drop_from_X = ['target']
    if PATIENT_ID_COL in final_samples_df.columns: cols_to_drop_from_X.append(PATIENT_ID_COL)
    X = final_samples_df.drop(columns=cols_to_drop_from_X, errors='ignore')

    print(f"X 생성 후 형상: {X.shape}, y 분포:\n{y.value_counts(normalize=True)}")

    # 최종 X에 대한 결측치 처리 (SimpleImputer - median)
    if X.isnull().values.any():
        print(f"\n최종 특징 행렬 X에 결측치 발견 (형상: {X.shape}). SimpleImputer (median)로 대치합니다.")
        # 모든 컬럼이 숫자형이라고 가정하고 imputation 시도. 만약 비숫자형 있다면 에러 발생 가능.
        # 더 안전하게는 numeric_cols만 선택해서 imputation
        numeric_cols = X.select_dtypes(include=np.number).columns
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns

        if not numeric_cols.empty:
            imputer_final = SimpleImputer(strategy='median')
            X_numeric_imputed = pd.DataFrame(imputer_final.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
            if not non_numeric_cols.empty:
                print(f"경고: 다음 비숫자형 컬럼은 imputation에서 제외됩니다: {list(non_numeric_cols)}")
                X = pd.concat([X_numeric_imputed, X[non_numeric_cols]], axis=1)[X.columns] # 원본 X의 컬럼 순서로
            else:
                X = X_numeric_imputed
            print(f"결측치 처리 후 X 형상: {X.shape}")
            if X.isnull().values.any(): print("경고: 최종 결측치 처리 후에도 NaN 존재. 확인 필요.")
        else:
            print("경고: 최종 X에 숫자형 컬럼이 없어 imputation을 건너<0xEB>니다.")
    else:
        print("\n최종 특징 행렬 X에 결측치가 없습니다.")

    # 데이터 저장 (파일명 변경)
    try:
        features_output_path = 'final_sample_based_features_clin_omics_revised.csv' # 임상+오믹스 포함 명시
        target_output_path = 'final_sample_based_target_clin_omics_revised.csv'
        X.to_csv(features_output_path, index=True)
        y.to_csv(target_output_path, index=True, header=['target'])
        print(f"\n전처리된 샘플 기반 데이터 저장 완료: {features_output_path}, {target_output_path}")
    except Exception as e: print(f"데이터 저장 중 오류: {e}")
    print("\n===== 샘플 기반 '암 vs 정상' 전처리 파이프라인 (임상 데이터 포함, 특징 유출 방지 버전) 종료 =====")

if __name__ == '__main__':
    main_preprocessing_revised()