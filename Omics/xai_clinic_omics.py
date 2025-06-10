import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer # 명시적 임포트
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
import os
import joblib
import platform # 한글 폰트 설정용
import matplotlib.font_manager as fm # 한글 폰트 설정용

# --- 한글 폰트 설정 ---
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
    print("INFO: Windows 환경으로 감지되어 글꼴을 'Malgun Gothic'으로 설정합니다.")
elif platform.system() == 'Darwin': # macOS의 경우
    plt.rc('font', family='AppleGothic')
    print("INFO: macOS 환경으로 감지되어 글꼴을 'AppleGothic'으로 설정합니다.")
else: # Linux 또는 기타
    # Colab이나 Linux 환경에서는 적절한 한글 폰트 설치가 선행되어야 합니다.
    # 예: !sudo apt-get install -y fonts-nanum  (Colab에서)
    #     sudo apt-get install fonts-nanum* (Linux에서)
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # 일반적인 나눔고딕 경로
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path, size=10).get_name()
        plt.rc('font', family=font_name)
        print(f"INFO: Linux/기타 환경, '{font_name}' 글꼴을 사용합니다.")
    else:
        print("WARNING: 'NanumGothic'을 찾을 수 없습니다. 한글이 깨질 수 있습니다. 폰트 설치를 확인하세요.")
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지


# --- 0. 전역 상수 및 설정 ---
SAMPLE_ID_COL = 'sample_id' 
N_SPLITS_KFOLD = 5 
TOP_N_FEATURES_DISPLAY = 15
PLOTS_DIR = "modeling_plots"
PCA_MODELS_DIR = "pca_models_and_features" 

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# --- 1. 데이터 로드 함수 ---
def load_revised_data(feature_path, target_path):
    try:
        X = pd.read_csv(feature_path, index_col=SAMPLE_ID_COL) 
        y_df = pd.read_csv(target_path, index_col=SAMPLE_ID_COL)
        y = y_df.squeeze() 
        print(f"특징 데이터 로드 완료: {X.shape}")
        print(f"진단 타겟 데이터 로드 완료: {y.shape}, 분포:\n{y.value_counts(normalize=True)}")
        # 숫자형 변환 시도 (오류 시 NaN 발생 가능)
        for col in X.columns:
            if X[col].dtype == 'object':
                try: X[col] = pd.to_numeric(X[col])
                except ValueError: print(f"경고: 컬럼 '{col}'은 숫자형 변환 불가. NaN으로 처리될 수 있음.")
        return X, y
    except FileNotFoundError as e: print(f"오류: 데이터 파일 로드 실패 - {e}"); return None, None
    except Exception as e: print(f"데이터 로드 중 일반 오류 발생: {e}"); return None, None

# --- 2. 평가 함수 ---
def evaluate_classification_model(model_name, y_true, y_pred, y_pred_proba=None, le=None, fold_num=None):
    fold_str = f" (Fold {fold_num})" if fold_num is not None else " (Test Set)"
    print(f"\n===== {model_name}{fold_str} 평가 결과 =====")
    
    labels_for_report = []
    custom_target_names = []

    if le and hasattr(le, 'classes_') and le.classes_ is not None:
        labels_for_report = sorted(list(le.classes_)) # 항상 정렬된 순서로
        try:
            if len(labels_for_report) == 2: # 이진 분류
                 # le.classes_의 0과 1 순서에 맞게 이름 매핑
                 name_map = {0: 'Normal_Sample', 1: 'Cancer_Sample'}
                 custom_target_names = [name_map.get(l, str(l)) for l in labels_for_report]
            else: # 다중 클래스 (현재 시나리오에서는 아님)
                 custom_target_names = [str(le.inverse_transform([l])[0]) for l in labels_for_report]
        except Exception as e_label_names:
            print(f"  target_names 생성 중 경고/오류: {e_label_names}. 기본 숫자 레이블 사용.")
            custom_target_names = [str(l) for l in labels_for_report]
    else:
        labels_for_report = sorted(list(np.unique(y_true)))
        custom_target_names = [str(l) for l in labels_for_report]
        if len(labels_for_report) == 2 : 
            name_map = {0: 'Normal_Sample(auto)', 1: 'Cancer_Sample(auto)'}
            custom_target_names = [name_map.get(l, str(l)) for l in labels_for_report]
    
    if not custom_target_names or len(labels_for_report) != len(custom_target_names) :
        custom_target_names = [str(l) for l in labels_for_report] # 최종 fallback

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_for_report)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_for_report)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_for_report)
    
    print(f"정확도: {accuracy:.4f}, 정밀도(W): {precision:.4f}, 재현율(W): {recall:.4f}, F1(W): {f1:.4f}")
    print("혼동 행렬:"); print(confusion_matrix(y_true, y_pred, labels=labels_for_report))
    print("\n분류 보고서:"); 
    try: 
        print(classification_report(y_true, y_pred, labels=labels_for_report, target_names=custom_target_names, zero_division=0))
    except Exception as e_report: print(f"분류 보고서 생성 오류: {e_report}")

    roc_auc = np.nan
    if y_pred_proba is not None and len(labels_for_report) == 2 :
        try: roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1]); print(f"ROC AUC: {roc_auc:.4f}")
        except Exception as e: print(f"ROC AUC 계산 오류: {e}")
    elif len(labels_for_report) < 2: print("ROC AUC: 유효 클래스 2개 미만")
    else: print("ROC AUC: 다중 클래스는 현재 미지원")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc if not np.isnan(roc_auc) else 0.0}

# --- 3. 특징 이름 수정 함수 ---
def sanitize_feature_names(df_to_sanitize):
    cols = df_to_sanitize.columns; new_cols = []; 
    name_counts = {} 
    for original_col_name in cols:
        sanitized_c = str(original_col_name)
        sanitized_c = re.sub(r"[^A-Za-z0-9_]", "_", sanitized_c)
        sanitized_c = re.sub(r"_+", "_", sanitized_c) 
        sanitized_c = sanitized_c.strip('_')
        if not sanitized_c or sanitized_c.isdigit(): 
            sanitized_c = f"feature_{sanitized_c}" if sanitized_c else f"feature_{len(new_cols)}"
        
        current_name_to_add = sanitized_c
        # 중복 처리: 이미 생성된 new_cols 리스트에서 확인
        suffix_count = 1
        while current_name_to_add in new_cols:
            current_name_to_add = f"{sanitized_c}_{suffix_count}"
            suffix_count += 1
        new_cols.append(current_name_to_add)
            
    df_to_sanitize.columns = new_cols
    return df_to_sanitize

# --- 4. 특징 중요도 추출 및 SHAP 시각화 함수 ---
def get_and_print_feature_importances(model, model_name, feature_names, X_test_for_shap=None, top_n=15, fold_num=None, save_plots=True, plots_dir=PLOTS_DIR):
    # ... (이전 답변의 get_and_print_feature_importances 함수 내용과 거의 동일하게 유지, 
    #      단, feature_names는 X_test_for_shap.columns를 사용하도록 내부에서 조정하는 것이 더 안전할 수 있음) ...
    fold_str_filename = f"_fold{fold_num}" if fold_num is not None else ""
    fold_str_title = f" (Fold {fold_num})" if fold_num is not None else ""
    print(f"\n--- {model_name}{fold_str_title}: 상위 {top_n}개 중요 특징 ---")
    
    model_specific_importance_df = pd.DataFrame()
    actual_feature_names_list = list(feature_names) 

    if hasattr(model, 'coef_'): 
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        if len(coefs) == len(actual_feature_names_list):
            model_specific_importance_df = pd.DataFrame({'feature': actual_feature_names_list, 'importance': np.abs(coefs)})
            model_specific_importance_df = model_specific_importance_df.sort_values(by='importance', ascending=False)
    elif hasattr(model, 'feature_importances_'):
        if len(model.feature_importances_) == len(actual_feature_names_list):
            model_specific_importance_df = pd.DataFrame({'feature': actual_feature_names_list, 'importance': model.feature_importances_})
            model_specific_importance_df = model_specific_importance_df.sort_values(by='importance', ascending=False)
    
    if not model_specific_importance_df.empty:
        print("모델 고유 특징 중요도:")
        print(model_specific_importance_df.head(top_n))
    else: print(f"{model_name}: 특징 중요도 직접 추출 불가 또는 길이 불일치.")

    shap_importance_df = pd.DataFrame()
    if X_test_for_shap is not None and not X_test_for_shap.empty and shap is not None:
        print(f"\n--- {model_name}{fold_str_title}: SHAP 값 기반 특징 중요도 (상위 {top_n}개) ---")
        shap_values_for_summary = None
        X_test_df_for_shap = X_test_for_shap.copy() # 원본 보존
        if not isinstance(X_test_df_for_shap, pd.DataFrame): 
             X_test_df_for_shap = pd.DataFrame(X_test_df_for_shap, columns=actual_feature_names_list)
        
        # SHAP 계산 전 X_test_df_for_shap의 dtype을 명시적으로 float으로 변환 (TreeExplainer 호환성)
        try:
            X_test_df_for_shap = X_test_df_for_shap.astype(float)
        except Exception as e_dtype:
            print(f"SHAP용 X_test 데이터 타입 변환 중 오류: {e_dtype}")
            # 변환 실패 시에도 일단 진행 시도

        try:
            explainer = None
            if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
                # TreeExplainer는 데이터프레임 컬럼명이 문자열이어야 하고, 데이터는 숫자형이어야 함
                shap_values = explainer.shap_values(X_test_df_for_shap) 
                if isinstance(shap_values, list) and len(shap_values) == 2: shap_values_for_summary = shap_values[1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2: shap_values_for_summary = shap_values
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] == 2 : shap_values_for_summary = shap_values[:,:,-1] 
            elif isinstance(model, LogisticRegression) or (isinstance(model, SVC) and model.kernel == 'linear'):
                explainer = shap.LinearExplainer(model, X_test_df_for_shap); shap_values = explainer.shap_values(X_test_df_for_shap); shap_values_for_summary = shap_values
            
            if explainer is not None and shap_values_for_summary is not None:
                if shap_values_for_summary.ndim == 1: shap_values_for_summary = shap_values_for_summary.reshape(1, -1) if X_test_df_for_shap.shape[0]==1 else shap_values_for_summary.reshape(-1,1) if X_test_df_for_shap.shape[1]==1 else shap_values_for_summary 
                if shap_values_for_summary.shape[1] == X_test_df_for_shap.shape[1] and shap_values_for_summary.shape[0] == X_test_df_for_shap.shape[0]:
                    mean_abs_shap = np.abs(shap_values_for_summary).mean(axis=0)
                    if len(mean_abs_shap) == X_test_df_for_shap.shape[1]:
                        shap_importance_df = pd.DataFrame({'feature': X_test_df_for_shap.columns, 'shap_importance': mean_abs_shap})
                        shap_importance_df = shap_importance_df.sort_values(by='shap_importance', ascending=False)
                        print(shap_importance_df.head(top_n))
                        if save_plots:
                            try:
                                # Bar plot
                                plt.figure(figsize=(12, 8)) # 예시: Figure 크기 조정 (필요에 따라 값 변경)
                                shap.summary_plot(shap_values_for_summary, X_test_df_for_shap, plot_type="bar", feature_names=X_test_df_for_shap.columns, max_display=top_n, show=False)
                                plt.title(f"SHAP Bar Plot - {model_name}{fold_str_title}")
                                plt.savefig(os.path.join(plots_dir, f"shap_summary_bar_{model_name}{fold_str_filename}.png"), bbox_inches='tight'); plt.close()

                                # Dot plot
                                plt.figure(figsize=(12, 8)) # 예시: Figure 크기 조정
                                shap.summary_plot(shap_values_for_summary, X_test_df_for_shap, feature_names=X_test_df_for_shap.columns, max_display=top_n, show=False)
                                plt.title(f"SHAP Dot Plot - {model_name}{fold_str_title}")
                                plt.savefig(os.path.join(plots_dir, f"shap_summary_dot_{model_name}{fold_str_filename}.png"), bbox_inches='tight'); plt.close()
                                print(f"SHAP summary plots saved to '{plots_dir}/' for {model_name}{fold_str_filename}.")
                            except Exception as e_plot: print(f"SHAP 플롯 저장 중 오류: {e_plot}")
                else: print(f"SHAP 값 형상 불일치: {shap_values_for_summary.shape} vs X_test {X_test_df_for_shap.shape}")
            else: print(f"{model_name}: SHAP Explainer 구성/값 추출 실패.")
        except Exception as e_shap: print(f"SHAP 값 계산/플롯 생성 중 오류 ({model_name}{fold_str_title}): {e_shap}")
    return model_specific_importance_df, shap_importance_df


# --- 5. PCA 컴포넌트 해석 및 시각화 함수 ---
def visualize_pca_component_loadings(pca_feature_name, top_n=10, plots_dir=PLOTS_DIR, pca_models_dir=PCA_MODELS_DIR):
    print(f"\n--- PCA 컴포넌트 해석 시도: {pca_feature_name} ---")
    try:
        # 예: pca_feature_name = "GeneExp_PCA_PC1"
        match = re.match(r"(.+?)_PCA_PC(\d+)", pca_feature_name) # sample_type 정보 제거
        if not match:
            print(f"오류: PCA 특징 이름 형식 오류 - {pca_feature_name}. 예상 형식: 'OmicsName_PCA_PCx'")
            return

        omics_name_from_feature = match.group(1) # 예: GeneExp
        pc_index = int(match.group(2)) - 1 # PC 번호는 1부터 시작하므로 인덱스는 -1

        # PCA 모델 및 특징 파일명 구성 (오믹스 타입명만 사용)
        # 전처리 스크립트에서 저장된 파일명 형식과 일치해야 함
        pca_model_path = os.path.join(pca_models_dir, f"{omics_name_from_feature}_pca_model.joblib")
        pre_pca_features_path = os.path.join(pca_models_dir, f"{omics_name_from_feature}_pre_pca_features.csv")

        if not (os.path.exists(pca_model_path) and os.path.exists(pre_pca_features_path)):
            print(f"오류: {omics_name_from_feature}에 대한 PCA 정보 파일 없음.")
            print(f"  모델 경로 확인: {pca_model_path}")
            print(f"  특징 경로 확인: {pre_pca_features_path}")
            return

        print(f"  > {omics_name_from_feature}의 PCA 정보 로드 중 ({pca_feature_name})...")
        pca_model = joblib.load(pca_model_path)
        original_feature_names_df = pd.read_csv(pre_pca_features_path)
        original_feature_names = original_feature_names_df['feature'].tolist()

        if pc_index >= len(pca_model.components_):
            print(f"  오류: PC 인덱스({pc_index+1})가 PCA 모델의 컴포넌트 수({len(pca_model.components_)})를 초과합니다.")
            return

        loadings = pca_model.components_[pc_index]
        if len(loadings) != len(original_feature_names):
            print(f"  오류: 로딩 벡터 길이({len(loadings)})와 원본 특징 수({len(original_feature_names)})가 일치하지 않습니다.")
            return

        loadings_df = pd.DataFrame({'original_feature': original_feature_names, 'loading': loadings})
        loadings_df['abs_loading'] = np.abs(loadings_df['loading'])
        loadings_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        loadings_df.dropna(subset=['abs_loading'], inplace=True)

        if loadings_df.empty:
            print(f"  정보: 유효한 로딩 값이 없습니다 ({pca_feature_name}).")
            return

        sorted_loadings_df = loadings_df.sort_values(by='abs_loading', ascending=False).head(top_n)

        if sorted_loadings_df.empty:
            print(f"  정보: 상위 로딩 특징을 찾을 수 없습니다 ({pca_feature_name}).")
            return

        print(f"\n  --- {pca_feature_name} (from {omics_name_from_feature} data) 구성 상위 {len(sorted_loadings_df)}개 원본 특징 ---")
        print(sorted_loadings_df[['original_feature', 'loading']])

        plt.figure(figsize=(12, max(6, len(sorted_loadings_df) * 0.5)))
        sns.barplot(x='loading', y='original_feature', data=sorted_loadings_df, palette="coolwarm", hue='original_feature', dodge=False, legend=False) # hue, legend 수정
        plt.title(f"Top {len(sorted_loadings_df)} Original Feature Loadings for {pca_feature_name}\n(derived from {omics_name_from_feature} data processing)", fontsize=14)
        plt.xlabel("Loading Value", fontsize=12); plt.ylabel("Original Feature", fontsize=12)
        plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tight_layout()

        safe_pca_feature_name = re.sub(r'[\\/*?:"<>|]',"", pca_feature_name) # 파일명 특수문자 제거
        plot_filename = os.path.join(plots_dir, f"pca_loadings_{safe_pca_feature_name}.png")
        plt.savefig(plot_filename, bbox_inches='tight'); plt.close()
        print(f"  PCA 로딩 플롯 저장: {plot_filename}")

    except Exception as e:
        print(f"PCA 컴포넌트 해석 중 오류 ({pca_feature_name}): {e}")
        import traceback
        traceback.print_exc()
        
# --- 6. 모델 성능 비교 시각화 함수 ---
def visualize_cv_model_performance(all_model_metrics_dict=None, metrics_to_plot=['accuracy', 'f1', 'roc_auc'], plots_dir=PLOTS_DIR):
    # ... (이전 답변의 visualize_cv_model_performance 함수 내용과 동일하게 유지) ...
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    if not all_model_metrics_dict: print("시각화할 모델 성능 데이터가 없습니다."); return
    metrics_df = pd.DataFrame(all_model_metrics_dict).T 
    for metric in metrics_to_plot:
        if metric not in metrics_df.columns: print(f"경고: '{metric}' 평가지표 없음."); continue
        plt.figure(figsize=(12, 8)); sorted_df = metrics_df[metric].dropna().sort_values(ascending=False)
        if sorted_df.empty: print(f"'{metric}'에 유효 데이터 없음."); plt.close(); continue
        if sns: sns.barplot(x=sorted_df.index, y=sorted_df.values, hue=sorted_df.index, palette="viridis", legend=False, dodge=False) # .values 사용
        else: plt.bar(sorted_df.index, sorted_df.values)
        plt.title(f"모델별 교차 검증 평균 {metric.upper()} 비교", fontsize=16); plt.xlabel("모델", fontsize=12); plt.ylabel(metric.capitalize(), fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10); plt.yticks(fontsize=10)
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] and sorted_df.min() >= 0 and sorted_df.max() <=1 : plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        plot_filename = os.path.join(plots_dir, f"cv_avg_performance_{metric}_comparison.png")
        plt.savefig(plot_filename, bbox_inches='tight'); plt.close(); print(f"'{metric}' 비교 플롯 저장 완료: {plot_filename}")


# --- 7. 메인 모델링 함수 ---
def main_modeling():
    print("\n===== 모델링 파이프라인 시작 (샘플 기반 '암 vs 정상', 교차 검증) =====")
    feature_file = 'final_sample_based_features_clin_omics_revised.csv'
    target_file = 'final_sample_based_target_clin_omics_revised.csv'
    X_full, y_full_encoded = load_revised_data(feature_file, target_file)
    if X_full is None or y_full_encoded is None: print("데이터 로드 실패."); return

    le = LabelEncoder(); le.fit(y_full_encoded) 

    print("\n--- 전체 특징 이름 수정 중 (XGBoost/LightGBM 호환성) ---")
    X_full_sanitized = sanitize_feature_names(X_full.copy())
    print(f"전체 특징 이름 수정 완료. 수정된 특징 수: {len(X_full_sanitized.columns)}")
    if len(X_full_sanitized.columns) != len(set(X_full_sanitized.columns)):
        print("치명적 경고: 특징 이름 수정 후 중복된 컬럼명 존재!")


    if X_full_sanitized.isnull().values.any():
        print(f"\n결측치 발견. SimpleImputer (median)로 전체 X 데이터에 대치 적용...")
        numeric_cols = X_full_sanitized.select_dtypes(include=np.number).columns
        X_numeric_data = X_full_sanitized[numeric_cols]
        X_non_numeric_data = X_full_sanitized.select_dtypes(exclude=np.number) # 비숫자형 데이터 분리
        
        imputed_df_list = []
        if not X_numeric_data.empty:
            imputer = SimpleImputer(strategy='median')
            X_imputed_numeric_values = imputer.fit_transform(X_numeric_data)
            X_imputed_numeric_df = pd.DataFrame(X_imputed_numeric_values, columns=numeric_cols, index=X_full_sanitized.index)
            imputed_df_list.append(X_imputed_numeric_df)
        
        if not X_non_numeric_data.empty: # 비숫자형 컬럼이 있었다면 다시 합치기
            imputed_df_list.append(X_non_numeric_data)
        
        if imputed_df_list:
            X_full_sanitized = pd.concat(imputed_df_list, axis=1)
            X_full_sanitized = X_full_sanitized[X_full.columns] # 원본 컬럼 순서 유지 시도
        
        if X_full_sanitized.isnull().values.any(): print("경고: Imputation 후에도 NaN 존재!")
        else: print("전체 X 데이터에 대한 결측치 대치 완료.")
    else:
        print("전체 X 데이터에 결측치가 없습니다 (로드 직후 또는 이전 처리 완료).")


    models_to_run = {
        "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=2000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced_subsample'),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1), # use_label_encoder=False 추가
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', verbosity=-1),
        "SVM (Linear Kernel)": SVC(kernel='linear', C=1.0, probability=True, random_state=42, class_weight='balanced'), 
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, probability=True, random_state=42, class_weight='balanced'), 
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,32), alpha=0.01, max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=10)
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
    all_models_cv_metrics = {}
    all_models_aggregated_model_importances = {}
    all_models_aggregated_shap_importances = {}

    for model_name, model_prototype in models_to_run.items():
        print(f"\n\n<<<<< {model_name} 모델 교차 검증 시작 >>>>>")
        fold_metrics_list = []
        current_model_fold_model_imp_dfs = []
        current_model_fold_shap_imp_dfs = []

        for fold_num, (train_index, test_index) in enumerate(skf.split(X_full_sanitized, y_full_encoded)):
            current_fold_num = fold_num + 1
            print(f"\n--- {model_name} - Fold {current_fold_num}/{N_SPLITS_KFOLD} ---")
            # 중요: X_full_sanitized에서 iloc 사용
            X_train_fold = X_full_sanitized.iloc[train_index]
            X_test_fold = X_full_sanitized.iloc[test_index]
            y_train_fold = y_full_encoded.iloc[train_index]
            y_test_fold = y_full_encoded.iloc[test_index]

            y_train_series = pd.Series(y_train_fold); class_counts_train = y_train_series.value_counts()
            smote_k_neighbors = 5
            if not class_counts_train.empty and len(class_counts_train) > 1:
                min_samples_in_minority = class_counts_train.min()
                if min_samples_in_minority <= smote_k_neighbors : smote_k_neighbors = max(1, min_samples_in_minority - 1)
            else: smote_k_neighbors = 0
            
            X_train_fold_resampled, y_train_fold_resampled = X_train_fold.copy(), y_train_fold.copy()
            if smote_k_neighbors > 0:
                smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                print(f"SMOTE 적용 (Fold {current_fold_num}, k_neighbors={smote_k_neighbors})...")
                X_train_resampled_np, y_train_fold_resampled = smote.fit_resample(X_train_fold, y_train_fold)
                X_train_fold_resampled = pd.DataFrame(X_train_resampled_np, columns=X_train_fold.columns) # 올바른 인덱스 자동 생성
                print(f"SMOTE 후 학습 데이터: {X_train_fold_resampled.shape}, {y_train_fold_resampled.shape}")
            else: print("SMOTE 오버샘플링 건너뜀.")
            
            current_model = clone(model_prototype)
            try:
                if model_name == "XGBoost": current_model.fit(X_train_fold_resampled.values, y_train_fold_resampled)
                else: current_model.fit(X_train_fold_resampled, y_train_fold_resampled)
                
                y_pred = current_model.predict(X_test_fold.values if model_name == "XGBoost" else X_test_fold)
                y_pred_proba = None
                if hasattr(current_model, "predict_proba"):
                    y_pred_proba = current_model.predict_proba(X_test_fold.values if model_name == "XGBoost" else X_test_fold)
                
                metrics = evaluate_classification_model(model_name, y_test_fold, y_pred, y_pred_proba, le=le, fold_num=current_fold_num)
                fold_metrics_list.append(metrics)
                
                model_imp_df, shap_imp_df = get_and_print_feature_importances(current_model, model_name, X_test_fold.columns, 
                                                                           X_test_for_shap=X_test_fold, top_n=TOP_N_FEATURES_DISPLAY, 
                                                                           fold_num=current_fold_num, save_plots=True)
                if not model_imp_df.empty: current_model_fold_model_imp_dfs.append(model_imp_df.set_index('feature'))
                if not shap_imp_df.empty: current_model_fold_shap_imp_dfs.append(shap_imp_df.set_index('feature'))
            except Exception as e: print(f"{model_name} - Fold {current_fold_num} 처리 중 오류: {e}"); import traceback; traceback.print_exc()
        
        if fold_metrics_list:
            avg_metrics_df = pd.DataFrame(fold_metrics_list).mean(axis=0)
            print(f"\n===== {model_name} 교차 검증 평균 성능 ====="); print(avg_metrics_df)
            all_models_cv_metrics[model_name] = avg_metrics_df
        
        if current_model_fold_model_imp_dfs:
            try:
                concat_model_imp = pd.concat(current_model_fold_model_imp_dfs)
                aggregated_model_imp = concat_model_imp.groupby(concat_model_imp.index)['importance'].mean().sort_values(ascending=False)
                all_models_aggregated_model_importances[model_name] = aggregated_model_imp
                print(f"\n--- {model_name}: CV 평균 모델 고유 특징 중요도 (Top {TOP_N_FEATURES_DISPLAY}) ---"); print(aggregated_model_imp.head(TOP_N_FEATURES_DISPLAY))
                plt.figure(figsize=(10, max(5, TOP_N_FEATURES_DISPLAY / 2.5))); aggregated_model_imp.head(TOP_N_FEATURES_DISPLAY).sort_values(ascending=True).plot(kind='barh')
                plt.title(f"{model_name} - Avg Model Importance (Top {TOP_N_FEATURES_DISPLAY})"); plt.savefig(os.path.join(PLOTS_DIR, f"avg_model_importance_{model_name}.png"), bbox_inches='tight'); plt.close()
                print(f"'{model_name}'의 평균 모델 중요도 플롯 저장됨.")
            except Exception as e_agg_model_imp: print(f"모델 중요도 집계/플롯 중 오류 ({model_name}): {e_agg_model_imp}")

        if current_model_fold_shap_imp_dfs: # 변수명 오타 수정: current_model_fold_shap_imp_dfs
            try:
                concat_shap_imp = pd.concat(current_model_fold_shap_imp_dfs)
                aggregated_shap_imp = concat_shap_imp.groupby(concat_shap_imp.index)['shap_importance'].mean().sort_values(ascending=False)
                all_models_aggregated_shap_importances[model_name] = aggregated_shap_imp
                print(f"\n--- {model_name}: CV 평균 SHAP 특징 중요도 (Top {TOP_N_FEATURES_DISPLAY}) ---"); print(aggregated_shap_imp.head(TOP_N_FEATURES_DISPLAY))
                plt.figure(figsize=(10, max(5, TOP_N_FEATURES_DISPLAY / 2.5))); aggregated_shap_imp.head(TOP_N_FEATURES_DISPLAY).sort_values(ascending=True).plot(kind='barh')
                plt.title(f"{model_name} - Avg SHAP Importance (Top {TOP_N_FEATURES_DISPLAY})"); plt.savefig(os.path.join(PLOTS_DIR, f"avg_shap_importance_{model_name}.png"), bbox_inches='tight'); plt.close()
                print(f"'{model_name}'의 평균 SHAP 중요도 플롯 저장됨.")
            except Exception as e_agg_shap_imp: print(f"SHAP 중요도 집계/플롯 중 오류 ({model_name}): {e_agg_shap_imp}")
            
    print("\n\n===== 최종 모델별 교차 검증 평균 성능 요약 =====")
    if all_models_cv_metrics:
        final_summary_df = pd.DataFrame(all_models_cv_metrics).T; print(final_summary_df)
        try: final_summary_df.to_csv("cross_validation_summary_metrics.csv", index=True); print("CV 성능 요약 저장 완료.")
        except Exception as e_csv: print(f"요약 저장 중 오류: {e_csv}")
    else: print("CV 결과 없어 요약 생성 불가.")
    
    # 모델 성능 비교 시각화 호출
    if all_models_cv_metrics: # 데이터가 있을 때만 호출
        visualize_cv_model_performance(all_model_metrics_dict=all_models_cv_metrics)

     # PCA 컴포넌트 해석 호출 예시 (모든 모델의 상위 SHAP 특징 중 PCA 컴포넌트에 대해 시도)
    if all_models_aggregated_shap_importances: # SHAP 중요도 결과가 있다면
        print("\n\n<<<<< 모든 모델의 중요 PCA 특징에 대한 원본 특징 해석 시도 >>>>>")
        # SHAP 중요도 결과가 딕셔너리 형태 { '모델명': Series(feature, shap_importance) }로 저장되어 있다고 가정
        for model_name_to_interpret, top_features_series in all_models_aggregated_shap_importances.items():
            if top_features_series.empty:
                print(f"\n--- {model_name_to_interpret}: SHAP 중요도 정보가 없어 PCA 해석을 건너<0xEB>니다. ---")
                continue

            print(f"\n--- {model_name_to_interpret} 모델의 상위 SHAP 특징 중 PCA 컴포넌트 해석 ---")
            # 예시: 각 모델별 상위 3개의 SHAP 특징에 대해 PCA 해석 시도
            #      실제로는 몇 개를 볼지, 어떤 기준으로 볼지 조절 가능
            for feature_name in top_features_series.head(3).index: 
                if "_PCA_PC" in feature_name: # 특징 이름에 PCA 컴포넌트 표시가 있다면
                    print(f"  > {model_name_to_interpret} 모델에서 중요하게 나타난 PCA 특징: {feature_name}")
                    visualize_pca_component_loadings(feature_name, top_n=10, plots_dir=PLOTS_DIR, pca_models_dir=PCA_MODELS_DIR)
                else:
                    print(f"  > '{feature_name}' 은(는) PCA 컴포넌트가 아니므로 원본 특징 해석을 건너<0xEB>니다.")
    else:
        print("\n집계된 SHAP 중요도 정보가 없어 PCA 컴포넌트 해석을 자동으로 호출할 수 없습니다.")
    
    print("\n===== 모델링 파이프라인 (교차 검증 포함) 종료 =====")
    
if __name__ == '__main__':
    main_modeling()