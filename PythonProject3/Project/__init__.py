
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, classification_report)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap
import warnings
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
warnings.filterwarnings('ignore')


# 1. 数据加载与预处理

print("===== 1. 数据加载与预处理 =====")
df = pd.read_csv('heart.csv')  # 本地文件

# 目标变量二值化（原始0-4，其中1-4为患病）
df['target'] = (df['target'] > 0).astype(int)

print("原始数据形状:", df.shape)
print("目标分布:\n", df['target'].value_counts())
print("\n缺失值情况:\n", df.isnull().sum())

# 缺失值填充
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 异常值剔除（IQR法）
outlier_indices = set()
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].index
    outlier_indices.update(outliers)

print(f"剔除异常值前样本数: {len(df)}")
df_clean = df.drop(index=outlier_indices)
print(f"剔除异常值后样本数: {len(df_clean)}")


# 2. 特征工程
print("\n===== 2. 特征工程 =====")
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# 衍生特征
X['age_group'] = pd.cut(X['age'], bins=[0, 40, 55, 65, 100], labels=[1, 2, 3, 4]).astype(int)
X['bp_chol_product'] = X['trestbps'] * X['chol']
X['hr_reserve'] = 220 - X['age'] - X['thalach']
X['st_heart_ratio'] = X['oldpeak'] / (X['thalach'] + 1e-5)

# 连续特征列表
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                   'bp_chol_product', 'hr_reserve', 'st_heart_ratio']
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])


# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                  random_state=42, stratify=y_train)


# 4. 多阶段特征选择
print("\n===== 4. 特征选择 =====")
# 过滤式
selector_f = SelectKBest(f_classif, k=min(10, X_train.shape[1]))
X_train_f = selector_f.fit_transform(X_train, y_train)
selected_f = X_train.columns[selector_f.get_support()].tolist()
print("过滤式选择特征:", selected_f)

# 包裹式 (RFE)
lr = LogisticRegression(max_iter=1000)
n_features = min(8, X_train.shape[1])
selector_rfe = RFE(lr, n_features_to_select=n_features, step=1)
X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
selected_rfe = X_train.columns[selector_rfe.support_].tolist()
print("包裹式选择特征:", selected_rfe)

# 嵌入式：Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
selected_lasso = X_train.columns[(np.abs(lasso.coef_) > 0)].tolist()
print("Lasso选择特征:", selected_lasso)

# 嵌入式：XGBoost
xgb_tmp = XGBClassifier(n_estimators=100, random_state=42)
xgb_tmp.fit(X_train, y_train)
importance = pd.Series(xgb_tmp.feature_importances_, index=X_train.columns)
threshold = np.percentile(importance, 70)
selected_xgb = importance[importance > threshold].index.tolist()
print("XGBoost选择特征:", selected_xgb)

# 最终特征集（并集）
final_features = list(set(selected_rfe + selected_lasso + selected_xgb))
if not final_features:
    final_features = selected_f[:5]
print("最终核心特征:", final_features)

X_train_final = X_train[final_features]
X_val_final = X_val[final_features]
X_test_final = X_test[final_features]


# 5. 模型构建与调优
print("\n===== 5. 模型训练与调优 =====")
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'EBM': ExplainableBoostingClassifier(random_state=42, interactions=5),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'XGB': XGBClassifier(random_state=42)
}

param_grids = {
    'LR': {'C': [0.01, 0.1, 1, 10]},
    'EBM': {'learning_rate': [0.01, 0.05, 0.1]},
    'DT': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    'RF': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    'XGB': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
}

best_models = {}
for name, model in models.items():
    print(f"正在调优 {name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_final, y_train)
    best_models[name] = gs.best_estimator_
    print(f"{name} 最佳参数: {gs.best_params_}, AUC: {gs.best_score_:.4f}")


# 6. 验证集评估
val_scores = {}
for name, model in best_models.items():
    y_pred = model.predict(X_val_final)
    y_proba = model.predict_proba(X_val_final)[:, 1]
    val_scores[name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_proba)
    }
val_df = pd.DataFrame(val_scores).T
print("\n验证集各模型性能:\n", val_df.round(4))

# 7. 加权融合模型
weights = {name: val_scores[name]['auc'] for name in best_models}
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}
print("\n融合权重（基于验证集AUC）:", weights)


def weighted_ensemble_proba(X):
    probas = np.zeros((X.shape[0], 2))
    for name, model in best_models.items():
        probas += weights[name] * model.predict_proba(X)
    return probas[:, 1]


y_val_proba_ensemble = weighted_ensemble_proba(X_val_final)
y_val_pred_ensemble = (y_val_proba_ensemble >= 0.5).astype(int)
ensemble_val = {
    'accuracy': accuracy_score(y_val, y_val_pred_ensemble),
    'recall': recall_score(y_val, y_val_pred_ensemble),
    'f1': f1_score(y_val, y_val_pred_ensemble),
    'auc': roc_auc_score(y_val, y_val_proba_ensemble)
}
print("\n加权融合验证集性能:", ensemble_val)

# 8. 测试集最终评估
y_test_proba = weighted_ensemble_proba(X_test_final)
y_test_pred = (y_test_proba >= 0.5).astype(int)

test_metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1': f1_score(y_test, y_test_pred),
    'auc': roc_auc_score(y_test, y_test_proba)
}
print("\n最终融合模型在测试集上的性能:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# 单一模型测试集性能
test_scores = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]
    test_scores[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
test_scores['Ensemble'] = test_metrics
test_df = pd.DataFrame(test_scores).T
print("\n测试集各模型性能对比:\n", test_df.round(4))

# 9. 稳定性验证（5次重复实验）
print("\n===== 9. 稳定性验证 =====")


def run_experiment(seed):
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train_tmp, X_val_tmp, y_train_tmp, y_val_tmp = train_test_split(
        X_train_tmp, y_train_tmp, test_size=0.25, random_state=seed, stratify=y_train_tmp)

    X_train_f = X_train_tmp[final_features]
    X_val_f = X_val_tmp[final_features]
    X_test_f = X_test_tmp[final_features]

    # 重新训练基础模型（使用最佳参数）
    models_tmp = {}
    for name, model in best_models.items():
        new_model = model.__class__(**model.get_params())
        new_model.fit(X_train_f, y_train_tmp)
        models_tmp[name] = new_model

    # 重新计算验证集AUC权重
    val_auc = {}
    for name, model in models_tmp.items():
        y_proba = model.predict_proba(X_val_f)[:, 1]
        val_auc[name] = roc_auc_score(y_val_tmp, y_proba)
    total = sum(val_auc.values())
    w = {k: v / total for k, v in val_auc.items()}

    def weighted_proba(X):
        p = np.zeros((X.shape[0], 2))
        for name, model in models_tmp.items():
            p += w[name] * model.predict_proba(X)
        return p[:, 1]

    y_test_proba = weighted_proba(X_test_f)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_test_tmp, y_test_pred),
        'recall': recall_score(y_test_tmp, y_test_pred),
        'f1': f1_score(y_test_tmp, y_test_pred),
        'auc': roc_auc_score(y_test_tmp, y_test_proba)
    }


repeats = 5
results = []
for seed in range(42, 42 + repeats):
    res = run_experiment(seed)
    results.append(res)
results_df = pd.DataFrame(results)
print("\n5次重复实验的测试集性能（均值±标准差）:")
for metric in results_df.columns:
    mean_val = results_df[metric].mean()
    std_val = results_df[metric].std()
    print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

# 10. 可视化部分（论文用图）
print("\n===== 10. 生成可视化图表 =====")
# 创建保存图片的文件夹
import os

if not os.path.exists('./figures'):
    os.makedirs('./figures')

# 10.1 目标变量分布
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='Set2')
plt.title('目标变量分布')
plt.xlabel('是否有心脏病 (0=无, 1=有)')
plt.ylabel('样本数量')
plt.savefig('./figures/target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.2 异常值剔除前后样本量对比
fig, ax = plt.subplots(figsize=(6, 4))
before = len(df)
after = len(df_clean)
ax.bar(['剔除前', '剔除后'], [before, after], color=['lightcoral', 'lightgreen'])
ax.set_ylabel('样本量')
ax.set_title('异常值剔除前后样本量对比')
for i, v in enumerate([before, after]):
    ax.text(i, v + 3, str(v), ha='center')
plt.savefig('./figures/outlier_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.3 EBM特征重要性（主效应）
if 'EBM' in best_models:
    ebm = best_models['EBM']
    ebm_global = ebm.explain_global()
    data = ebm_global.data()
    if 'overall' in data:
        imp_data = data['overall']
    else:
        imp_data = data
    if 'names' in imp_data and 'scores' in imp_data:
        names = imp_data['names']
        scores = imp_data['scores']
    else:
        names = final_features
        scores = [0] * len(final_features)
    # 只取主效应（不含&）
    main_effects = [(n, s) for n, s in zip(names, scores) if '&' not in n]
    main_names = [n for n, s in main_effects]
    main_scores = [s for n, s in main_effects]
    sorted_idx = np.argsort(main_scores)
    main_names_sorted = [main_names[i] for i in sorted_idx]
    main_scores_sorted = [main_scores[i] for i in sorted_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(main_names_sorted, main_scores_sorted, color='steelblue')
    plt.xlabel('特征重要性')
    plt.title('EBM 模型特征重要性（主效应）')
    plt.tight_layout()
    plt.savefig('./figures/feature_importance_ebm.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10.4 验证集各模型性能对比
val_metrics = pd.DataFrame(val_scores).T
metrics_to_plot = ['accuracy', 'recall', 'f1', 'auc']
val_metrics[metrics_to_plot].plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('验证集各模型性能对比')
plt.ylabel('分数')
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')

ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('./figures/val_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.5 测试集各模型性能对比
test_metrics_df = test_df[metrics_to_plot]
test_metrics_df.plot(kind='bar', figsize=(10, 6), colormap='plasma')
plt.title('测试集各模型性能对比')
plt.ylabel('分数')
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('./figures/test_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.6 ROC曲线对比
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
for i, (name, model) in enumerate(best_models.items()):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_final)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
# 融合模型
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_test_proba)
auc_ens = roc_auc_score(y_test, y_test_proba)
plt.plot(fpr_ens, tpr_ens, color='black', lw=2, linestyle='--', label=f'Ensemble (AUC = {auc_ens:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线对比')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('./figures/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.7 融合模型混淆矩阵热力图
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['无病', '有病'], yticklabels=['无病', '有病'])
plt.title('融合模型混淆矩阵（测试集）')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('./figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.8 稳定性箱线图
results_df.boxplot()
plt.title('5次重复实验性能稳定性')
plt.ylabel('分数')
plt.ylim(0.6, 1.0)
plt.grid(axis='y', alpha=0.3)
plt.savefig('./figures/stability_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()


# 11. 特征选择前后对比
print("\n===== 11. 特征选择前后性能对比 =====")
# 使用全部特征重新训练
X_train_all = X_train
X_test_all = X_test

lr_all = LogisticRegression(C=best_models['LR'].C, max_iter=1000, random_state=42)
xgb_all = XGBClassifier(learning_rate=best_models['XGB'].learning_rate,
                        max_depth=best_models['XGB'].max_depth,
                        n_estimators=best_models['XGB'].n_estimators,
                        random_state=42)
lr_all.fit(X_train_all, y_train)
xgb_all.fit(X_train_all, y_train)

lr_all_proba = lr_all.predict_proba(X_test_all)[:, 1]
xgb_all_proba = xgb_all.predict_proba(X_test_all)[:, 1]
lr_all_auc = roc_auc_score(y_test, lr_all_proba)
xgb_all_auc = roc_auc_score(y_test, xgb_all_proba)

# 使用筛选后特征
lr_sel = LogisticRegression(C=best_models['LR'].C, max_iter=1000, random_state=42)
xgb_sel = XGBClassifier(learning_rate=best_models['XGB'].learning_rate,
                        max_depth=best_models['XGB'].max_depth,
                        n_estimators=best_models['XGB'].n_estimators,
                        random_state=42)
lr_sel.fit(X_train_final, y_train)
xgb_sel.fit(X_train_final, y_train)

lr_sel_proba = lr_sel.predict_proba(X_test_final)[:, 1]
xgb_sel_proba = xgb_sel.predict_proba(X_test_final)[:, 1]
lr_sel_auc = roc_auc_score(y_test, lr_sel_proba)
xgb_sel_auc = roc_auc_score(y_test, xgb_sel_proba)

comparison_df = pd.DataFrame({
    '模型': ['逻辑回归', 'XGBoost'],
    '特征选择前(AUC)': [lr_all_auc, xgb_all_auc],
    '特征选择后(AUC)': [lr_sel_auc, xgb_sel_auc],
    '提升幅度': [lr_sel_auc - lr_all_auc, xgb_sel_auc - xgb_all_auc]
})
print(comparison_df.round(4))

# 12. SHAP 可解释性分析
print("\n===== 12. SHAP 可解释性分析 =====")
# 选择 XGBoost 模型
explainer = shap.TreeExplainer(best_models['XGB'])
shap_values = explainer.shap_values(X_test_final)

# 全局特征重要性（条形图）
shap.summary_plot(shap_values, X_test_final, plot_type="bar", show=False)
plt.title('SHAP 全局特征重要性')
plt.tight_layout()
plt.savefig('./figures/shap_global_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 蜂群图
shap.summary_plot(shap_values, X_test_final, show=False)
plt.title('SHAP 特征贡献分布')
plt.tight_layout()
plt.savefig('./figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 单个样本解释（第一个测试样本）
sample_idx = 0
shap.waterfall_plot(shap.Explanation(values=shap_values[sample_idx],
                                     base_values=explainer.expected_value,
                                     data=X_test_final.iloc[sample_idx],
                                     feature_names=X_test_final.columns.tolist()),
                    show=False)
plt.title(f'样本 {sample_idx} 的 SHAP 解释')
plt.tight_layout()
plt.savefig('./figures/shap_waterfall_sample0.png', dpi=300, bbox_inches='tight')
plt.show()
#LIME
model_to_explain = best_models['XGB']

# 创建 LIME 解释器（使用训练集数据）
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_final.values,
    feature_names=X_train_final.columns.tolist(),
    class_names=['无病', '有病'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# 选择测试集中第一个样本进行解释
sample_idx = 0
exp = explainer_lime.explain_instance(
    data_row=X_test_final.values[sample_idx],
    predict_fn=model_to_explain.predict_proba,
    num_features=len(X_test_final.columns)
)
fig = exp.as_pyplot_figure()
plt.title(f'样本 {sample_idx} 的 LIME 解释')
plt.tight_layout()
plt.savefig('./figures/lime_explanation_sample0.png', dpi=300, bbox_inches='tight')
plt.show()


# 13. 分类报告
print("\n===== 13. 融合模型分类报告 =====")
report = classification_report(y_test, y_test_pred, target_names=['无病', '有病'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(4))
report_df.to_csv('./figures/classification_report.csv')
print("\n所有图表和报告已保存至 ./figures 目录")