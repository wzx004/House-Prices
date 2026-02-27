import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载特征工程后的数据
fe_df = pd.read_csv('train_featured.csv')
print('数据加载完成！')

# 分离特征和目标变量
X = fe_df.drop('SalePrice', axis=1)
y = fe_df['SalePrice']

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}')

# 存储模型性能
model_performance = {}

# 1. 线性回归模型
print('\n训练线性回归模型...')
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

mse_lr = mean_squared_error(y_val, y_pred_lr)
r2_lr = r2_score(y_val, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

model_performance['Linear Regression'] = {'RMSE': rmse_lr, 'R2': r2_lr}
print(f'线性回归 RMSE: {rmse_lr:.2f}, R2: {r2_lr:.4f}')

# 2. 随机森林模型
print('\n训练随机森林模型...')
rf = RandomForestRegressor(random_state=42)

# 随机森林参数调优
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_val)

mse_rf = mean_squared_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

model_performance['Random Forest'] = {'RMSE': rmse_rf, 'R2': r2_rf}
print(f'随机森林 RMSE: {rmse_rf:.2f}, R2: {r2_rf:.4f}')
print(f'最佳参数: {grid_rf.best_params_}')

# 3. XGBoost模型
print('\n训练XGBoost模型...')
xgb_model = xgb.XGBRegressor(random_state=42)

# XGBoost参数调优
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_val)

mse_xgb = mean_squared_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

model_performance['XGBoost'] = {'RMSE': rmse_xgb, 'R2': r2_xgb}
print(f'XGBoost RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.4f}')
print(f'最佳参数: {grid_xgb.best_params_}')

# 可视化模型性能
print('\n模型性能比较:')
for model, metrics in model_performance.items():
    print(f'{model}: RMSE = {metrics["RMSE"]:.2f}, R2 = {metrics["R2"]:.4f}')

# 绘制模型性能对比图
plt.figure(figsize=(12, 6))
model_names = list(model_performance.keys())
rmse_values = [metrics['RMSE'] for metrics in model_performance.values()]
r2_values = [metrics['R2'] for metrics in model_performance.values()]

plt.subplot(1, 2, 1)
sns.barplot(x=model_names, y=rmse_values)
plt.title('模型RMSE对比')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
sns.barplot(x=model_names, y=r2_values)
plt.title('模型R2对比')
plt.ylabel('R2')

plt.tight_layout()
plt.savefig('model_performance.png')
print('\n模型性能对比图已保存为 model_performance.png')

# 保存最佳模型
best_model = best_xgb  # XGBoost通常表现最好
joblib.dump(best_model, 'best_model.pkl')
print('\n最佳模型已保存为 best_model.pkl')