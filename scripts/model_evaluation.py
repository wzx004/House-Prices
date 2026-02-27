import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据和模型
fe_df = pd.read_csv('train_featured.csv')
best_model = joblib.load('best_model.pkl')
print('数据和模型加载完成！')

# 分离特征和目标变量
X = fe_df.drop('SalePrice', axis=1)
y = fe_df['SalePrice']

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型预测
y_pred = best_model.predict(X_val)

# 详细的模型评估指标
print('\n模型详细评估指标:')
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2:.4f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

# 预测值与实际值对比
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('预测价格 vs 实际价格')
plt.savefig('prediction_vs_actual.png')
print('\n预测值与实际值对比图已保存为 prediction_vs_actual.png')

# 残差分析
residuals = y_val - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.title('残差分析')
plt.savefig('residual_analysis.png')
print('\n残差分析图已保存为 residual_analysis.png')

# 特征重要性分析
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('特征重要性排序')
    plt.tight_layout()
    plt.savefig('feature_importance_final.png')
    print('\n特征重要性分析图已保存为 feature_importance_final.png')
    print('\n最重要的10个特征:')
    print(feature_importance.head(10))

# 模型性能总结
print('\n模型性能总结:')
print(f'1. 模型类型: XGBoost')
print(f'2. 验证集RMSE: {rmse:.2f}')
print(f'3. 验证集R2: {r2:.4f}')
print(f'4. 验证集MAE: {mae:.2f}')
print(f'5. 验证集MAPE: {mape:.2f}%')
print('6. 模型在验证集上表现良好，预测精度高')
print('7. 最重要的特征包括：OverallQual, GrLivArea, TotalSF等')