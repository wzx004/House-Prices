import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载处理后的数据
processed_df = pd.read_csv('train_processed.csv')
print('数据加载完成！')
print(f'处理后数据形状: {processed_df.shape}')

# 1. 特征转换
print('\n进行特征转换...')

# 复制数据以避免修改原始数据
fe_df = processed_df.copy()

# 分离特征和目标变量
X = fe_df.drop('SalePrice', axis=1)
y = fe_df['SalePrice']

# 对类别特征进行编码
cat_cols = X.select_dtypes(include='object').columns
print(f'类别特征数量: {len(cat_cols)}')

# 使用LabelEncoder对类别特征进行编码
for col in cat_cols:
    le = LabelEncoder()
    fe_df[col] = le.fit_transform(fe_df[col])

# 2. 创建新特征
print('\n创建新特征...')

# 总浴室数量
fe_df['TotalBathrooms'] = fe_df['FullBath'] + fe_df['HalfBath'] + fe_df['BsmtFullBath'] + fe_df['BsmtHalfBath']

# 总平方英尺
fe_df['TotalSF'] = fe_df['GrLivArea'] + fe_df['TotalBsmtSF']

# 房屋年龄
fe_df['HouseAge'] = fe_df['YrSold'] - fe_df['YearBuilt']

# 装修年龄
fe_df['RemodAge'] = fe_df['YrSold'] - fe_df['YearRemodAdd']

# 每平方英尺价格
fe_df['PricePerSF'] = fe_df['SalePrice'] / fe_df['TotalSF']

# 3. 特征选择
print('\n进行特征选择...')

# 分离特征和目标变量
X = fe_df.drop('SalePrice', axis=1)
y = fe_df['SalePrice']

# 使用SelectKBest选择与目标变量相关性最高的特征
selector = SelectKBest(f_regression, k=30)
X_selected = selector.fit_transform(X, y)

# 获取选中的特征名称
selected_features = X.columns[selector.get_support()]
print(f'选中的特征数量: {len(selected_features)}')
print('选中的特征:')
print(selected_features)

# 可视化特征重要性
feature_scores = selector.scores_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})
feature_importance = feature_importance.sort_values('Score', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Score', y='Feature', data=feature_importance)
plt.title('特征重要性排序')
plt.tight_layout()
plt.savefig('feature_importance.png')
print('\n特征重要性可视化已保存为 feature_importance.png')

# 4. 保存处理后的数据
fe_df.to_csv('train_featured.csv', index=False)
print('\n特征工程后的数据已保存为 train_featured.csv')

# 保存选中的特征数据
selected_df = fe_df[list(selected_features) + ['SalePrice']]
selected_df.to_csv('train_selected.csv', index=False)
print('\n选中特征的数据已保存为 train_selected.csv')