import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练数据
train_df = pd.read_csv('train.csv')
print('数据加载完成！')
print(f'训练数据形状: {train_df.shape}')
print('\n数据前5行:')
print(train_df.head())
print('\n数据基本信息:')
train_df.info()
print('\n数值型特征统计描述:')
print(train_df.describe())
print('\n类别型特征统计:')
for col in train_df.select_dtypes(include='object').columns:
    print(f'\n{col}: {train_df[col].nunique()} 个唯一值')
    print(train_df[col].value_counts().head())

# 查看目标变量分布
print('\n目标变量 SalePrice 分布:')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('SalePrice 分布')
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['SalePrice'])
plt.title('SalePrice 箱线图')
plt.tight_layout()
plt.savefig('saleprice_distribution.png')
print('\n目标变量分布可视化已保存为 saleprice_distribution.png')

# 查看缺失值情况
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print('\n缺失值统计:')
print(missing_values.sort_values(ascending=False))

# 查看相关性矩阵（只对数值型特征）
print('\n计算相关性矩阵...')
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_df[numeric_cols].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('特征相关性矩阵')
plt.savefig('correlation_matrix.png')
print('\n相关性矩阵可视化已保存为 correlation_matrix.png')

# 查看与目标变量相关性最高的特征
top_corr = corr_matrix['SalePrice'].sort_values(ascending=False).head(10)
print('\n与 SalePrice 相关性最高的10个特征:')
print(top_corr)