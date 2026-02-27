import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载训练数据
train_df = pd.read_csv('train.csv')
print('数据加载完成！')

# 1. 处理缺失值
print('\n处理缺失值...')

# 复制数据以避免修改原始数据
processed_df = train_df.copy()

# 分类特征缺失值处理
cat_cols = processed_df.select_dtypes(include='object').columns
for col in cat_cols:
    # 对于缺失值较多的特征，用'None'填充
    if processed_df[col].isnull().sum() > 500:
        processed_df[col] = processed_df[col].fillna('None')
    else:
        # 对于缺失值较少的特征，用最常见的值填充
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

# 数值特征缺失值处理
num_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if processed_df[col].isnull().sum() > 0:
        # 用中位数填充数值型特征
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())

# 验证缺失值处理结果
print('\n处理后缺失值统计:')
missing_values = processed_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values if len(missing_values) > 0 else '无缺失值')

# 2. 检测和处理异常值
print('\n检测和处理异常值...')

# 查看与目标变量相关性高的特征的异常值
key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=processed_df[feature])
    plt.title(f'{feature} 箱线图')
plt.tight_layout()
plt.savefig('key_features_boxplot.png')
print('\n关键特征箱线图已保存为 key_features_boxplot.png')

# 检测GrLivArea的异常值
print('\n检测GrLivArea的异常值...')
Q1 = processed_df['GrLivArea'].quantile(0.25)
Q3 = processed_df['GrLivArea'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = processed_df[(processed_df['GrLivArea'] < lower_bound) | (processed_df['GrLivArea'] > upper_bound)]
print(f'GrLivArea 异常值数量: {len(outliers)}')
print('异常值详情:')
print(outliers[['GrLivArea', 'SalePrice']])

# 可视化GrLivArea与SalePrice的关系，标记异常值
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=processed_df)
plt.scatter(outliers['GrLivArea'], outliers['SalePrice'], color='red', label='异常值')
plt.title('GrLivArea 与 SalePrice 的关系')
plt.legend()
plt.savefig('grlivarea_saleprice.png')
print('\nGrLivArea与SalePrice关系图已保存为 grlivarea_saleprice.png')

# 移除明显的异常值（GrLivArea > 4000且SalePrice较低的情况）
processed_df = processed_df[~((processed_df['GrLivArea'] > 4000) & (processed_df['SalePrice'] < 300000))]
print(f'移除异常值后数据形状: {processed_df.shape}')

# 3. 保存处理后的数据
processed_df.to_csv('train_processed.csv', index=False)
print('\n处理后的数据已保存为 train_processed.csv')