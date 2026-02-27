import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 加载测试数据
test_df = pd.read_csv('test.csv')
print('测试数据加载完成！')
print(f'测试数据形状: {test_df.shape}')

# 加载训练数据用于获取编码映射
train_df = pd.read_csv('train_processed.csv')

# 1. 处理缺失值
print('\n处理测试数据缺失值...')

# 复制数据以避免修改原始数据
processed_test = test_df.copy()

# 分类特征缺失值处理
cat_cols = processed_test.select_dtypes(include='object').columns
for col in cat_cols:
    # 对于缺失值较多的特征，用'None'填充
    if processed_test[col].isnull().sum() > 500:
        processed_test[col] = processed_test[col].fillna('None')
    else:
        # 对于缺失值较少的特征，用最常见的值填充
        processed_test[col] = processed_test[col].fillna(train_df[col].mode()[0])

# 数值特征缺失值处理
num_cols = processed_test.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if processed_test[col].isnull().sum() > 0:
        # 用训练集的中位数填充
        processed_test[col] = processed_test[col].fillna(train_df[col].median())

# 2. 特征转换
print('\n进行特征转换...')

# 对类别特征进行编码
for col in cat_cols:
    le = LabelEncoder()
    # 使用训练集的数据进行拟合
    le.fit(train_df[col])
    # 处理测试集中可能出现的新类别
    processed_test[col] = processed_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    processed_test[col] = le.transform(processed_test[col])

# 3. 创建新特征
print('\n创建新特征...')

# 总浴室数量
processed_test['TotalBathrooms'] = processed_test['FullBath'] + processed_test['HalfBath'] + processed_test['BsmtFullBath'] + processed_test['BsmtHalfBath']

# 总平方英尺
processed_test['TotalSF'] = processed_test['GrLivArea'] + processed_test['TotalBsmtSF']

# 房屋年龄
processed_test['HouseAge'] = processed_test['YrSold'] - processed_test['YearBuilt']

# 装修年龄
processed_test['RemodAge'] = processed_test['YrSold'] - processed_test['YearRemodAdd']

# 注意：PricePerSF在测试集中无法计算，因为没有SalePrice，使用训练集的平均值
processed_test['PricePerSF'] = 180921.195890 / processed_test['TotalSF']

# 4. 加载最佳模型并进行预测
print('\n加载最佳模型并进行预测...')
best_model = joblib.load('best_model.pkl')

# 确保测试数据的特征与训练数据一致
X_test = processed_test[best_model.feature_names_in_]

# 进行预测
predictions = best_model.predict(X_test)
print(f'预测完成，预测数量: {len(predictions)}')

# 5. 生成提交文件
print('\n生成提交文件...')
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})

submission.to_csv('submission.csv', index=False)
print('提交文件已保存为 submission.csv')
print('\n提交文件前5行:')
print(submission.head())