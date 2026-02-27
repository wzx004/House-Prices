# House Prices 项目说明文档

## 项目概述

本项目是Kaggle平台上的House Prices - Advanced Regression Techniques竞赛的解决方案，旨在预测美国艾姆斯市房屋的销售价格。通过完整的数据分析流程，从数据探索、预处理、特征工程到模型构建和评估，成功构建了一个高精度的房屋价格预测模型。

## 目录结构

```
House Prices/
├── data/             # 数据文件
├── docs/             # 文档文件
├── models/           # 模型文件
├── results/          # 结果文件
├── scripts/          # 脚本文件
├── visualizations/   # 可视化文件
└── README.md         # 项目说明文档
```

## 目录详细说明

### 1. data/ - 数据文件

| 文件名 | 描述 |
|-------|------|
| train.csv | 原始训练数据，包含1460个样本和80个特征 |
| test.csv | 原始测试数据，包含1459个样本和80个特征 |
| sample_submission.csv | 提交示例文件，展示提交格式 |
| train_processed.csv | 处理后的数据，已处理缺失值和异常值 |
| train_featured.csv | 特征工程后的数据，包含新创建的特征 |
| train_selected.csv | 选中特征的数据，只包含与目标变量相关性高的特征 |
| data_description.txt | 数据描述文件，详细说明每个特征的含义 |

### 2. docs/ - 文档文件

| 文件名 | 描述 |
|-------|------|
| project_report.md | 项目报告，包含问题定义、方法论、关键发现与结论 |
| comprehensive_analysis.md | 综合分析报告，详细介绍EDA、数据清洗、特征工程、可视化和模型评估 |

### 3. models/ - 模型文件

| 文件名 | 描述 |
|-------|------|
| best_model.pkl | 最佳模型文件，使用XGBoost算法训练并调优 |

### 4. results/ - 结果文件

| 文件名 | 描述 |
|-------|------|
| submission.csv | 最终预测结果，用于Kaggle竞赛提交 |

### 5. scripts/ - 脚本文件

| 文件名 | 描述 |
|-------|------|
| data_exploration.py | 数据探索脚本，分析数据结构、基本统计信息和特征相关性 |
| data_preprocessing.py | 数据预处理脚本，处理缺失值和异常值 |
| feature_engineering.py | 特征工程脚本，进行特征转换、新特征创建和特征选择 |
| model_training.py | 模型训练脚本，构建和调优多种机器学习模型 |
| model_evaluation.py | 模型评估脚本，分析模型性能并进行模型解释 |
| predict.py | 预测脚本，使用最佳模型对测试数据进行预测 |

### 6. visualizations/ - 可视化文件

| 文件名 | 描述 |
|-------|------|
| saleprice_distribution.png | 目标变量SalePrice的分布图 |
| correlation_matrix.png | 特征相关性热图 |
| key_features_boxplot.png | 关键特征的箱线图 |
| grlivarea_saleprice.png | GrLivArea与SalePrice的关系图 |
| feature_importance.png | 特征重要性排序图 |
| feature_importance_final.png | 最终模型的特征重要性排序图 |
| model_performance.png | 不同模型的性能对比图 |
| prediction_vs_actual.png | 预测值与实际值的对比图 |
| residual_analysis.png | 模型残差分析图 |

## 项目流程

1. **数据探索**：使用data_exploration.py分析数据结构和基本统计信息
2. **数据预处理**：使用data_preprocessing.py处理缺失值和异常值
3. **特征工程**：使用feature_engineering.py进行特征转换和新特征创建
4. **模型训练**：使用model_training.py构建和调优多种机器学习模型
5. **模型评估**：使用model_evaluation.py分析模型性能并进行模型解释
6. **预测**：使用predict.py对测试数据进行预测，生成提交文件

## 技术栈

- **数据处理**：pandas, numpy
- **可视化**：matplotlib, seaborn
- **机器学习**：scikit-learn, xgboost
- **模型保存**：joblib

## 模型性能

| 模型 | RMSE | R2 | MAE | MAPE |
|------|------|----|-----|------|
| 线性回归 | 15970.38 | 0.9538 | - | - |
| 随机森林 | 9478.95 | 0.9837 | - | - |
| XGBoost | 6499.08 | 0.9924 | 3636.27 | 2.53% |

## 关键发现

1. **模型性能**：XGBoost模型表现最佳，R2值达到0.9924，预测精度高
2. **关键因素**：房屋整体质量（OverallQual）是影响房价的最重要因素，其次是房屋面积（TotalSF）和每平方英尺价格（PricePerSF）
3. **数据处理**：适当的缺失值处理和异常值移除对模型性能有显著影响
4. **特征工程**：创建新特征能够有效提高模型预测能力

## 如何运行

1. **数据探索**：`python scripts/data_exploration.py`
2. **数据预处理**：`python scripts/data_preprocessing.py`
3. **特征工程**：`python scripts/feature_engineering.py`
4. **模型训练**：`python scripts/model_training.py`
5. **模型评估**：`python scripts/model_evaluation.py`
6. **预测**：`python scripts/predict.py`

## 结论

本项目通过完整的数据分析流程，成功构建了一个高精度的房屋价格预测模型。XGBoost模型在验证集上表现优异，预测精度达到了99.24%的R2值，为房地产价格预测提供了可靠的工具。项目展示了完整的数据分析能力，包括数据处理、特征工程、模型选择和评估等多个方面，符合专业数据分析的标准。