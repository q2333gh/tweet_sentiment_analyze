# Tweet情感分析Python模块

本目录包含了从Jupyter notebooks转换而来的Python模块，实现了完整的Tweet情感分析流水线。

## 模块结构

```
src/
├── __init__.py              # 包初始化文件
├── config.py                # 配置文件
├── utils.py                 # 通用工具函数
├── data_extraction.py       # 数据提取模块 (对应 0_extract_ten_percent.ipynb)
├── data_ingestion.py        # 数据摄取和统计分析 (对应 1_data_ingestion_and_stats.ipynb)
├── data_cleaning.py         # 数据清洗模块 (对应 2_data_cleaning.ipynb)
├── sentiment_analysis.py    # 情感分析模块 (对应 3_eda_and_sentiment_analysis.ipynb)
├── topic_modeling.py        # 主题建模模块 (对应 4_topic_modeling_lda.ipynb)
├── classification.py        # 分类建模模块 (对应 5_classification_modeling.ipynb)
├── main.py                  # 主运行脚本
└── README.md               # 本文档
```

## 功能特性

### 🔧 核心功能
- **数据提取**: 从原始数据中提取10%样本
- **数据摄取**: 基本统计分析和数据探索
- **数据清洗**: 文本清洗、去重、分词、停用词处理
- **情感分析**: 使用VADER进行情感分析和分类
- **主题建模**: 使用LDA进行主题发现
- **分类建模**: 朴素贝叶斯和随机森林分类器

### 🚀 技术栈
- **大数据处理**: PySpark
- **机器学习**: PySpark MLlib + scikit-learn
- **情感分析**: VADER Sentiment
- **可视化**: matplotlib + seaborn + wordcloud
- **数据存储**: Parquet格式

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整流水线

```python
from src.main import TweetSentimentAnalysisPipeline

# 创建流水线实例
pipeline = TweetSentimentAnalysisPipeline()

# 运行完整流水线
final_report = pipeline.run_full_pipeline()
```

### 2. 运行单个模块

#### 数据提取
```python
from src.data_extraction import extract_ten_percent_sample

# 提取10%样本数据
success = extract_ten_percent_sample()
```

#### 数据摄取分析
```python
from src.data_ingestion import analyze_data_ingestion

# 执行数据摄取分析
report = analyze_data_ingestion(use_sample=True, save_plots=True)
```

#### 数据清洗
```python
from src.data_cleaning import clean_data

# 执行数据清洗
report = clean_data()
```

#### 情感分析
```python
from src.sentiment_analysis import analyze_sentiment

# 执行情感分析
report = analyze_sentiment()
```

#### 主题建模
```python
from src.topic_modeling import perform_topic_modeling

# 执行主题建模
report = perform_topic_modeling()
```

#### 分类建模
```python
from src.classification import perform_classification

# 执行分类建模
report = perform_classification()
```

### 3. 命令行使用

```bash
# 运行完整流水线
python -m src.main --step all

# 运行单个步骤
python -m src.main --step extraction
python -m src.main --step ingestion
python -m src.main --step cleaning
python -m src.main --step sentiment
python -m src.main --step topic
python -m src.main --step classification

# 跳过某些步骤
python -m src.main --step all --skip data_extraction data_ingestion
```

## 配置说明

主要配置在 `config.py` 中定义：

```python
# 数据路径配置
DATA_PATHS = {
    'raw_data': '/path/to/raw/data.csv',
    'processed_dir': '/path/to/processed/',
    # ...
}

# 处理参数
DATA_PROCESSING = {
    'sample_fraction': 0.1,
    'random_state': 42,
    'min_text_length': 10,
    # ...
}

# 情感分析参数
SENTIMENT_CONFIG = {
    'positive_threshold': 0.05,
    'negative_threshold': -0.05,
    # ...
}

# 主题建模参数
TOPIC_MODELING = {
    'num_topics': 5,
    'max_iterations': 10,
    # ...
}
```

## 输出文件

流水线执行后会生成以下文件：

```
data/processed/
├── the-reddit-climate-change-dataset-comments-ten-percent.parquet  # 10%样本数据
├── cleaned_comments.parquet                                        # 清洗后数据
├── sentiment_analyzed_comments.parquet                             # 情感分析结果
└── topic_analyzed_comments.parquet                                # 主题分析结果
```

## 可视化输出

各模块会生成以下可视化图表：

- **数据摄取**: 子版块分析图、情感分布图、文本长度分布图
- **情感分析**: 情感分布饼图、子版块情感分析图、词频分析图、词云图
- **主题建模**: 主题关键词图、主题分布图、主题词云图、主题-情感热力图
- **分类建模**: 模型性能对比图、混淆矩阵

## 错误处理

所有模块都包含完善的错误处理机制：

- 自动检测数据文件是否存在
- 提供备选数据源
- 详细的错误日志记录
- 优雅的资源清理

## 性能优化

- 使用PySpark进行分布式计算
- 数据缓存策略
- 采样策略减少计算负担
- Parquet格式提高I/O性能

## 扩展性

模块设计具有良好的扩展性：

- 可以轻松添加新的分析模块
- 支持自定义配置参数
- 模块间松耦合设计
- 支持不同的数据源格式

## 注意事项

1. **内存要求**: 建议至少16GB内存用于Spark计算
2. **数据路径**: 请根据实际情况修改 `config.py` 中的路径配置
3. **依赖安装**: 确保所有依赖库正确安装
4. **Java环境**: PySpark需要Java 8+环境

## 故障排除

### 常见问题

1. **Spark Session创建失败**
   - 检查Java环境是否正确安装
   - 检查内存配置是否充足

2. **数据文件未找到**
   - 检查 `config.py` 中的路径配置
   - 确保数据文件存在且有读取权限

3. **内存不足错误**
   - 减少 `sample_fraction` 参数
   - 调整Spark内存配置

4. **依赖库导入错误**
   - 重新安装 requirements.txt 中的依赖
   - 检查Python环境是否正确

## 许可证

本项目遵循MIT许可证。 