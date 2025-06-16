# Tweet Sentiment Analysis Project

基于 PySpark 的推文情感与主题分析系统

## 项目结构

```
tweet_sentiment_analyze/
├── docker-compose.yml       # Docker编排文件
├── Dockerfile               # 自定义Docker镜像配置文件
├── data/
│   ├── raw/                 # 存放原始数据集
│   └── processed/           # 存放清洗和处理后的数据
├── notebooks/
│   ├── 1_data_ingestion_and_stats.ipynb  # 数据加载和基本统计
│   ├── 2_data_cleaning.ipynb             # 数据清洗
│   ├── 3_eda_and_sentiment_analysis.ipynb # EDA和情感分析
│   ├── 4_topic_modeling.ipynb            # 主题建模
│   └── 5_classification_modeling.ipynb   # 分类建模
├── src/                     # 存放可复用的Python代码
│   ├── __init__.py
│   └── utils.py
└── README.md                # 项目说明
```

## 快速开始

### 1. 启动环境

```bash
# 构建并启动Docker容器
docker-compose up --build -d
```

### 2. 访问 JupyterLab

打开浏览器访问：`http://localhost:8888`

### 3. 开始分析

按顺序运行 notebooks 目录下的文件：

1. **1_data_ingestion_and_stats.ipynb** - 数据加载和基本统计分析
2. **2_data_cleaning.ipynb** - 数据清洗
3. **3_eda_and_sentiment_analysis.ipynb** - 探索性分析和情感分析
4. **4_topic_modeling.ipynb** - 主题建模
5. **5_classification_modeling.ipynb** - 分类建模

## 技术栈

- **Apache Spark 3.x** - 分布式计算引擎
- **PySpark** - Spark的Python API
- **VADER** - 情感分析工具
- **scikit-learn** - 机器学习库
- **JupyterLab** - 交互式开发环境
- **Docker** - 容器化部署

## 分析流程

1. **数据加载** - 加载原始推文数据
2. **数据清洗** - 文本预处理和数据清洗
3. **探索性分析** - 数据可视化和趋势分析
4. **情感分析** - 使用VADER进行情感打分
5. **主题建模** - 使用LDA提取主题
6. **分类建模** - 训练情感分类模型

## 当前进度

✅ **已完成**：
1. **环境搭建** - Docker 容器和 Spark 环境
2. **数据加载** - `1_data_ingestion_and_stats.ipynb` 
3. **数据清洗** - `2_data_cleaning.ipynb`
4. **EDA准备** - `3_eda_and_sentiment_analysis.ipynb` (已创建)

🔄 **进行中**：
- 探索性数据分析和情感分析

⏭️ **下一步**：
1. 运行 `notebooks/2_data_cleaning.ipynb` 进行数据清洗
2. 运行 `notebooks/3_eda_and_sentiment_analysis.ipynb` 进行深度分析
3. 创建主题建模和分类建模的 notebooks

## 使用说明

1. **访问 JupyterLab**: http://localhost:8888
2. **按顺序运行 notebooks**：
   - `1_data_ingestion_and_stats.ipynb` ✅ (已完成基础分析)
   - `2_data_cleaning.ipynb` ⏭️ (下一步运行)
   - `3_eda_and_sentiment_analysis.ipynb` ⏭️ (清洗完成后运行) 