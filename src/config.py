"""
配置文件 - 定义项目中使用的所有路径和参数
"""
import os

# 数据路径配置
DATA_PATHS = {
    'raw_data': '/home/jovyan/work/data/raw/the-reddit-climate-change-dataset-comments.csv',
    'processed_dir': '/home/jovyan/work/data/processed/',
    'ten_percent_sample': '/home/jovyan/work/data/processed/the-reddit-climate-change-dataset-comments-ten-percent.parquet',
    'cleaned_data': '/home/jovyan/work/data/processed/cleaned_comments.parquet',
    'sentiment_analyzed': '/home/jovyan/work/data/processed/sentiment_analyzed_comments.parquet',
    'topic_analyzed': '/home/jovyan/work/data/processed/topic_analyzed_comments.parquet'
}

# Spark配置
SPARK_CONFIG = {
    'app_name_prefix': 'TweetAnalysis',
    'master': 'local[*]',
    'driver_memory': '16g',
    'arrow_enabled': 'true'
}

# 数据处理参数
DATA_PROCESSING = {
    'sample_fraction': 0.1,
    'random_state': 42,
    'min_text_length': 10,
    'min_tokens': 5,
    'vocab_size': 3000,
    'min_doc_freq': 3.0
}

# 情感分析参数
SENTIMENT_CONFIG = {
    'positive_threshold': 0.05,
    'negative_threshold': -0.05,
    'sentiment_categories': ['Positive', 'Negative', 'Neutral']
}

# 主题建模参数
TOPIC_MODELING = {
    'num_topics': 5,
    'max_iterations': 10,
    'vocab_size': 2000,
    'min_doc_freq': 10.0,
    'sample_fraction': 0.3,
    'max_terms_per_topic': 15
}

# 分类模型参数
CLASSIFICATION = {
    'test_ratio': 0.2,
    'cross_validation_folds': 3,
    'max_depth': 10,
    'num_trees': 50
}

# 气候变化相关关键词
CLIMATE_KEYWORDS = [
    'climate', 'warming', 'carbon', 'emission', 'greenhouse', 'temperature',
    'fossil', 'renewable', 'energy', 'pollution', 'environment', 'sustainability',
    'weather', 'ice', 'sea', 'level', 'drought', 'flood'
]

# 创建输出目录
def ensure_output_dirs():
    """确保输出目录存在"""
    processed_dir = DATA_PATHS['processed_dir']
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir, exist_ok=True)
    return processed_dir 