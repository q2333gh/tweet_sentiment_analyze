"""
通用工具函数模块
"""
import re
import os
import logging
from typing import Optional, List, Dict, Any

# 尝试导入PySpark，如果失败则设置标志
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType, BooleanType
    PYSPARK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  PySpark导入失败: {e}")
    print("📝 将使用pandas作为后备方案")
    PYSPARK_AVAILABLE = False
    SparkSession = None
    udf = None
    StringType = None
    BooleanType = None

from .config import SPARK_CONFIG, CLIMATE_KEYWORDS

def create_spark_session(app_name: str) -> SparkSession:
    """
    创建Spark Session
    
    Args:
        app_name: 应用名称
        
    Returns:
        SparkSession对象
    """
    spark = SparkSession.builder \
        .appName(f"{SPARK_CONFIG['app_name_prefix']}_{app_name}") \
        .master(SPARK_CONFIG['master']) \
        .config("spark.driver.memory", SPARK_CONFIG['driver_memory']) \
        .config("spark.sql.execution.arrow.pyspark.enabled", SPARK_CONFIG['arrow_enabled']) \
        .getOrCreate()
    
    print(f"Spark Session created: {spark.version}")
    print(f"Available cores: {spark.sparkContext.defaultParallelism}")
    
    return spark

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_level: 日志级别
        
    Returns:
        Logger对象
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_text_cleaning_udf():
    """
    创建文本清洗的UDF函数
    
    Returns:
        PySpark UDF函数
    """
    def clean_text_func(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        
        text = str(text)
        
        # 1. 转换为小写
        text = text.lower()
        
        # 2. 去除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. 去除Reddit特有的格式
        text = re.sub(r'/u/\w+', '', text)  # 去除用户名
        text = re.sub(r'/r/\w+', '', text)  # 去除子版块名
        text = re.sub(r'&gt;', '', text)   # 去除引用符号
        text = re.sub(r'&lt;', '', text)
        text = re.sub(r'&amp;', 'and', text)
        
        # 4. 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 5. 去除特殊字符，保留字母、数字、空格和基本标点
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', ' ', text)
        
        # 6. 去除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 7. 过滤过短的文本
        if len(text) < 10:
            return None
            
        return text
    
    return udf(clean_text_func, StringType())

def create_climate_filter_udf():
    """
    创建气候变化关键词过滤的UDF函数
    
    Returns:
        PySpark UDF函数
    """
    def contains_climate_keywords(tokens: Optional[List[str]]) -> bool:
        if tokens is None:
            return False
        tokens_lower = [token.lower() for token in tokens]
        return any(keyword in tokens_lower for keyword in CLIMATE_KEYWORDS)
    
    return udf(contains_climate_keywords, BooleanType())

def print_data_info(df, title: str = "数据信息"):
    """
    打印数据框的基本信息
    
    Args:
        df: Spark DataFrame
        title: 标题
    """
    print(f"\n=== {title} ===")
    print(f"记录数: {df.count():,}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {df.columns}")
    print("\n数据结构:")
    df.printSchema()

def calculate_data_completeness(df) -> Dict[str, float]:
    """
    计算数据完整性
    
    Args:
        df: Spark DataFrame
        
    Returns:
        每列的完整性百分比字典
    """
    total_count = df.count()
    completeness = {}
    
    for col_name in df.columns:
        # 处理包含点号的列名
        if "." in col_name:
            null_count = df.filter(df[f"`{col_name}`"].isNull()).count()
        else:
            null_count = df.filter(df[col_name].isNull()).count()
        
        completeness[col_name] = ((total_count - null_count) / total_count * 100)
    
    return completeness

def safe_divide(a: float, b: float) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        a: 被除数
        b: 除数
        
    Returns:
        除法结果，如果除数为0则返回0
    """
    return a / b if b != 0 else 0.0

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值
        decimal_places: 小数位数
        
    Returns:
        格式化的百分比字符串
    """
    return f"{value:.{decimal_places}f}%"

def ensure_directory_exists(path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def get_topic_words_from_model(lda_model, vocabulary: List[str], max_terms: int = 10) -> List[Dict[str, Any]]:
    """
    从LDA模型中提取主题关键词
    
    Args:
        lda_model: 训练好的LDA模型
        vocabulary: 词汇表
        max_terms: 每个主题的最大词汇数
        
    Returns:
        主题关键词列表
    """
    topics = lda_model.describeTopics(maxTermsPerTopic=max_terms)
    topics_list = []
    
    for row in topics.collect():
        topic_id = row['topic']
        term_indices = row['termIndices']
        term_weights = row['termWeights']
        
        # 转换索引为词汇
        words = [vocabulary[idx] for idx in term_indices]
        
        topics_list.append({
            'topic_id': topic_id,
            'words': words,
            'weights': term_weights
        })
    
    return topics_list

def print_classification_metrics(y_true, y_pred, target_names: Optional[List[str]] = None):
    """
    打印分类模型的评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n=== 分类模型评估指标 ===")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return cm 