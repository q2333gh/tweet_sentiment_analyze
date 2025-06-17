"""
数据摄取和统计分析模块
对应notebook: 1_data_ingestion_and_stats.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from .config import DATA_PATHS
from .utils import create_spark_session, setup_logging, print_data_info, calculate_data_completeness

logger = setup_logging()

class DataIngestionAnalyzer:
    """数据摄取和统计分析器"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化分析器
        
        Args:
            data_path: 数据路径，默认使用原始数据路径
        """
        self.data_path = data_path or DATA_PATHS['raw_data']
        self.spark = create_spark_session("DataIngestion")
        self.df = None
        
    def load_data(self, use_sample: bool = False) -> bool:
        """
        加载数据
        
        Args:
            use_sample: 是否使用10%样本数据
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if use_sample:
                # 使用10%样本数据
                data_path = DATA_PATHS['ten_percent_sample']
                logger.info(f"加载10%样本数据: {data_path}")
                self.df = self.spark.read.parquet(data_path)
            else:
                # 使用原始CSV数据
                logger.info(f"加载原始CSV数据: {self.data_path}")
                self.df = self.spark.read.csv(
                    self.data_path, 
                    header=True, 
                    inferSchema=True, 
                    multiLine=True, 
                    escape='"'
                )
            
            # 缓存DataFrame
            self.df.cache()
            logger.info("数据加载完成并已缓存！")
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        基本统计分析
        
        Returns:
            dict: 统计结果
        """
        if self.df is None:
            logger.error("请先加载数据")
            return {}
        
        print("=== 基本统计信息 ===")
        
        # 1. 数据总量
        total_count = self.df.count()
        logger.info(f"总评论数量: {total_count:,}")
        
        # 2. 列信息
        num_columns = len(self.df.columns)
        logger.info(f"列数: {num_columns}")
        logger.info(f"列名: {self.df.columns}")
        
        # 3. 数据结构
        print("\n数据结构:")
        self.df.printSchema()
        
        # 4. 数据完整性
        completeness = calculate_data_completeness(self.df)
        print("\n数据完整性:")
        for col, pct in completeness.items():
            print(f"  {col}: {pct:.1f}%")
        
        return {
            'total_count': total_count,
            'num_columns': num_columns,
            'columns': self.df.columns,
            'completeness': completeness
        }
    
    def column_analysis(self) -> None:
        """各列详细分析"""
        if self.df is None:
            logger.error("请先加载数据")
            return
        
        print("\n=== 各列详细信息 ===")
        total_count = self.df.count()
        
        for col_name in self.df.columns:
            print(f"\n列名: {col_name}")
            print(f"数据类型: {dict(self.df.dtypes)[col_name]}")
            
            # 计算非空值数量
            if '.' in col_name:
                non_null_count = self.df.filter(F.col(f"`{col_name}`").isNotNull()).count()
            else:
                non_null_count = self.df.filter(F.col(col_name).isNotNull()).count()
            
            null_count = total_count - non_null_count
            print(f"非空值: {non_null_count:,} | 空值: {null_count:,} ({null_count/total_count*100:.2f}%)")
            
            # 显示样本值
            if col_name in ['body', 'permalink']:
                # 对于长文本，限制显示长度
                sample_values = self.df.select(col_name).limit(2).collect()
                for i, row in enumerate(sample_values):
                    value = str(row[0])[:100] + "..." if row[0] and len(str(row[0])) > 100 else row[0]
                    print(f"  样本{i+1}: {value}")
            else:
                sample_values = self.df.select(col_name).limit(3).collect()
                for i, row in enumerate(sample_values):
                    print(f"  样本{i+1}: {row[0]}")
            print("-" * 50)
    
    def time_analysis(self) -> Dict[str, Any]:
        """时间范围分析"""
        if self.df is None:
            logger.error("请先加载数据")
            return {}
        
        print("\n=== 时间范围分析 ===")
        
        # 转换时间戳为可读格式
        df_with_time = self.df.withColumn("timestamp", F.from_unixtime(F.col("created_utc")))
        
        # 获取时间范围
        time_stats = df_with_time.select(
            F.min("timestamp").alias("earliest_time"),
            F.max("timestamp").alias("latest_time")
        ).collect()[0]
        
        print(f"最早评论时间: {time_stats['earliest_time']}")
        print(f"最晚评论时间: {time_stats['latest_time']}")
        
        # 按年份统计评论数量
        yearly_stats = df_with_time.withColumn("year", F.year("timestamp")) \
                                  .groupBy("year") \
                                  .count() \
                                  .orderBy("year")
        
        print("\n按年份统计评论数量:")
        yearly_stats.show()
        
        # 转换为pandas进行可视化
        yearly_data = yearly_stats.toPandas()
        
        return {
            'earliest_time': time_stats['earliest_time'],
            'latest_time': time_stats['latest_time'],
            'yearly_data': yearly_data
        }
    
    def subreddit_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """子版块分析"""
        if self.df is None:
            logger.error("请先加载数据")
            return pd.DataFrame()
        
        print("\n=== 子版块分析 ===")
        
        # 统计各子版块的评论数量
        subreddit_stats = self.df.groupBy(F.col("`subreddit.name`")) \
                                .count() \
                                .orderBy(F.desc("count"))
        
        print(f"评论数量最多的前{top_n}个子版块:")
        subreddit_stats.show(top_n)
        
        # 转换为pandas进行可视化
        top_subreddits = subreddit_stats.limit(15).toPandas()
        
        return top_subreddits
    
    def sentiment_analysis(self) -> Dict[str, Any]:
        """情感分数分析"""
        if self.df is None:
            logger.error("请先加载数据")
            return {}
        
        print("\n=== 情感分数分析 ===")
        
        # 基本统计
        sentiment_stats = self.df.select("sentiment").describe()
        sentiment_stats.show()
        
        # 情感分数分布
        print("情感分数分布:")
        sentiment_ranges = self.df.withColumn("sentiment_range",
            F.when(F.col("sentiment") >= 0.1, "positive")
             .when(F.col("sentiment") <= -0.1, "negative")
             .otherwise("neutral")
        ).groupBy("sentiment_range").count().orderBy(F.desc("count"))
        
        sentiment_ranges.show()
        
        # 转换为pandas用于可视化
        sentiment_dist = sentiment_ranges.toPandas()
        
        return {
            'distribution': sentiment_dist,
            'stats': sentiment_stats.toPandas()
        }
    
    def text_length_analysis(self) -> Dict[str, Any]:
        """评论长度分析"""
        if self.df is None:
            logger.error("请先加载数据")
            return {}
        
        print("\n=== 评论长度分析 ===")
        
        # 计算评论长度
        df_with_length = self.df.withColumn("body_length", F.length(F.col("body")))
        
        # 长度统计
        length_stats = df_with_length.select("body_length").describe()
        length_stats.show()
        
        # 长度分布
        print("评论长度分布:")
        length_ranges = df_with_length.withColumn("length_range",
            F.when(F.col("body_length") <= 50, "very_short")
             .when(F.col("body_length") <= 200, "short")
             .when(F.col("body_length") <= 500, "medium")
             .when(F.col("body_length") <= 1000, "long")
             .otherwise("very_long")
        ).groupBy("length_range").count().orderBy(F.desc("count"))
        
        length_ranges.show()
        
        # 转换为pandas用于可视化
        length_dist = length_ranges.toPandas()
        
        return {
            'distribution': length_dist,
            'stats': length_stats.toPandas()
        }
    
    def create_visualizations(self, save_plots: bool = False) -> None:
        """创建可视化图表"""
        if self.df is None:
            logger.error("请先加载数据")
            return
        
        print("\n=== 生成可视化图表 ===")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 子版块分析图表
        top_subreddits = self.subreddit_analysis(15)
        if not top_subreddits.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_subreddits, x='count', y='subreddit.name')
            plt.title('评论数量最多的前15个子版块')
            plt.xlabel('评论数量')
            plt.ylabel('子版块名称')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('subreddit_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. 情感分析图表
        sentiment_data = self.sentiment_analysis()
        if sentiment_data and 'distribution' in sentiment_data:
            plt.figure(figsize=(10, 6))
            
            # 饼图
            plt.subplot(1, 2, 1)
            plt.pie(sentiment_data['distribution']['count'], 
                   labels=sentiment_data['distribution']['sentiment_range'],
                   autopct='%1.1f%%', startangle=90)
            plt.title('情感分布')
            
            # 柱状图
            plt.subplot(1, 2, 2)
            sns.barplot(data=sentiment_data['distribution'], 
                       x='sentiment_range', y='count')
            plt.title('各情感类别评论数量')
            plt.xlabel('情感类别')
            plt.ylabel('评论数量')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 文本长度分析图表
        length_data = self.text_length_analysis()
        if length_data and 'distribution' in length_data:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=length_data['distribution'], 
                       x='length_range', y='count')
            plt.title('评论长度分布')
            plt.xlabel('长度范围')
            plt.ylabel('评论数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('text_length_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_full_report(self, use_sample: bool = True, save_plots: bool = False) -> Dict[str, Any]:
        """生成完整的数据摄取报告"""
        logger.info("开始生成数据摄取报告...")
        
        # 加载数据
        if not self.load_data(use_sample=use_sample):
            logger.error("数据加载失败，无法生成报告")
            return {}
        
        # 执行各项分析
        basic_stats = self.basic_statistics()
        time_stats = self.time_analysis()
        subreddit_data = self.subreddit_analysis()
        sentiment_data = self.sentiment_analysis()
        length_data = self.text_length_analysis()
        
        # 生成可视化
        self.create_visualizations(save_plots=save_plots)
        
        # 详细列分析
        self.column_analysis()
        
        report = {
            'basic_statistics': basic_stats,
            'time_analysis': time_stats,
            'subreddit_analysis': subreddit_data,
            'sentiment_analysis': sentiment_data,
            'text_length_analysis': length_data,
            'data_source': 'sample' if use_sample else 'full'
        }
        
        logger.info("✅ 数据摄取报告生成完成！")
        return report
    
    def cleanup(self):
        """清理资源"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark Session已关闭")

def analyze_data_ingestion(use_sample: bool = True, save_plots: bool = False) -> Dict[str, Any]:
    """
    便捷函数：执行数据摄取分析
    
    Args:
        use_sample: 是否使用样本数据
        save_plots: 是否保存图表
        
    Returns:
        dict: 分析报告
    """
    analyzer = DataIngestionAnalyzer()
    
    try:
        report = analyzer.generate_full_report(use_sample=use_sample, save_plots=save_plots)
        return report
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    # 直接运行此模块时执行数据摄取分析
    logger.info("开始执行数据摄取分析...")
    report = analyze_data_ingestion(use_sample=True, save_plots=True)
    
    if report:
        print("\n=== 分析报告摘要 ===")
        basic_stats = report.get('basic_statistics', {})
        print(f"总记录数: {basic_stats.get('total_count', 0):,}")
        print(f"列数: {basic_stats.get('num_columns', 0)}")
        print(f"数据源: {report.get('data_source', 'unknown')}")
    else:
        logger.error("分析报告生成失败") 