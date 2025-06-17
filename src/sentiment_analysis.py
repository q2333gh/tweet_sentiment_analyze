"""
情感分析模块
对应notebook: 3_eda_and_sentiment_analysis.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Any, List
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, DoubleType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from .config import DATA_PATHS, SENTIMENT_CONFIG
from .utils import create_spark_session, setup_logging, print_data_info

logger = setup_logging()

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, input_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        初始化情感分析器
        
        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
        """
        self.input_path = input_path or DATA_PATHS['cleaned_data']
        self.output_path = output_path or DATA_PATHS['sentiment_analyzed']
        self.spark = create_spark_session("SentimentAnalysis")
        self.df = None
        self.analyzer = SentimentIntensityAnalyzer()
        
    def load_data(self) -> bool:
        """
        加载清洗后的数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info(f"加载清洗后的数据: {self.input_path}")
            self.df = self.spark.read.parquet(self.input_path)
            self.df.cache()
            
            record_count = self.df.count()
            logger.info(f"✅ 数据加载完成，共 {record_count:,} 条记录")
            
            print("\n数据结构:")
            self.df.printSchema()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def basic_data_overview(self) -> Dict[str, Any]:
        """
        基础数据概览
        
        Returns:
            dict: 数据概览信息
        """
        if self.df is None:
            logger.error("请先加载数据")
            return {}
        
        print("=== 基础数据概览 ===")
        
        # 1. 基本信息
        total_count = self.df.count()
        num_columns = len(self.df.columns)
        
        print(f"总记录数: {total_count:,}")
        print(f"总列数: {num_columns}")
        
        # 2. 数据完整性检查
        print("\n数据完整性:")
        for col_name in self.df.columns:
            if "." in col_name:
                null_count = self.df.filter(F.col(f"`{col_name}`").isNull()).count()
            else:
                null_count = self.df.filter(F.col(col_name).isNull()).count()
            
            completeness = ((total_count - null_count) / total_count * 100)
            print(f"  {col_name}: {completeness:.1f}% 完整")
        
        # 3. 原始情感分数统计
        if 'sentiment' in self.df.columns:
            print("\n原始情感分数分布:")
            sentiment_stats = self.df.select("sentiment").describe()
            sentiment_stats.show()
        
        # 4. 子版块分布
        if '`subreddit.name`' in self.df.columns or 'subreddit.name' in self.df.columns:
            print("\n子版块分布 (Top 10):")
            subreddit_col = "`subreddit.name`" if "`subreddit.name`" in self.df.columns else "subreddit.name"
            subreddit_dist = self.df.groupBy(subreddit_col).count().orderBy(F.desc("count")).limit(10)
            subreddit_dist.show(truncate=False)
        
        return {
            'total_count': total_count,
            'num_columns': num_columns,
            'columns': self.df.columns
        }
    
    def apply_vader_sentiment(self) -> bool:
        """
        使用VADER进行情感分析
        
        Returns:
            bool: 分析是否成功
        """
        if self.df is None:
            logger.error("请先加载数据")
            return False
        
        logger.info("开始使用VADER进行情感分析...")
        
        def analyze_sentiment_vader(text: Optional[str]) -> Optional[float]:
            """使用VADER进行情感分析"""
            if text is None:
                return None
            scores = self.analyzer.polarity_scores(str(text))
            return scores['compound']  # 返回复合情感分数 (-1 到 1)
        
        # 创建UDF
        sentiment_udf = F.udf(analyze_sentiment_vader, DoubleType())
        
        # 应用VADER情感分析
        self.df = self.df.withColumn("vader_sentiment", sentiment_udf(F.col("cleaned_body")))
        
        # 缓存结果
        self.df.cache()
        
        logger.info("✅ VADER情感分析完成！")
        
        # 比较原始情感分数和VADER分数
        if 'sentiment' in self.df.columns:
            print("\n原始情感分数 vs VADER分数对比:")
            comparison = self.df.select("sentiment", "vader_sentiment").filter(
                F.col("sentiment").isNotNull() & F.col("vader_sentiment").isNotNull()
            ).limit(10)
            comparison.show()
        
        # VADER情感分数统计
        print("\nVADER情感分数统计:")
        vader_stats = self.df.select("vader_sentiment").describe()
        vader_stats.show()
        
        return True
    
    def categorize_sentiment(self) -> bool:
        """
        情感分类
        
        Returns:
            bool: 分类是否成功
        """
        if self.df is None or 'vader_sentiment' not in self.df.columns:
            logger.error("请先执行VADER情感分析")
            return False
        
        logger.info("开始情感分类...")
        
        def categorize_sentiment_func(score: Optional[float]) -> Optional[str]:
            if score is None:
                return "Unknown"
            elif score > SENTIMENT_CONFIG['positive_threshold']:
                return "Positive"
            elif score < SENTIMENT_CONFIG['negative_threshold']:
                return "Negative"
            else:
                return "Neutral"
        
        categorize_udf = F.udf(categorize_sentiment_func, StringType())
        
        # 应用情感分类
        self.df = self.df.withColumn("sentiment_category", categorize_udf(F.col("vader_sentiment")))
        
        logger.info("✅ 情感分类完成！")
        return True
    
    def analyze_sentiment_distribution(self) -> Dict[str, Any]:
        """
        分析情感分布
        
        Returns:
            dict: 情感分布分析结果
        """
        if self.df is None or 'sentiment_category' not in self.df.columns:
            logger.error("请先执行情感分类")
            return {}
        
        print("=== 情感分布分析 ===")
        
        # 情感分布统计
        sentiment_dist = self.df.groupBy("sentiment_category").count().toPandas()
        print("\n情感分布:")
        print(sentiment_dist)
        
        # 计算百分比
        total_count = sentiment_dist['count'].sum()
        sentiment_dist['percentage'] = (sentiment_dist['count'] / total_count * 100).round(2)
        
        return {
            'distribution': sentiment_dist,
            'total_count': total_count
        }
    
    def analyze_subreddit_sentiment(self) -> Dict[str, Any]:
        """
        子版块情感分析
        
        Returns:
            dict: 子版块情感分析结果
        """
        if self.df is None or 'sentiment_category' not in self.df.columns:
            logger.error("请先执行情感分类")
            return {}
        
        print("=== 子版块情感分析 ===")
        
        # 确定子版块列名
        subreddit_col = None
        if "`subreddit.name`" in self.df.columns:
            subreddit_col = "`subreddit.name`"
        elif "subreddit.name" in self.df.columns:
            subreddit_col = "subreddit.name"
        else:
            logger.warning("未找到子版块列")
            return {}
        
        # 计算各子版块的情感统计
        subreddit_sentiment = self.df.groupBy(subreddit_col).agg(
            F.count("*").alias("comment_count"),
            F.avg("vader_sentiment").alias("avg_sentiment"),
            F.sum(F.when(F.col("sentiment_category") == "Positive", 1).otherwise(0)).alias("positive_count"),
            F.sum(F.when(F.col("sentiment_category") == "Negative", 1).otherwise(0)).alias("negative_count"),
            F.sum(F.when(F.col("sentiment_category") == "Neutral", 1).otherwise(0)).alias("neutral_count")
        ).filter(F.col("comment_count") >= 100).orderBy(F.desc("comment_count"))  # 至少100条评论
        
        # 转换为Pandas
        subreddit_sentiment_pd = subreddit_sentiment.toPandas()
        
        print("主要子版块情感分析结果:")
        print(subreddit_sentiment_pd.head(10))
        
        return {
            'subreddit_sentiment': subreddit_sentiment_pd
        }
    
    def analyze_temporal_sentiment(self) -> Dict[str, Any]:
        """
        时间序列情感分析
        
        Returns:
            dict: 时间序列情感分析结果
        """
        if self.df is None or 'vader_sentiment' not in self.df.columns:
            logger.error("请先执行VADER情感分析")
            return {}
        
        print("=== 时间序列情感分析 ===")
        
        if 'timestamp' not in self.df.columns:
            logger.warning("未找到timestamp列，跳过时间序列分析")
            return {}
        
        # 按年月统计情感
        temporal_sentiment = self.df.withColumn("year_month", F.date_format("timestamp", "yyyy-MM")) \
                                   .groupBy("year_month") \
                                   .agg(
                                       F.count("*").alias("comment_count"),
                                       F.avg("vader_sentiment").alias("avg_sentiment"),
                                       F.sum(F.when(F.col("sentiment_category") == "Positive", 1).otherwise(0)).alias("positive_count"),
                                       F.sum(F.when(F.col("sentiment_category") == "Negative", 1).otherwise(0)).alias("negative_count")
                                   ).orderBy("year_month")
        
        temporal_data = temporal_sentiment.toPandas()
        
        print("时间序列情感趋势:")
        print(temporal_data.head(10))
        
        return {
            'temporal_sentiment': temporal_data
        }
    
    def generate_word_frequency_analysis(self, top_n: int = 50) -> Dict[str, Any]:
        """
        词频分析
        
        Args:
            top_n: 返回前N个高频词
            
        Returns:
            dict: 词频分析结果
        """
        if self.df is None or 'tokens_cleaned' not in self.df.columns:
            logger.error("请先加载包含分词结果的数据")
            return {}
        
        print("=== 词频分析 ===")
        
        # 展开tokens并统计词频
        word_freq = self.df.select(F.explode("tokens_cleaned").alias("word")) \
                          .groupBy("word") \
                          .count() \
                          .orderBy(F.desc("count")) \
                          .limit(top_n)
        
        word_freq_pd = word_freq.toPandas()
        
        print(f"前{top_n}个高频词:")
        print(word_freq_pd.head(20))
        
        return {
            'word_frequency': word_freq_pd
        }
    
    def create_visualizations(self, save_plots: bool = False) -> None:
        """
        创建可视化图表
        
        Args:
            save_plots: 是否保存图表
        """
        if self.df is None:
            logger.error("请先加载数据")
            return
        
        print("\n=== 生成可视化图表 ===")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 情感分布图
        sentiment_data = self.analyze_sentiment_distribution()
        if sentiment_data:
            plt.figure(figsize=(12, 5))
            
            # 饼图
            plt.subplot(1, 2, 1)
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            plt.pie(sentiment_data['distribution']['count'], 
                   labels=sentiment_data['distribution']['sentiment_category'],
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('气候变化评论情感分布')
            
            # 柱状图
            plt.subplot(1, 2, 2)
            bars = plt.bar(sentiment_data['distribution']['sentiment_category'], 
                          sentiment_data['distribution']['count'], color=colors)
            plt.title('各情感类别评论数量')
            plt.xlabel('情感类别')
            plt.ylabel('评论数量')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. 子版块情感分析图
        subreddit_data = self.analyze_subreddit_sentiment()
        if subreddit_data and not subreddit_data['subreddit_sentiment'].empty:
            plt.figure(figsize=(12, 8))
            
            # 平均情感分数
            top_subreddits = subreddit_data['subreddit_sentiment'].head(10)
            plt.subplot(2, 1, 1)
            sns.barplot(data=top_subreddits, x='avg_sentiment', y='subreddit.name')
            plt.title('主要子版块平均情感分数')
            plt.xlabel('平均情感分数')
            
            # 评论数量
            plt.subplot(2, 1, 2)
            sns.barplot(data=top_subreddits, x='comment_count', y='subreddit.name')
            plt.title('主要子版块评论数量')
            plt.xlabel('评论数量')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('subreddit_sentiment.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 词频分析图
        word_data = self.generate_word_frequency_analysis(30)
        if word_data:
            plt.figure(figsize=(12, 8))
            top_words = word_data['word_frequency'].head(20)
            sns.barplot(data=top_words, x='count', y='word')
            plt.title('前20个高频词汇')
            plt.xlabel('词频')
            plt.ylabel('词汇')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('word_frequency.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 生成词云
            if len(word_data['word_frequency']) > 0:
                word_freq_dict = dict(zip(word_data['word_frequency']['word'], 
                                        word_data['word_frequency']['count']))
                
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    max_words=100).generate_from_frequencies(word_freq_dict)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('高频词汇词云图')
                
                if save_plots:
                    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def save_results(self) -> bool:
        """
        保存情感分析结果
        
        Returns:
            bool: 保存是否成功
        """
        if self.df is None:
            logger.error("请先完成情感分析")
            return False
        
        try:
            logger.info(f"保存情感分析结果到: {self.output_path}")
            
            # 选择需要保存的列
            columns_to_save = [
                "id", 
                "`subreddit.name`" if "`subreddit.name`" in self.df.columns else "subreddit.name",
                "created_utc", 
                "timestamp",
                "body", 
                "cleaned_body", 
                "tokens_cleaned",
                "sentiment",  # 原始情感分数
                "vader_sentiment",  # VADER情感分数
                "sentiment_category",  # 情感分类
                "score"
            ]
            
            # 过滤存在的列
            available_columns = []
            for col in columns_to_save:
                if col in self.df.columns:
                    available_columns.append(col)
            
            df_final = self.df.select(*available_columns)
            
            # 保存为Parquet格式
            df_final.write.mode("overwrite").parquet(self.output_path)
            
            final_count = df_final.count()
            logger.info(f"✅ 情感分析结果保存完成！共 {final_count:,} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False
    
    def execute_full_sentiment_analysis(self) -> Dict[str, Any]:
        """
        执行完整的情感分析流程
        
        Returns:
            dict: 分析报告
        """
        logger.info("开始执行完整情感分析流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return {'success': False, 'error': '数据加载失败'}
        
        # 2. 基础数据概览
        overview = self.basic_data_overview()
        
        # 3. VADER情感分析
        if not self.apply_vader_sentiment():
            return {'success': False, 'error': 'VADER情感分析失败'}
        
        # 4. 情感分类
        if not self.categorize_sentiment():
            return {'success': False, 'error': '情感分类失败'}
        
        # 5. 各项分析
        sentiment_dist = self.analyze_sentiment_distribution()
        subreddit_analysis = self.analyze_subreddit_sentiment()
        temporal_analysis = self.analyze_temporal_sentiment()
        word_analysis = self.generate_word_frequency_analysis()
        
        # 6. 生成可视化
        self.create_visualizations(save_plots=True)
        
        # 7. 保存结果
        save_success = self.save_results()
        
        # 生成分析报告
        analysis_report = {
            'success': save_success,
            'overview': overview,
            'sentiment_distribution': sentiment_dist,
            'subreddit_analysis': subreddit_analysis,
            'temporal_analysis': temporal_analysis,
            'word_analysis': word_analysis,
            'output_path': self.output_path
        }
        
        logger.info("✅ 完整情感分析流程执行完成！")
        return analysis_report
    
    def cleanup(self):
        """清理资源"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark Session已关闭")

def analyze_sentiment(input_path: Optional[str] = None, 
                     output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：执行情感分析
    
    Args:
        input_path: 输入数据路径
        output_path: 输出数据路径
        
    Returns:
        dict: 分析报告
    """
    analyzer = SentimentAnalyzer(input_path, output_path)
    
    try:
        report = analyzer.execute_full_sentiment_analysis()
        return report
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    # 直接运行此模块时执行情感分析
    logger.info("开始执行情感分析...")
    report = analyze_sentiment()
    
    if report.get('success', False):
        print(f"\n✅ 情感分析成功完成！")
        overview = report.get('overview', {})
        print(f"分析记录数: {overview.get('total_count', 0):,}")
        
        sentiment_dist = report.get('sentiment_distribution', {})
        if sentiment_dist:
            print("\n情感分布:")
            for _, row in sentiment_dist['distribution'].iterrows():
                print(f"  {row['sentiment_category']}: {row['count']:,} ({row['percentage']:.1f}%)")
    else:
        logger.error("❌ 情感分析失败")
        if 'error' in report:
            print(f"错误信息: {report['error']}") 