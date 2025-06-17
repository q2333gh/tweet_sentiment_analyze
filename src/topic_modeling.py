"""
主题建模模块 - LDA主题建模
对应notebook: 4_topic_modeling_lda.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Any, List
import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from wordcloud import WordCloud
from .config import DATA_PATHS, TOPIC_MODELING, CLIMATE_KEYWORDS
from .utils import (create_spark_session, setup_logging, create_climate_filter_udf, 
                   get_topic_words_from_model, print_data_info)

logger = setup_logging()

class TopicModeler:
    """主题建模器"""
    
    def __init__(self, input_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        初始化主题建模器
        
        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
        """
        self.input_path = input_path or DATA_PATHS['sentiment_analyzed']
        self.output_path = output_path or DATA_PATHS['topic_analyzed']
        self.spark = create_spark_session("TopicModeling")
        self.df = None
        self.lda_model = None
        self.vocabulary = None
        self.topic_words = None
        
    def load_data(self) -> bool:
        """
        加载情感分析后的数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info(f"加载情感分析数据: {self.input_path}")
            
            # 尝试加载情感分析结果
            try:
                self.df = self.spark.read.parquet(self.input_path)
            except:
                # 备选方案：加载清洗后的数据
                logger.warning("情感分析数据加载失败，尝试加载清洗后数据...")
                self.df = self.spark.read.parquet(DATA_PATHS['cleaned_data'])
            
            self.df.cache()
            record_count = self.df.count()
            logger.info(f"✅ 数据加载完成，共 {record_count:,} 条记录")
            
            print("\n数据结构:")
            self.df.printSchema()
            
            # 检查必要的列
            if 'tokens_cleaned' not in self.df.columns:
                logger.error("❌ 未找到tokens_cleaned列，需要重新分词")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def preprocess_for_topic_modeling(self) -> bool:
        """
        为主题建模预处理数据
        
        Returns:
            bool: 预处理是否成功
        """
        if self.df is None:
            logger.error("请先加载数据")
            return False
        
        logger.info("开始数据预处理...")
        
        # 1. 过滤词汇数量过少的文档
        min_tokens = TOPIC_MODELING.get('min_tokens', 5)
        df_filtered = self.df.filter(F.size(F.col("tokens_cleaned")) >= min_tokens)
        
        filtered_count = df_filtered.count()
        original_count = self.df.count()
        logger.info(f"过滤后数据量: {filtered_count:,} 条记录")
        logger.info(f"保留比例: {filtered_count/original_count*100:.1f}%")
        
        # 2. 过滤Climate Change相关关键词
        climate_filter_udf = create_climate_filter_udf()
        df_climate = df_filtered.filter(climate_filter_udf(F.col("tokens_cleaned")))
        
        climate_count = df_climate.count()
        logger.info(f"Climate相关评论: {climate_count:,} 条记录")
        logger.info(f"占过滤后数据: {climate_count/filtered_count*100:.1f}%")
        
        # 3. 采样数据以减少计算负担
        sample_fraction = TOPIC_MODELING['sample_fraction']
        self.df = df_climate.sample(fraction=sample_fraction, seed=42)
        sample_count = self.df.count()
        
        logger.info(f"采样后数据量: {sample_count:,} 条记录 (原数据的 {sample_fraction*100}%)")
        
        # 缓存最终用于建模的数据
        self.df.cache()
        
        return True
    
    def build_feature_vectors(self) -> bool:
        """
        构建特征向量
        
        Returns:
            bool: 构建是否成功
        """
        if self.df is None:
            logger.error("请先完成数据预处理")
            return False
        
        logger.info("开始构建特征向量...")
        
        # 1. CountVectorizer：将tokens转换为词频向量
        vocab_size = TOPIC_MODELING['vocab_size']
        min_df = TOPIC_MODELING['min_doc_freq']
        
        count_vectorizer = CountVectorizer(
            inputCol="tokens_cleaned", 
            outputCol="raw_features",
            vocabSize=vocab_size,
            minDF=min_df
        )
        
        logger.info("训练CountVectorizer...")
        count_model = count_vectorizer.fit(self.df)
        df_vectorized = count_model.transform(self.df)
        
        # 2. TF-IDF：计算词汇重要性权重
        idf = IDF(inputCol="raw_features", outputCol="features")
        logger.info("训练IDF...")
        idf_model = idf.fit(df_vectorized)
        self.df = idf_model.transform(df_vectorized)
        
        # 保存词汇表
        self.vocabulary = count_model.vocabulary
        
        logger.info(f"✅ 特征向量构建完成")
        logger.info(f"词汇表大小: {len(self.vocabulary)}")
        logger.info(f"特征向量维度: {len(self.vocabulary)}")
        
        # 显示词汇表示例
        print("\n词汇表示例（前20个词）:")
        for i, word in enumerate(self.vocabulary[:20]):
            print(f"{i}: {word}")
        
        return True
    
    def train_lda_model(self) -> bool:
        """
        训练LDA模型
        
        Returns:
            bool: 训练是否成功
        """
        if self.df is None or self.vocabulary is None:
            logger.error("请先构建特征向量")
            return False
        
        logger.info("开始训练LDA模型...")
        
        # 设置LDA参数
        num_topics = TOPIC_MODELING['num_topics']
        max_iter = TOPIC_MODELING['max_iterations']
        
        # 创建LDA模型
        lda = LDA(
            featuresCol="features", 
            topicsCol="topic_distribution",
            k=num_topics,
            maxIter=max_iter,
            seed=42
        )
        
        logger.info(f"训练LDA模型（{num_topics}个主题，最大迭代{max_iter}次）...")
        self.lda_model = lda.fit(self.df)
        
        # 应用模型得到主题分布
        self.df = self.lda_model.transform(self.df)
        
        logger.info("✅ LDA模型训练完成！")
        logger.info(f"模型困惑度: {self.lda_model.logPerplexity(self.df):.2f}")
        logger.info(f"模型对数似然: {self.lda_model.logLikelihood(self.df):.2f}")
        
        return True
    
    def extract_topic_keywords(self) -> bool:
        """
        提取主题关键词
        
        Returns:
            bool: 提取是否成功
        """
        if self.lda_model is None or self.vocabulary is None:
            logger.error("请先训练LDA模型")
            return False
        
        logger.info("开始提取主题关键词...")
        
        max_terms = TOPIC_MODELING['max_terms_per_topic']
        self.topic_words = get_topic_words_from_model(self.lda_model, self.vocabulary, max_terms)
        
        # 显示主题关键词
        print("\n=== 主题关键词列表 ===")
        for topic in self.topic_words:
            topic_id = topic['topic_id']
            words = topic['words'][:10]  # 显示前10个关键词
            weights = topic['weights'][:10]
            
            print(f"\n🔍 主题 {topic_id}:")
            for word, weight in zip(words, weights):
                print(f"  {word}: {weight:.4f}")
        
        logger.info("✅ 主题关键词提取完成！")
        return True
    
    def analyze_topic_distribution(self) -> Dict[str, Any]:
        """
        分析主题分布
        
        Returns:
            dict: 主题分布分析结果
        """
        if self.df is None or 'topic_distribution' not in self.df.columns:
            logger.error("请先训练LDA模型")
            return {}
        
        logger.info("开始分析主题分布...")
        
        # 计算每个文档的主导主题
        def get_dominant_topic(topic_dist):
            if topic_dist is None:
                return -1
            return int(np.argmax(topic_dist.toArray()))
        
        from pyspark.sql.types import IntegerType
        dominant_topic_udf = F.udf(get_dominant_topic, IntegerType())
        
        self.df = self.df.withColumn("dominant_topic", dominant_topic_udf("topic_distribution"))
        
        # 统计各主题的文档数量
        topic_dist = self.df.groupBy("dominant_topic").count().orderBy("dominant_topic")
        topic_dist_pd = topic_dist.toPandas()
        
        print("\n=== 主题分布统计 ===")
        print(topic_dist_pd)
        
        # 计算百分比
        total_docs = topic_dist_pd['count'].sum()
        topic_dist_pd['percentage'] = (topic_dist_pd['count'] / total_docs * 100).round(2)
        
        return {
            'topic_distribution': topic_dist_pd,
            'total_documents': total_docs
        }
    
    def analyze_topic_sentiment(self) -> Dict[str, Any]:
        """
        分析各主题的情感分布
        
        Returns:
            dict: 主题情感分析结果
        """
        if self.df is None or 'dominant_topic' not in self.df.columns:
            logger.error("请先分析主题分布")
            return {}
        
        logger.info("开始分析主题情感分布...")
        
        # 检查是否有情感分类列
        sentiment_col = None
        if 'sentiment_category' in self.df.columns:
            sentiment_col = 'sentiment_category'
        elif 'vader_sentiment' in self.df.columns:
            # 如果没有分类，基于VADER分数创建分类
            def categorize_sentiment(score):
                if score is None:
                    return "Unknown"
                elif score > 0.05:
                    return "Positive"
                elif score < -0.05:
                    return "Negative"
                else:
                    return "Neutral"
            
            from pyspark.sql.types import StringType
            categorize_udf = F.udf(categorize_sentiment, StringType())
            self.df = self.df.withColumn("sentiment_category", categorize_udf("vader_sentiment"))
            sentiment_col = 'sentiment_category'
        
        if sentiment_col is None:
            logger.warning("未找到情感分析结果，跳过主题情感分析")
            return {}
        
        # 统计各主题的情感分布
        topic_sentiment = self.df.groupBy("dominant_topic", sentiment_col) \
                               .count() \
                               .orderBy("dominant_topic", sentiment_col)
        
        topic_sentiment_pd = topic_sentiment.toPandas()
        
        print("\n=== 主题情感分布 ===")
        print(topic_sentiment_pd)
        
        return {
            'topic_sentiment': topic_sentiment_pd
        }
    
    def create_topic_visualizations(self, save_plots: bool = False) -> None:
        """
        创建主题建模可视化
        
        Args:
            save_plots: 是否保存图表
        """
        if self.topic_words is None:
            logger.error("请先提取主题关键词")
            return
        
        logger.info("开始生成主题建模可视化...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 主题关键词可视化
        num_topics = len(self.topic_words)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, topic in enumerate(self.topic_words):
            if i >= 6:  # 最多显示6个主题
                break
                
            topic_id = topic['topic_id']
            words = topic['words'][:10]
            weights = topic['weights'][:10]
            
            ax = axes[i]
            y_pos = np.arange(len(words))
            ax.barh(y_pos, weights, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('权重')
            ax.set_title(f'主题 {topic_id} 关键词')
        
        # 隐藏多余的子图
        for i in range(num_topics, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('topic_keywords.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 生成各主题的词云
        for topic in self.topic_words[:3]:  # 前3个主题
            topic_id = topic['topic_id']
            words = topic['words'][:20]
            weights = topic['weights'][:20]
            
            word_freq_dict = dict(zip(words, weights))
            
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=50).generate_from_frequencies(word_freq_dict)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'主题 {topic_id} 词云图')
            
            if save_plots:
                plt.savefig(f'topic_{topic_id}_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. 主题分布图
        topic_dist_data = self.analyze_topic_distribution()
        if topic_dist_data:
            plt.figure(figsize=(10, 6))
            
            # 饼图
            plt.subplot(1, 2, 1)
            plt.pie(topic_dist_data['topic_distribution']['count'], 
                   labels=[f"主题 {i}" for i in topic_dist_data['topic_distribution']['dominant_topic']],
                   autopct='%1.1f%%', startangle=90)
            plt.title('主题分布')
            
            # 柱状图
            plt.subplot(1, 2, 2)
            sns.barplot(data=topic_dist_data['topic_distribution'], 
                       x='dominant_topic', y='count')
            plt.title('各主题文档数量')
            plt.xlabel('主题ID')
            plt.ylabel('文档数量')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. 主题情感分析图
        topic_sentiment_data = self.analyze_topic_sentiment()
        if topic_sentiment_data and not topic_sentiment_data['topic_sentiment'].empty:
            # 透视表用于热力图
            pivot_table = topic_sentiment_data['topic_sentiment'].pivot(
                index='dominant_topic', 
                columns='sentiment_category', 
                values='count'
            ).fillna(0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_table, annot=True, fmt='g', cmap='YlOrRd')
            plt.title('主题-情感分布热力图')
            plt.xlabel('情感类别')
            plt.ylabel('主题ID')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('topic_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_results(self) -> bool:
        """
        保存主题建模结果
        
        Returns:
            bool: 保存是否成功
        """
        if self.df is None or 'dominant_topic' not in self.df.columns:
            logger.error("请先完成主题建模")
            return False
        
        try:
            logger.info(f"保存主题建模结果到: {self.output_path}")
            
            # 选择需要保存的列
            columns_to_save = [
                "id", 
                "`subreddit.name`" if "`subreddit.name`" in self.df.columns else "subreddit.name",
                "created_utc", 
                "timestamp",
                "body", 
                "cleaned_body", 
                "tokens_cleaned",
                "sentiment",
                "vader_sentiment" if "vader_sentiment" in self.df.columns else None,
                "sentiment_category" if "sentiment_category" in self.df.columns else None,
                "topic_distribution",
                "dominant_topic",
                "score"
            ]
            
            # 过滤存在的列
            available_columns = [col for col in columns_to_save if col and col in self.df.columns]
            
            df_final = self.df.select(*available_columns)
            
            # 保存为Parquet格式
            df_final.write.mode("overwrite").parquet(self.output_path)
            
            final_count = df_final.count()
            logger.info(f"✅ 主题建模结果保存完成！共 {final_count:,} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False
    
    def execute_full_topic_modeling(self) -> Dict[str, Any]:
        """
        执行完整的主题建模流程
        
        Returns:
            dict: 建模报告
        """
        logger.info("开始执行完整主题建模流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return {'success': False, 'error': '数据加载失败'}
        
        # 2. 数据预处理
        if not self.preprocess_for_topic_modeling():
            return {'success': False, 'error': '数据预处理失败'}
        
        # 3. 构建特征向量
        if not self.build_feature_vectors():
            return {'success': False, 'error': '特征向量构建失败'}
        
        # 4. 训练LDA模型
        if not self.train_lda_model():
            return {'success': False, 'error': 'LDA模型训练失败'}
        
        # 5. 提取主题关键词
        if not self.extract_topic_keywords():
            return {'success': False, 'error': '主题关键词提取失败'}
        
        # 6. 分析主题分布
        topic_distribution = self.analyze_topic_distribution()
        topic_sentiment = self.analyze_topic_sentiment()
        
        # 7. 生成可视化
        self.create_topic_visualizations(save_plots=True)
        
        # 8. 保存结果
        save_success = self.save_results()
        
        # 生成建模报告
        modeling_report = {
            'success': save_success,
            'num_topics': TOPIC_MODELING['num_topics'],
            'vocabulary_size': len(self.vocabulary) if self.vocabulary else 0,
            'topic_words': self.topic_words,
            'topic_distribution': topic_distribution,
            'topic_sentiment': topic_sentiment,
            'model_perplexity': self.lda_model.logPerplexity(self.df) if self.lda_model else None,
            'model_likelihood': self.lda_model.logLikelihood(self.df) if self.lda_model else None,
            'output_path': self.output_path
        }
        
        logger.info("✅ 完整主题建模流程执行完成！")
        return modeling_report
    
    def cleanup(self):
        """清理资源"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark Session已关闭")

def perform_topic_modeling(input_path: Optional[str] = None, 
                          output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：执行主题建模
    
    Args:
        input_path: 输入数据路径
        output_path: 输出数据路径
        
    Returns:
        dict: 建模报告
    """
    modeler = TopicModeler(input_path, output_path)
    
    try:
        report = modeler.execute_full_topic_modeling()
        return report
    finally:
        modeler.cleanup()

if __name__ == "__main__":
    # 直接运行此模块时执行主题建模
    logger.info("开始执行主题建模...")
    report = perform_topic_modeling()
    
    if report.get('success', False):
        print(f"\n✅ 主题建模成功完成！")
        print(f"主题数量: {report['num_topics']}")
        print(f"词汇表大小: {report['vocabulary_size']}")
        print(f"模型困惑度: {report.get('model_perplexity', 'N/A')}")
        
        # 显示主题关键词摘要
        if report.get('topic_words'):
            print("\n主题关键词摘要:")
            for topic in report['topic_words'][:3]:
                topic_id = topic['topic_id']
                words = topic['words'][:5]
                print(f"  主题 {topic_id}: {', '.join(words)}")
    else:
        logger.error("❌ 主题建模失败")
        if 'error' in report:
            print(f"错误信息: {report['error']}") 