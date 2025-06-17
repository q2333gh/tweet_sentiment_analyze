"""
分类建模模块
对应notebook: 5_classification_modeling.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString, CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import classification_report, confusion_matrix
from .config import DATA_PATHS, CLASSIFICATION, SENTIMENT_CONFIG
from .utils import create_spark_session, setup_logging, print_classification_metrics

logger = setup_logging()

class ClassificationModeler:
    """分类建模器"""
    
    def __init__(self, input_path: Optional[str] = None):
        """
        初始化分类建模器
        
        Args:
            input_path: 输入数据路径
        """
        self.input_path = input_path or DATA_PATHS['topic_analyzed']
        self.spark = create_spark_session("Classification")
        self.df = None
        self.train_df = None
        self.test_df = None
        self.models = {}
        self.predictions = {}
        self.evaluations = {}
        
    def load_data(self) -> bool:
        """
        加载数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info(f"加载数据: {self.input_path}")
            
            # 尝试加载主题分析结果
            try:
                self.df = self.spark.read.parquet(self.input_path)
            except:
                # 备选方案：加载情感分析数据
                logger.warning("主题分析数据加载失败，尝试加载情感分析数据...")
                self.df = self.spark.read.parquet(DATA_PATHS['sentiment_analyzed'])
            
            self.df.cache()
            record_count = self.df.count()
            logger.info(f"✅ 数据加载完成，共 {record_count:,} 条记录")
            
            print("\n数据结构:")
            self.df.printSchema()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def prepare_features_and_labels(self) -> bool:
        """
        准备特征和标签
        
        Returns:
            bool: 准备是否成功
        """
        if self.df is None:
            logger.error("请先加载数据")
            return False
        
        logger.info("开始准备特征和标签...")
        
        # 1. 检查并创建情感标签
        if 'sentiment_category' not in self.df.columns:
            if 'vader_sentiment' in self.df.columns:
                # 基于VADER分数创建分类
                def classify_sentiment(score):
                    if score is None:
                        return "Unknown"
                    elif score > SENTIMENT_CONFIG['positive_threshold']:
                        return "Positive"
                    elif score < SENTIMENT_CONFIG['negative_threshold']:
                        return "Negative"
                    else:
                        return "Neutral"
                
                classify_sentiment_udf = F.udf(classify_sentiment, StringType())
                self.df = self.df.withColumn(
                    "sentiment_category",
                    classify_sentiment_udf(F.col("vader_sentiment"))
                )
            elif 'sentiment' in self.df.columns:
                # 基于原始sentiment分数创建分类
                def classify_sentiment(score):
                    if score is None:
                        return "Unknown"
                    elif score > 0.1:
                        return "Positive"
                    elif score < -0.1:
                        return "Negative"
                    else:
                        return "Neutral"
                
                classify_sentiment_udf = F.udf(classify_sentiment, StringType())
                self.df = self.df.withColumn(
                    "sentiment_category",
                    classify_sentiment_udf(F.col("sentiment"))
                )
            else:
                logger.error("未找到情感分数列")
                return False
        
        # 2. 过滤掉"Unknown"标签的数据
        self.df = self.df.filter(F.col("sentiment_category") != "Unknown")
        filtered_count = self.df.count()
        logger.info(f"过滤后数据量: {filtered_count:,} 条记录")
        
        # 3. 检查情感标签分布
        print("情感标签分布:")
        sentiment_dist = self.df.groupBy("sentiment_category").count().orderBy(F.desc("count"))
        sentiment_dist.show()
        
        # 4. 确保有分词结果
        if 'tokens_cleaned' not in self.df.columns:
            logger.info("重新进行文本分词...")
            from pyspark.ml.feature import Tokenizer, StopWordsRemover
            
            # 分词
            tokenizer = Tokenizer(inputCol="cleaned_body", outputCol="tokens_raw")
            self.df = tokenizer.transform(self.df)
            
            # 去停用词
            remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens_cleaned")
            self.df = remover.transform(self.df)
            
            logger.info("分词完成")
        
        logger.info(f"最终用于建模的数据: {self.df.count():,} 条记录")
        return True
    
    def build_text_features(self) -> bool:
        """
        构建文本特征
        
        Returns:
            bool: 构建是否成功
        """
        if self.df is None:
            logger.error("请先准备特征和标签")
            return False
        
        logger.info("开始构建文本特征...")
        
        # 1. TF-IDF向量化
        vocab_size = min(3000, self.df.count() // 10)  # 动态调整词汇表大小
        min_df = max(3.0, self.df.count() * 0.001)     # 动态调整最小文档频率
        
        count_vectorizer = CountVectorizer(
            inputCol="tokens_cleaned", 
            outputCol="raw_features",
            vocabSize=vocab_size,
            minDF=min_df
        )
        
        count_model = count_vectorizer.fit(self.df)
        self.df = count_model.transform(self.df)
        
        # TF-IDF
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        idf_model = idf.fit(self.df)
        self.df = idf_model.transform(self.df)
        
        logger.info(f"TF-IDF词汇表大小: {len(count_model.vocabulary)}")
        
        # 2. 主题特征处理（如果存在）
        feature_cols = ["tfidf_features"]
        
        if 'dominant_topic' in self.df.columns:
            logger.info("添加主题特征...")
            # 将主题ID转换为数值特征
            from pyspark.ml.feature import OneHotEncoder
            
            # 先转换为数值索引
            topic_indexer = StringIndexer(inputCol="dominant_topic", outputCol="topic_indexed")
            topic_indexer_model = topic_indexer.fit(self.df)
            self.df = topic_indexer_model.transform(self.df)
            
            # OneHot编码
            topic_encoder = OneHotEncoder(inputCol="topic_indexed", outputCol="topic_features")
            self.df = topic_encoder.transform(self.df)
            
            feature_cols.append("topic_features")
        
        # 3. 组合所有特征
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        self.df = assembler.transform(self.df)
        
        logger.info("✅ 文本特征构建完成")
        return True
    
    def prepare_labels(self) -> bool:
        """
        准备标签索引
        
        Returns:
            bool: 准备是否成功
        """
        if self.df is None:
            logger.error("请先构建特征")
            return False
        
        logger.info("开始准备标签索引...")
        
        # 将字符串标签转换为数值索引
        label_indexer = StringIndexer(inputCol="sentiment_category", outputCol="label")
        label_indexer_model = label_indexer.fit(self.df)
        self.df = label_indexer_model.transform(self.df)
        
        # 保存标签映射（用于后续解释）
        self.label_mapping = {i: label for i, label in enumerate(label_indexer_model.labels)}
        
        logger.info(f"标签映射: {self.label_mapping}")
        return True
    
    def split_data(self) -> bool:
        """
        划分训练和测试集
        
        Returns:
            bool: 划分是否成功
        """
        if self.df is None or 'label' not in self.df.columns:
            logger.error("请先准备标签")
            return False
        
        logger.info("开始划分训练和测试集...")
        
        test_ratio = CLASSIFICATION['test_ratio']
        self.train_df, self.test_df = self.df.randomSplit([1-test_ratio, test_ratio], seed=42)
        
        train_count = self.train_df.count()
        test_count = self.test_df.count()
        
        logger.info(f"训练集: {train_count:,} 条记录")
        logger.info(f"测试集: {test_count:,} 条记录")
        logger.info(f"测试集比例: {test_count/(train_count+test_count)*100:.1f}%")
        
        # 缓存数据集
        self.train_df.cache()
        self.test_df.cache()
        
        return True
    
    def train_naive_bayes(self) -> bool:
        """
        训练朴素贝叶斯模型
        
        Returns:
            bool: 训练是否成功
        """
        if self.train_df is None:
            logger.error("请先划分数据集")
            return False
        
        logger.info("开始训练朴素贝叶斯模型...")
        
        # 创建朴素贝叶斯分类器
        nb = NaiveBayes(featuresCol="features", labelCol="label")
        
        # 训练模型
        self.models['naive_bayes'] = nb.fit(self.train_df)
        
        # 预测
        self.predictions['naive_bayes'] = self.models['naive_bayes'].transform(self.test_df)
        
        logger.info("✅ 朴素贝叶斯模型训练完成")
        return True
    
    def train_random_forest(self) -> bool:
        """
        训练随机森林模型
        
        Returns:
            bool: 训练是否成功
        """
        if self.train_df is None:
            logger.error("请先划分数据集")
            return False
        
        logger.info("开始训练随机森林模型...")
        
        # 创建随机森林分类器
        rf = RandomForestClassifier(
            featuresCol="features", 
            labelCol="label",
            numTrees=CLASSIFICATION['num_trees'],
            maxDepth=CLASSIFICATION['max_depth'],
            seed=42
        )
        
        # 训练模型
        self.models['random_forest'] = rf.fit(self.train_df)
        
        # 预测
        self.predictions['random_forest'] = self.models['random_forest'].transform(self.test_df)
        
        logger.info("✅ 随机森林模型训练完成")
        return True
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        评估模型性能
        
        Returns:
            dict: 评估结果
        """
        if not self.predictions:
            logger.error("请先训练模型")
            return {}
        
        logger.info("开始评估模型性能...")
        
        # 创建评估器
        evaluators = {
            'accuracy': MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy"),
            'f1': MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1"),
            'precision': MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision"),
            'recall': MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
        }
        
        evaluation_results = {}
        
        for model_name, prediction_df in self.predictions.items():
            logger.info(f"评估 {model_name} 模型...")
            
            model_results = {}
            for metric_name, evaluator in evaluators.items():
                score = evaluator.evaluate(prediction_df)
                model_results[metric_name] = score
            
            evaluation_results[model_name] = model_results
            
            # 打印结果
            print(f"\n=== {model_name.upper()} 模型评估结果 ===")
            for metric, score in model_results.items():
                print(f"{metric.capitalize()}: {score:.4f}")
        
        self.evaluations = evaluation_results
        return evaluation_results
    
    def generate_detailed_evaluation(self) -> Dict[str, Any]:
        """
        生成详细的评估报告
        
        Returns:
            dict: 详细评估结果
        """
        if not self.predictions:
            logger.error("请先训练模型")
            return {}
        
        logger.info("生成详细评估报告...")
        
        detailed_results = {}
        
        for model_name, prediction_df in self.predictions.items():
            logger.info(f"生成 {model_name} 详细报告...")
            
            # 转换为Pandas进行详细分析
            results_pd = prediction_df.select("label", "prediction").toPandas()
            
            # 将数值标签转换回字符串
            y_true = [self.label_mapping[int(label)] for label in results_pd['label']]
            y_pred = [self.label_mapping[int(pred)] for pred in results_pd['prediction']]
            
            # 分类报告
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # 混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            detailed_results[model_name] = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            # 打印分类报告
            print(f"\n=== {model_name.upper()} 详细分类报告 ===")
            print(classification_report(y_true, y_pred))
        
        return detailed_results
    
    def create_evaluation_visualizations(self, save_plots: bool = False) -> None:
        """
        创建评估可视化图表
        
        Args:
            save_plots: 是否保存图表
        """
        if not self.evaluations:
            logger.error("请先评估模型")
            return
        
        logger.info("生成评估可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 模型性能对比
        metrics_df = pd.DataFrame(self.evaluations).T
        
        plt.figure(figsize=(12, 8))
        
        # 各指标对比
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        x = np.arange(len(metrics))
        width = 0.35
        
        models = list(self.evaluations.keys())
        for i, model in enumerate(models):
            scores = [metrics_df.loc[model, metric] for metric in metrics]
            plt.bar(x + i*width, scores, width, label=model.replace('_', ' ').title())
        
        plt.xlabel('评估指标')
        plt.ylabel('分数')
        plt.title('模型性能对比')
        plt.xticks(x + width/2, [m.capitalize() for m in metrics])
        plt.legend()
        plt.ylim(0, 1)
        
        # 添加数值标签
        for i, model in enumerate(models):
            scores = [metrics_df.loc[model, metric] for metric in metrics]
            for j, score in enumerate(scores):
                plt.text(j + i*width, score + 0.01, f'{score:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 混淆矩阵
        detailed_results = self.generate_detailed_evaluation()
        
        fig, axes = plt.subplots(1, len(detailed_results), figsize=(6*len(detailed_results), 5))
        if len(detailed_results) == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(detailed_results.items()):
            conf_matrix = results['confusion_matrix']
            labels = list(set(results['y_true']))
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=axes[i])
            axes[i].set_title(f'{model_name.replace("_", " ").title()} 混淆矩阵')
            axes[i].set_xlabel('预测标签')
            axes[i].set_ylabel('真实标签')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        获取特征重要性（仅随机森林）
        
        Returns:
            dict: 特征重要性结果
        """
        if 'random_forest' not in self.models:
            logger.warning("随机森林模型未训练，无法获取特征重要性")
            return {}
        
        logger.info("分析随机森林特征重要性...")
        
        # 获取特征重要性
        rf_model = self.models['random_forest']
        importances = rf_model.featureImportances.toArray()
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature_index': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n=== 随机森林特征重要性 (Top 20) ===")
        print(feature_importance.head(20))
        
        return {
            'feature_importance': feature_importance
        }
    
    def execute_full_classification(self) -> Dict[str, Any]:
        """
        执行完整的分类建模流程
        
        Returns:
            dict: 建模报告
        """
        logger.info("开始执行完整分类建模流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return {'success': False, 'error': '数据加载失败'}
        
        # 2. 准备特征和标签
        if not self.prepare_features_and_labels():
            return {'success': False, 'error': '特征和标签准备失败'}
        
        # 3. 构建文本特征
        if not self.build_text_features():
            return {'success': False, 'error': '文本特征构建失败'}
        
        # 4. 准备标签索引
        if not self.prepare_labels():
            return {'success': False, 'error': '标签准备失败'}
        
        # 5. 划分数据集
        if not self.split_data():
            return {'success': False, 'error': '数据集划分失败'}
        
        # 6. 训练模型
        nb_success = self.train_naive_bayes()
        rf_success = self.train_random_forest()
        
        if not (nb_success or rf_success):
            return {'success': False, 'error': '所有模型训练失败'}
        
        # 7. 评估模型
        evaluations = self.evaluate_models()
        detailed_evaluations = self.generate_detailed_evaluation()
        
        # 8. 生成可视化
        self.create_evaluation_visualizations(save_plots=True)
        
        # 9. 特征重要性分析
        feature_importance = self.get_feature_importance()
        
        # 生成建模报告
        modeling_report = {
            'success': True,
            'models_trained': list(self.models.keys()),
            'label_mapping': self.label_mapping,
            'train_count': self.train_df.count() if self.train_df else 0,
            'test_count': self.test_df.count() if self.test_df else 0,
            'evaluations': evaluations,
            'detailed_evaluations': detailed_evaluations,
            'feature_importance': feature_importance
        }
        
        # 打印建模摘要
        print("\n=== 分类建模摘要 ===")
        print(f"训练集大小: {modeling_report['train_count']:,}")
        print(f"测试集大小: {modeling_report['test_count']:,} 条")
        print(f"训练的模型: {', '.join([m.replace('_', ' ').title() for m in modeling_report['models_trained']])}")
        
        if evaluations:
            print("\n模型性能对比:")
            for model, metrics in evaluations.items():
                print(f"  {model.replace('_', ' ').title()}:")
                print(f"    准确率: {metrics['accuracy']:.4f}")
                print(f"    F1分数: {metrics['f1']:.4f}")
        
        logger.info("✅ 完整分类建模流程执行完成！")
        return modeling_report
    
    def cleanup(self):
        """清理资源"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark Session已关闭")

def perform_classification(input_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：执行分类建模
    
    Args:
        input_path: 输入数据路径
        
    Returns:
        dict: 建模报告
    """
    modeler = ClassificationModeler(input_path)
    
    try:
        report = modeler.execute_full_classification()
        return report
    finally:
        modeler.cleanup()

if __name__ == "__main__":
    # 直接运行此模块时执行分类建模
    logger.info("开始执行分类建模...")
    report = perform_classification()
    
    if report.get('success', False):
        print(f"\n✅ 分类建模成功完成！")
        print(f"训练集: {report['train_count']:,} 条")
        print(f"测试集: {report['test_count']:,} 条")
        
        # 显示最佳模型
        if report.get('evaluations'):
            best_model = max(report['evaluations'].items(), key=lambda x: x[1]['accuracy'])
            print(f"最佳模型: {best_model[0].replace('_', ' ').title()}")
            print(f"最佳准确率: {best_model[1]['accuracy']:.4f}")
    else:
        logger.error("❌ 分类建模失败")
        if 'error' in report:
            print(f"错误信息: {report['error']}") 