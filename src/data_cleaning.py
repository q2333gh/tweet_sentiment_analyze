"""
数据清洗模块
对应notebook: 2_data_cleaning.ipynb
"""
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from .config import DATA_PATHS, DATA_PROCESSING
from .utils import create_spark_session, setup_logging, create_text_cleaning_udf, print_data_info

logger = setup_logging()

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, input_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        初始化数据清洗器
        
        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
        """
        self.input_path = input_path or DATA_PATHS['ten_percent_sample']
        self.output_path = output_path or DATA_PATHS['cleaned_data']
        self.spark = create_spark_session("DataCleaning")
        self.df_raw = None
        self.df_cleaned = None
        
    def load_data(self) -> bool:
        """
        加载原始数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info(f"加载数据: {self.input_path}")
            
            if self.input_path.endswith('.parquet'):
                self.df_raw = self.spark.read.parquet(self.input_path)
            else:
                self.df_raw = self.spark.read.csv(
                    self.input_path, 
                    header=True, 
                    inferSchema=True, 
                    multiLine=True, 
                    escape='"'
                )
            
            self.df_raw.cache()
            initial_count = self.df_raw.count()
            logger.info(f"数据加载完成，共 {initial_count:,} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def remove_duplicates(self) -> Tuple[int, int]:
        """
        去除重复记录
        
        Returns:
            tuple: (去重前数量, 去重后数量)
        """
        if self.df_raw is None:
            logger.error("请先加载数据")
            return 0, 0
        
        logger.info("开始去除重复记录...")
        initial_count = self.df_raw.count()
        
        # 基于id列去重
        self.df_cleaned = self.df_raw.dropDuplicates(['id'])
        after_dedup_count = self.df_cleaned.count()
        
        removed_count = initial_count - after_dedup_count
        logger.info(f"去重完成: 删除了 {removed_count:,} 条重复记录")
        logger.info(f"去重前: {initial_count:,} 条，去重后: {after_dedup_count:,} 条")
        
        return initial_count, after_dedup_count
    
    def handle_missing_values(self) -> int:
        """
        处理缺失值
        
        Returns:
            int: 处理后的记录数量
        """
        if self.df_cleaned is None:
            logger.error("请先执行去重操作")
            return 0
        
        logger.info("开始处理缺失值...")
        before_count = self.df_cleaned.count()
        
        # 删除body为空的记录（核心分析字段）
        self.df_cleaned = self.df_cleaned.filter(F.col("body").isNotNull())
        self.df_cleaned = self.df_cleaned.filter(F.col("body") != "")
        
        after_count = self.df_cleaned.count()
        removed_count = before_count - after_count
        
        logger.info(f"缺失值处理完成: 删除了 {removed_count:,} 条空评论记录")
        logger.info(f"剩余记录: {after_count:,} 条")
        
        return after_count
    
    def clean_text(self) -> int:
        """
        清洗文本内容
        
        Returns:
            int: 清洗后的记录数量
        """
        if self.df_cleaned is None:
            logger.error("请先处理缺失值")
            return 0
        
        logger.info("开始文本清洗...")
        before_count = self.df_cleaned.count()
        
        # 添加时间戳列
        self.df_cleaned = self.df_cleaned.withColumn(
            "timestamp", 
            F.from_unixtime(F.col("created_utc"))
        )
        
        # 应用文本清洗UDF
        clean_text_udf = create_text_cleaning_udf()
        self.df_cleaned = self.df_cleaned.withColumn(
            "cleaned_body", 
            clean_text_udf(F.col("body"))
        )
        
        # 过滤清洗后为空的记录
        self.df_cleaned = self.df_cleaned.filter(F.col("cleaned_body").isNotNull())
        
        after_count = self.df_cleaned.count()
        removed_count = before_count - after_count
        
        logger.info(f"文本清洗完成: 删除了 {removed_count:,} 条无效文本记录")
        logger.info(f"剩余记录: {after_count:,} 条")
        
        return after_count
    
    def tokenize_and_remove_stopwords(self) -> int:
        """
        分词和去除停用词
        
        Returns:
            int: 处理后的记录数量
        """
        if self.df_cleaned is None:
            logger.error("请先执行文本清洗")
            return 0
        
        logger.info("开始分词和停用词处理...")
        before_count = self.df_cleaned.count()
        
        # 1. 分词
        tokenizer = Tokenizer(inputCol="cleaned_body", outputCol="tokens_raw")
        self.df_cleaned = tokenizer.transform(self.df_cleaned)
        
        # 2. 去除停用词
        remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens_cleaned")
        self.df_cleaned = remover.transform(self.df_cleaned)
        
        # 3. 过滤掉分词后为空的记录
        self.df_cleaned = self.df_cleaned.filter(F.size(F.col("tokens_cleaned")) > 0)
        
        # 4. 过滤词汇数量过少的记录
        min_tokens = DATA_PROCESSING['min_tokens']
        self.df_cleaned = self.df_cleaned.filter(
            F.size(F.col("tokens_cleaned")) >= min_tokens
        )
        
        after_count = self.df_cleaned.count()
        removed_count = before_count - after_count
        
        logger.info(f"分词处理完成: 删除了 {removed_count:,} 条词汇不足的记录")
        logger.info(f"剩余记录: {after_count:,} 条")
        
        # 缓存结果
        self.df_cleaned.cache()
        
        return after_count
    
    def quality_check(self) -> Dict[str, Any]:
        """
        数据质量检查
        
        Returns:
            dict: 质量检查结果
        """
        if self.df_cleaned is None:
            logger.error("请先完成数据清洗流程")
            return {}
        
        logger.info("开始数据质量检查...")
        
        # 1. 基本统计
        total_count = self.df_cleaned.count()
        
        # 2. 文本长度统计
        length_stats = self.df_cleaned.withColumn(
            "cleaned_length", 
            F.length("cleaned_body")
        ).select("cleaned_length").describe().toPandas()
        
        # 3. 词汇数量统计
        token_stats = self.df_cleaned.withColumn(
            "token_count", 
            F.size("tokens_cleaned")
        ).select("token_count").describe().toPandas()
        
        # 4. 清洗前后对比样本
        comparison_samples = self.df_cleaned.select(
            "body", "cleaned_body", "tokens_cleaned"
        ).limit(3).collect()
        
        quality_report = {
            'total_records': total_count,
            'length_statistics': length_stats,
            'token_statistics': token_stats,
            'comparison_samples': comparison_samples
        }
        
        # 打印质量检查结果
        print("\n=== 数据质量检查 ===")
        print(f"总记录数: {total_count:,}")
        
        print("\n清洗后文本长度统计:")
        print(length_stats)
        
        print("\n分词后词汇数量统计:")
        print(token_stats)
        
        print("\n清洗前后对比样本:")
        for i, row in enumerate(comparison_samples):
            print(f"\n样本 {i+1}:")
            print(f"原文: {row['body'][:200]}...")
            print(f"清洗后: {row['cleaned_body'][:200]}...")
            print(f"分词结果: {row['tokens_cleaned'][:10]}...")
        
        return quality_report
    
    def save_cleaned_data(self) -> bool:
        """
        保存清洗后的数据
        
        Returns:
            bool: 保存是否成功
        """
        if self.df_cleaned is None:
            logger.error("请先完成数据清洗流程")
            return False
        
        try:
            logger.info("开始保存清洗后的数据...")
            
            # 选择需要保存的列
            columns_to_save = [
                "id", 
                "`subreddit.name`",  # 使用反引号处理包含点号的列名
                "created_utc", 
                "timestamp",
                "body", 
                "cleaned_body", 
                "tokens_cleaned",
                "sentiment", 
                "score"
            ]
            
            # 过滤存在的列
            available_columns = []
            for col in columns_to_save:
                if col.strip('`') in self.df_cleaned.columns or col in self.df_cleaned.columns:
                    available_columns.append(col)
            
            df_final = self.df_cleaned.select(*available_columns)
            
            logger.info(f"保存到: {self.output_path}")
            df_final.write.mode("overwrite").parquet(self.output_path)
            
            final_count = df_final.count()
            logger.info(f"✅ 数据保存完成！最终数据集包含 {final_count:,} 条清洗后的记录")
            
            # 显示最终数据结构
            print("\n最终数据结构:")
            df_final.printSchema()
            
            return True
            
        except Exception as e:
            logger.error(f"数据保存失败: {str(e)}")
            return False
    
    def execute_full_cleaning_pipeline(self) -> Dict[str, Any]:
        """
        执行完整的数据清洗流程
        
        Returns:
            dict: 清洗报告
        """
        logger.info("开始执行完整数据清洗流程...")
        
        # 1. 加载数据
        if not self.load_data():
            return {'success': False, 'error': '数据加载失败'}
        
        # 2. 去除重复记录
        initial_count, after_dedup = self.remove_duplicates()
        
        # 3. 处理缺失值
        after_missing = self.handle_missing_values()
        
        # 4. 文本清洗
        after_text_clean = self.clean_text()
        
        # 5. 分词和停用词处理
        final_count = self.tokenize_and_remove_stopwords()
        
        # 6. 质量检查
        quality_report = self.quality_check()
        
        # 7. 保存数据
        save_success = self.save_cleaned_data()
        
        # 生成清洗报告
        cleaning_report = {
            'success': save_success,
            'initial_count': initial_count,
            'after_deduplication': after_dedup,
            'after_missing_handling': after_missing,
            'after_text_cleaning': after_text_clean,
            'final_count': final_count,
            'total_removed': initial_count - final_count,
            'retention_rate': final_count / initial_count * 100 if initial_count > 0 else 0,
            'quality_report': quality_report,
            'output_path': self.output_path
        }
        
        # 打印清洗摘要
        print("\n=== 数据清洗摘要 ===")
        print(f"初始记录数: {initial_count:,}")
        print(f"去重后: {after_dedup:,}")
        print(f"处理缺失值后: {after_missing:,}")
        print(f"文本清洗后: {after_text_clean:,}")
        print(f"最终记录数: {final_count:,}")
        print(f"总删除记录: {initial_count - final_count:,}")
        print(f"数据保留率: {final_count / initial_count * 100:.2f}%")
        print(f"输出路径: {self.output_path}")
        
        logger.info("✅ 完整数据清洗流程执行完成！")
        return cleaning_report
    
    def cleanup(self):
        """清理资源"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark Session已关闭")

def clean_data(input_path: Optional[str] = None, 
               output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：执行数据清洗
    
    Args:
        input_path: 输入数据路径
        output_path: 输出数据路径
        
    Returns:
        dict: 清洗报告
    """
    cleaner = DataCleaner(input_path, output_path)
    
    try:
        report = cleaner.execute_full_cleaning_pipeline()
        return report
    finally:
        cleaner.cleanup()

if __name__ == "__main__":
    # 直接运行此模块时执行数据清洗
    logger.info("开始执行数据清洗...")
    report = clean_data()
    
    if report.get('success', False):
        print(f"\n✅ 数据清洗成功完成！")
        print(f"最终数据: {report['final_count']:,} 条记录")
        print(f"数据保留率: {report['retention_rate']:.2f}%")
    else:
        logger.error("❌ 数据清洗失败")
        if 'error' in report:
            print(f"错误信息: {report['error']}") 