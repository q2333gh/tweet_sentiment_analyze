"""
数据提取模块 - 从原始数据中提取10%样本
对应notebook: 0_extract_ten_percent.ipynb
"""
import pandas as pd
from typing import Optional
from .config import DATA_PATHS, DATA_PROCESSING, ensure_output_dirs
from .utils import setup_logging

logger = setup_logging()

class DataExtractor:
    """数据提取器类"""
    
    def __init__(self, raw_data_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        初始化数据提取器
        
        Args:
            raw_data_path: 原始数据路径
            output_path: 输出路径
        """
        self.raw_data_path = raw_data_path or DATA_PATHS['raw_data']
        self.output_path = output_path or DATA_PATHS['ten_percent_sample']
        self.sample_fraction = DATA_PROCESSING['sample_fraction']
        self.random_state = DATA_PROCESSING['random_state']
        
        # 确保输出目录存在
        ensure_output_dirs()
        
    def extract_sample(self) -> bool:
        """
        提取10%样本数据
        
        Returns:
            bool: 提取是否成功
        """
        try:
            logger.info("开始加载原始CSV数据...")
            logger.info(f"数据路径: {self.raw_data_path}")
            
            # 1. 加载原始CSV数据
            df = pd.read_csv(self.raw_data_path)
            original_count = len(df)
            logger.info(f"原始数据加载完成，共 {original_count:,} 条记录")
            
            # 2. 随机抽样10%
            logger.info(f"开始随机抽样 {self.sample_fraction*100}%...")
            df_sample = df.sample(
                frac=self.sample_fraction, 
                random_state=self.random_state
            )
            sample_count = len(df_sample)
            logger.info(f"抽样完成，获得 {sample_count:,} 条记录")
            
            # 3. 保存为Parquet格式（更快被Spark加载）
            logger.info(f"保存样本数据到: {self.output_path}")
            df_sample.to_parquet(self.output_path, index=False)
            
            logger.info("✅ 数据提取完成！")
            logger.info(f"原始数据: {original_count:,} 条")
            logger.info(f"样本数据: {sample_count:,} 条")
            logger.info(f"抽样比例: {sample_count/original_count*100:.2f}%")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"❌ 原始数据文件未找到: {self.raw_data_path}")
            return False
        except Exception as e:
            logger.error(f"❌ 数据提取失败: {str(e)}")
            return False
    
    def get_sample_info(self) -> dict:
        """
        获取样本数据信息
        
        Returns:
            dict: 样本数据基本信息
        """
        try:
            df_sample = pd.read_parquet(self.output_path)
            
            info = {
                'total_records': len(df_sample),
                'columns': list(df_sample.columns),
                'column_count': len(df_sample.columns),
                'memory_usage_mb': df_sample.memory_usage(deep=True).sum() / 1024 / 1024,
                'sample_fraction': self.sample_fraction,
                'output_path': self.output_path
            }
            
            # 检查主要列的数据完整性
            key_columns = ['body', 'sentiment', 'created_utc']
            completeness = {}
            for col in key_columns:
                if col in df_sample.columns:
                    non_null_count = df_sample[col].notna().sum()
                    completeness[col] = non_null_count / len(df_sample) * 100
            
            info['data_completeness'] = completeness
            
            return info
            
        except Exception as e:
            logger.error(f"获取样本信息失败: {str(e)}")
            return {}

def extract_ten_percent_sample(raw_data_path: Optional[str] = None, 
                              output_path: Optional[str] = None) -> bool:
    """
    便捷函数：提取10%样本数据
    
    Args:
        raw_data_path: 原始数据路径
        output_path: 输出路径
        
    Returns:
        bool: 提取是否成功
    """
    extractor = DataExtractor(raw_data_path, output_path)
    return extractor.extract_sample()

if __name__ == "__main__":
    # 直接运行此模块时执行数据提取
    logger.info("开始执行数据提取...")
    success = extract_ten_percent_sample()
    
    if success:
        # 显示样本信息
        extractor = DataExtractor()
        info = extractor.get_sample_info()
        
        print("\n=== 样本数据信息 ===")
        print(f"总记录数: {info.get('total_records', 0):,}")
        print(f"列数: {info.get('column_count', 0)}")
        print(f"内存使用: {info.get('memory_usage_mb', 0):.2f} MB")
        print(f"抽样比例: {info.get('sample_fraction', 0)*100}%")
        
        print("\n数据完整性:")
        for col, completeness in info.get('data_completeness', {}).items():
            print(f"  {col}: {completeness:.1f}%")
            
        print(f"\n输出文件: {info.get('output_path', 'N/A')}")
    else:
        logger.error("数据提取失败，请检查错误信息") 