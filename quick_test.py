#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境和导入
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_environment():
    """测试环境"""
    print("🧪 测试Python环境...")
    print(f"Python版本: {sys.version}")
    print(f"项目路径: {project_root}")
    
    # 测试基础库
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError:
        print("❌ pandas 未安装")
        
    try:
        import pyspark
        print(f"✅ pyspark {pyspark.__version__}")
    except ImportError:
        print("❌ pyspark 未安装")
        
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ vaderSentiment 可用")
    except ImportError:
        print("❌ vaderSentiment 未安装")

def test_imports():
    """测试项目模块导入"""
    print("\n🔧 测试项目模块导入...")
    
    modules = [
        'src.config',
        'src.utils', 
        'src.data_extraction',
        'src.sentiment_analysis'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")

def test_spark():
    """测试Spark"""
    print("\n⚡ 测试Spark...")
    try:
        from src.utils import create_spark_session
        spark = create_spark_session("test")
        print("✅ Spark Session 创建成功")
        spark.stop()
        print("✅ Spark Session 已关闭")
    except Exception as e:
        print(f"❌ Spark 测试失败: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("Tweet情感分析环境测试")
    print("=" * 50)
    
    test_environment()
    test_imports()
    test_spark()
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    main() 