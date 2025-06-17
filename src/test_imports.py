"""
测试所有模块是否可以正常导入
"""
import sys
import traceback

def test_imports():
    """测试所有模块的导入"""
    modules_to_test = [
        'src.config',
        'src.utils',
        'src.data_extraction',
        'src.data_ingestion',
        'src.data_cleaning',
        'src.sentiment_analysis',
        'src.topic_modeling',
        'src.classification',
        'src.main'
    ]
    
    print("🧪 开始测试模块导入...")
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name} - 导入失败: {str(e)}")
            print(f"   详细错误: {traceback.format_exc()}")
    
    print(f"\n📊 测试结果: {success_count}/{total_count} 模块导入成功")
    
    if success_count == total_count:
        print("🎉 所有模块导入测试通过！")
        return True
    else:
        print("⚠️  部分模块导入失败，请检查依赖和代码")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 开始测试基本功能...")
    
    try:
        # 测试配置模块
        from src.config import DATA_PATHS, SPARK_CONFIG
        print("✅ 配置模块 - 基本功能正常")
        
        # 测试工具模块
        from src.utils import setup_logging
        logger = setup_logging()
        print("✅ 工具模块 - 基本功能正常")
        
        # 测试数据提取模块
        from src.data_extraction import DataExtractor
        print("✅ 数据提取模块 - 基本功能正常")
        
        print("🎉 基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Tweet情感分析模块测试")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试基本功能
    if import_success:
        func_success = test_basic_functionality()
    else:
        func_success = False
    
    print("\n" + "=" * 50)
    if import_success and func_success:
        print("🎊 所有测试通过！模块已准备就绪。")
        sys.exit(0)
    else:
        print("💥 测试失败！请检查错误信息并修复问题。")
        sys.exit(1) 