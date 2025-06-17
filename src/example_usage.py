"""
使用示例脚本 - 展示如何使用各个模块
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_single_module():
    """示例：运行单个模块"""
    print("=" * 60)
    print("示例1: 运行单个模块")
    print("=" * 60)
    
    # 示例：数据提取
    print("\n1. 数据提取示例:")
    print("```python")
    print("from src.data_extraction import extract_ten_percent_sample")
    print("")
    print("# 提取10%样本数据")
    print("success = extract_ten_percent_sample()")
    print("print(f'数据提取成功: {success}')")
    print("```")
    
    # 示例：情感分析
    print("\n2. 情感分析示例:")
    print("```python")
    print("from src.sentiment_analysis import analyze_sentiment")
    print("")
    print("# 执行情感分析")
    print("report = analyze_sentiment()")
    print("if report.get('success'):")
    print("    print('情感分析完成！')")
    print("    dist = report.get('sentiment_distribution', {})")
    print("    if dist and 'distribution' in dist:")
    print("        print('情感分布:')")
    print("        for _, row in dist['distribution'].iterrows():")
    print("            print(f'  {row["sentiment_category"]}: {row["count"]:,}')")
    print("```")

def example_full_pipeline():
    """示例：运行完整流水线"""
    print("\n" + "=" * 60)
    print("示例2: 运行完整流水线")
    print("=" * 60)
    
    print("\n使用Pipeline类:")
    print("```python")
    print("from src.main import TweetSentimentAnalysisPipeline")
    print("")
    print("# 创建流水线实例")
    print("pipeline = TweetSentimentAnalysisPipeline()")
    print("")
    print("# 运行完整流水线")
    print("final_report = pipeline.run_full_pipeline()")
    print("")
    print("# 查看结果")
    print("if final_report['pipeline_summary']['success_rate'] > 80:")
    print("    print('流水线执行成功！')")
    print("    print(f'成功率: {final_report["pipeline_summary"]["success_rate"]:.1f}%')")
    print("    ")
    print("    # 查看关键指标")
    print("    metrics = final_report.get('key_metrics', {})")
    print("    if 'best_model' in metrics:")
    print("        best = metrics['best_model']")
    print("        print(f'最佳模型: {best["name"]} (准确率: {best["accuracy"]:.4f})')")
    print("```")

def example_step_by_step():
    """示例：逐步执行"""
    print("\n" + "=" * 60)
    print("示例3: 逐步执行流水线")
    print("=" * 60)
    
    print("\n```python")
    print("from src.main import TweetSentimentAnalysisPipeline")
    print("")
    print("# 创建流水线实例")
    print("pipeline = TweetSentimentAnalysisPipeline()")
    print("")
    print("# 逐步执行")
    print("print('步骤1: 数据提取')")
    print("success1 = pipeline.run_data_extraction()")
    print("")
    print("print('步骤2: 数据摄取分析')")
    print("success2 = pipeline.run_data_ingestion()")
    print("")
    print("print('步骤3: 数据清洗')")
    print("success3 = pipeline.run_data_cleaning()")
    print("")
    print("print('步骤4: 情感分析')")
    print("success4 = pipeline.run_sentiment_analysis()")
    print("")
    print("print('步骤5: 主题建模')")
    print("success5 = pipeline.run_topic_modeling()")
    print("")
    print("print('步骤6: 分类建模')")
    print("success6 = pipeline.run_classification()")
    print("")
    print("# 检查所有步骤是否成功")
    print("all_success = all([success1, success2, success3, success4, success5, success6])")
    print("print(f'所有步骤完成: {all_success}')")
    print("```")

def example_command_line():
    """示例：命令行使用"""
    print("\n" + "=" * 60)
    print("示例4: 命令行使用")
    print("=" * 60)
    
    print("\n1. 运行完整流水线:")
    print("```bash")
    print("python -m src.main --step all")
    print("```")
    
    print("\n2. 运行单个步骤:")
    print("```bash")
    print("# 数据提取")
    print("python -m src.main --step extraction")
    print("")
    print("# 情感分析")
    print("python -m src.main --step sentiment")
    print("")
    print("# 主题建模")
    print("python -m src.main --step topic")
    print("")
    print("# 分类建模")
    print("python -m src.main --step classification")
    print("```")
    
    print("\n3. 跳过某些步骤:")
    print("```bash")
    print("# 跳过数据提取和摄取，从清洗开始")
    print("python -m src.main --step all --skip data_extraction data_ingestion")
    print("```")

def example_configuration():
    """示例：配置自定义参数"""
    print("\n" + "=" * 60)
    print("示例5: 自定义配置")
    print("=" * 60)
    
    print("\n修改config.py中的参数:")
    print("```python")
    print("# 修改数据处理参数")
    print("DATA_PROCESSING = {")
    print("    'sample_fraction': 0.05,  # 改为5%样本")
    print("    'random_state': 42,")
    print("    'min_text_length': 20,    # 增加最小文本长度")
    print("    'min_tokens': 10,         # 增加最小词汇数")
    print("    'vocab_size': 5000,       # 增加词汇表大小")
    print("    'min_doc_freq': 5.0")
    print("}")
    print("")
    print("# 修改主题建模参数")
    print("TOPIC_MODELING = {")
    print("    'num_topics': 10,         # 增加主题数量")
    print("    'max_iterations': 20,     # 增加迭代次数")
    print("    'vocab_size': 3000,")
    print("    'min_doc_freq': 15.0,")
    print("    'sample_fraction': 0.5,   # 增加采样比例")
    print("    'max_terms_per_topic': 20")
    print("}")
    print("```")

def example_error_handling():
    """示例：错误处理"""
    print("\n" + "=" * 60)
    print("示例6: 错误处理")
    print("=" * 60)
    
    print("\n```python")
    print("from src.sentiment_analysis import analyze_sentiment")
    print("import logging")
    print("")
    print("# 设置日志级别")
    print("logging.basicConfig(level=logging.INFO)")
    print("")
    print("try:")
    print("    # 执行情感分析")
    print("    report = analyze_sentiment()")
    print("    ")
    print("    if report.get('success', False):")
    print("        print('✅ 情感分析成功完成')")
    print("        # 处理结果...")
    print("    else:")
    print("        print('❌ 情感分析失败')")
    print("        error_msg = report.get('error', '未知错误')")
    print("        print(f'错误信息: {error_msg}')")
    print("        ")
    print("except Exception as e:")
    print("    print(f'执行过程中出现异常: {str(e)}')")
    print("    # 记录详细错误信息")
    print("    import traceback")
    print("    traceback.print_exc()")
    print("```")

def main():
    """主函数"""
    print("🚀 Tweet情感分析模块使用示例")
    print("本脚本展示了如何使用各个模块进行情感分析")
    
    # 显示所有示例
    example_single_module()
    example_full_pipeline()
    example_step_by_step()
    example_command_line()
    example_configuration()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("📚 更多信息")
    print("=" * 60)
    print("- 详细文档: src/README.md")
    print("- 配置文件: src/config.py")
    print("- 测试脚本: src/test_imports.py")
    print("- 主运行脚本: src/main.py")
    
    print("\n💡 提示:")
    print("1. 运行前请确保安装所有依赖: pip install -r requirements.txt")
    print("2. 根据实际情况修改config.py中的数据路径")
    print("3. 确保有足够的内存运行Spark (建议16GB+)")
    print("4. 如遇到问题，请查看日志输出获取详细错误信息")

if __name__ == "__main__":
    main() 