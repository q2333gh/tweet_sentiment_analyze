#!/usr/bin/env python3
"""
Tweet情感分析运行脚本
解决容器中的导入问题
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """主函数"""
    print("🚀 开始Tweet情感分析...")
    
    try:
        # 导入主要的流水线类
        from src.main import TweetSentimentAnalysisPipeline
        
        # 创建流水线实例
        pipeline = TweetSentimentAnalysisPipeline()
        
        # 运行完整流水线
        print("正在执行完整分析流水线...")
        final_report = pipeline.run_full_pipeline()
        
        # 显示结果
        if final_report['pipeline_summary']['success_rate'] > 80:
            print("✅ 分析流水线执行成功！")
        else:
            print("⚠️ 分析流水线部分失败，请查看详细日志")
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 