"""
主运行脚本 - 整合所有分析流程
"""
import argparse
import sys
import os
from typing import Optional, Dict, Any

# 添加当前目录到Python路径，支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 尝试相对导入（作为模块运行时）
    from .utils import setup_logging, ensure_directory_exists
    from .config import ensure_output_dirs
    from .data_extraction import extract_ten_percent_sample
    from .data_ingestion import analyze_data_ingestion
    from .data_cleaning import clean_data
    from .sentiment_analysis import analyze_sentiment
    from .topic_modeling import perform_topic_modeling
    from .classification import perform_classification
except ImportError:
    # 绝对导入（直接运行时）
    from src.utils import setup_logging, ensure_directory_exists
    from src.config import ensure_output_dirs
    from src.data_extraction import extract_ten_percent_sample
    from src.data_ingestion import analyze_data_ingestion
    from src.data_cleaning import clean_data
    from src.sentiment_analysis import analyze_sentiment
    from src.topic_modeling import perform_topic_modeling
    from src.classification import perform_classification

logger = setup_logging()

class TweetSentimentAnalysisPipeline:
    """Tweet情感分析完整流水线"""
    
    def __init__(self):
        """初始化流水线"""
        self.results = {}
        ensure_output_dirs()
        
    def run_data_extraction(self) -> bool:
        """
        步骤1: 数据提取
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤1: 数据提取 ===")
        try:
            success = extract_ten_percent_sample()
            self.results['data_extraction'] = {'success': success}
            
            if success:
                logger.info("✅ 数据提取完成")
            else:
                logger.error("❌ 数据提取失败")
                
            return success
        except Exception as e:
            logger.error(f"数据提取过程中出现错误: {str(e)}")
            self.results['data_extraction'] = {'success': False, 'error': str(e)}
            return False
    
    def run_data_ingestion(self) -> bool:
        """
        步骤2: 数据摄取和统计分析
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤2: 数据摄取和统计分析 ===")
        try:
            report = analyze_data_ingestion(use_sample=True, save_plots=True)
            success = bool(report)
            
            self.results['data_ingestion'] = {
                'success': success,
                'report': report
            }
            
            if success:
                logger.info("✅ 数据摄取分析完成")
            else:
                logger.error("❌ 数据摄取分析失败")
                
            return success
        except Exception as e:
            logger.error(f"数据摄取分析过程中出现错误: {str(e)}")
            self.results['data_ingestion'] = {'success': False, 'error': str(e)}
            return False
    
    def run_data_cleaning(self) -> bool:
        """
        步骤3: 数据清洗
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤3: 数据清洗 ===")
        try:
            report = clean_data()
            success = report.get('success', False)
            
            self.results['data_cleaning'] = report
            
            if success:
                logger.info("✅ 数据清洗完成")
                logger.info(f"最终数据: {report.get('final_count', 0):,} 条记录")
            else:
                logger.error("❌ 数据清洗失败")
                
            return success
        except Exception as e:
            logger.error(f"数据清洗过程中出现错误: {str(e)}")
            self.results['data_cleaning'] = {'success': False, 'error': str(e)}
            return False
    
    def run_sentiment_analysis(self) -> bool:
        """
        步骤4: 情感分析
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤4: 情感分析 ===")
        try:
            report = analyze_sentiment()
            success = report.get('success', False)
            
            self.results['sentiment_analysis'] = report
            
            if success:
                logger.info("✅ 情感分析完成")
                overview = report.get('overview', {})
                logger.info(f"分析记录数: {overview.get('total_count', 0):,}")
            else:
                logger.error("❌ 情感分析失败")
                
            return success
        except Exception as e:
            logger.error(f"情感分析过程中出现错误: {str(e)}")
            self.results['sentiment_analysis'] = {'success': False, 'error': str(e)}
            return False
    
    def run_topic_modeling(self) -> bool:
        """
        步骤5: 主题建模
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤5: LDA主题建模 ===")
        try:
            report = perform_topic_modeling()
            success = report.get('success', False)
            
            self.results['topic_modeling'] = report
            
            if success:
                logger.info("✅ 主题建模完成")
                logger.info(f"主题数量: {report.get('num_topics', 0)}")
                logger.info(f"词汇表大小: {report.get('vocabulary_size', 0)}")
            else:
                logger.error("❌ 主题建模失败")
                
            return success
        except Exception as e:
            logger.error(f"主题建模过程中出现错误: {str(e)}")
            self.results['topic_modeling'] = {'success': False, 'error': str(e)}
            return False
    
    def run_classification(self) -> bool:
        """
        步骤6: 分类建模
        
        Returns:
            bool: 是否成功
        """
        logger.info("=== 步骤6: 分类建模 ===")
        try:
            report = perform_classification()
            success = report.get('success', False)
            
            self.results['classification'] = report
            
            if success:
                logger.info("✅ 分类建模完成")
                logger.info(f"训练模型: {', '.join(report.get('models_trained', []))}")
                
                # 显示最佳模型性能
                evaluations = report.get('evaluations', {})
                if evaluations:
                    best_model = max(evaluations.items(), key=lambda x: x[1].get('accuracy', 0))
                    logger.info(f"最佳模型: {best_model[0]} (准确率: {best_model[1].get('accuracy', 0):.4f})")
            else:
                logger.error("❌ 分类建模失败")
                
            return success
        except Exception as e:
            logger.error(f"分类建模过程中出现错误: {str(e)}")
            self.results['classification'] = {'success': False, 'error': str(e)}
            return False
    
    def run_full_pipeline(self, skip_steps: Optional[list] = None) -> Dict[str, Any]:
        """
        运行完整的分析流水线
        
        Args:
            skip_steps: 跳过的步骤列表
            
        Returns:
            dict: 完整的分析结果
        """
        logger.info("🚀 开始执行Tweet情感分析完整流水线...")
        
        skip_steps = skip_steps or []
        
        # 定义流水线步骤
        pipeline_steps = [
            ('data_extraction', self.run_data_extraction),
            ('data_ingestion', self.run_data_ingestion),
            ('data_cleaning', self.run_data_cleaning),
            ('sentiment_analysis', self.run_sentiment_analysis),
            ('topic_modeling', self.run_topic_modeling),
            ('classification', self.run_classification)
        ]
        
        # 执行每个步骤
        for step_name, step_func in pipeline_steps:
            if step_name in skip_steps:
                logger.info(f"⏭️  跳过步骤: {step_name}")
                continue
                
            logger.info(f"▶️  执行步骤: {step_name}")
            success = step_func()
            
            if not success:
                logger.error(f"❌ 步骤 {step_name} 失败，停止流水线执行")
                break
        
        # 生成最终报告
        final_report = self.generate_final_report()
        
        logger.info("🎉 Tweet情感分析流水线执行完成！")
        return final_report
    
    def generate_final_report(self) -> Dict[str, Any]:
        """
        生成最终分析报告
        
        Returns:
            dict: 最终报告
        """
        logger.info("生成最终分析报告...")
        
        # 统计成功的步骤
        successful_steps = []
        failed_steps = []
        
        for step_name, result in self.results.items():
            if result.get('success', False):
                successful_steps.append(step_name)
            else:
                failed_steps.append(step_name)
        
        # 汇总关键指标
        key_metrics = {}
        
        # 数据量统计
        if 'data_cleaning' in self.results and self.results['data_cleaning'].get('success'):
            cleaning_result = self.results['data_cleaning']
            key_metrics['initial_records'] = cleaning_result.get('initial_count', 0)
            key_metrics['final_records'] = cleaning_result.get('final_count', 0)
            key_metrics['data_retention_rate'] = cleaning_result.get('retention_rate', 0)
        
        # 情感分析结果
        if 'sentiment_analysis' in self.results and self.results['sentiment_analysis'].get('success'):
            sentiment_result = self.results['sentiment_analysis']
            sentiment_dist = sentiment_result.get('sentiment_distribution', {})
            if sentiment_dist and 'distribution' in sentiment_dist:
                key_metrics['sentiment_distribution'] = sentiment_dist['distribution'].to_dict('records')
        
        # 主题建模结果
        if 'topic_modeling' in self.results and self.results['topic_modeling'].get('success'):
            topic_result = self.results['topic_modeling']
            key_metrics['num_topics'] = topic_result.get('num_topics', 0)
            key_metrics['vocabulary_size'] = topic_result.get('vocabulary_size', 0)
        
        # 分类模型结果
        if 'classification' in self.results and self.results['classification'].get('success'):
            classification_result = self.results['classification']
            key_metrics['models_trained'] = classification_result.get('models_trained', [])
            
            evaluations = classification_result.get('evaluations', {})
            if evaluations:
                best_model = max(evaluations.items(), key=lambda x: x[1].get('accuracy', 0))
                key_metrics['best_model'] = {
                    'name': best_model[0],
                    'accuracy': best_model[1].get('accuracy', 0),
                    'f1_score': best_model[1].get('f1', 0)
                }
        
        final_report = {
            'pipeline_summary': {
                'total_steps': len(self.results),
                'successful_steps': len(successful_steps),
                'failed_steps': len(failed_steps),
                'success_rate': len(successful_steps) / len(self.results) * 100 if self.results else 0
            },
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'key_metrics': key_metrics,
            'detailed_results': self.results
        }
        
        # 打印最终报告摘要
        print("\n" + "="*60)
        print("🎯 TWEET情感分析最终报告")
        print("="*60)
        
        print(f"\n📊 流水线执行摘要:")
        print(f"  总步骤数: {final_report['pipeline_summary']['total_steps']}")
        print(f"  成功步骤: {final_report['pipeline_summary']['successful_steps']}")
        print(f"  失败步骤: {final_report['pipeline_summary']['failed_steps']}")
        print(f"  成功率: {final_report['pipeline_summary']['success_rate']:.1f}%")
        
        if key_metrics:
            print(f"\n📈 关键指标:")
            if 'initial_records' in key_metrics:
                print(f"  初始数据: {key_metrics['initial_records']:,} 条记录")
                print(f"  最终数据: {key_metrics['final_records']:,} 条记录")
                print(f"  数据保留率: {key_metrics['data_retention_rate']:.2f}%")
            
            if 'num_topics' in key_metrics:
                print(f"  识别主题数: {key_metrics['num_topics']}")
                print(f"  词汇表大小: {key_metrics['vocabulary_size']:,}")
            
            if 'best_model' in key_metrics:
                best = key_metrics['best_model']
                print(f"  最佳模型: {best['name'].replace('_', ' ').title()}")
                print(f"  模型准确率: {best['accuracy']:.4f}")
                print(f"  模型F1分数: {best['f1_score']:.4f}")
        
        if failed_steps:
            print(f"\n❌ 失败步骤: {', '.join(failed_steps)}")
            print("请检查日志获取详细错误信息")
        
        print("\n" + "="*60)
        
        return final_report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Tweet情感分析流水线')
    parser.add_argument('--step', choices=[
        'extraction', 'ingestion', 'cleaning', 'sentiment', 'topic', 'classification', 'all'
    ], default='all', help='要执行的步骤')
    parser.add_argument('--skip', nargs='*', choices=[
        'data_extraction', 'data_ingestion', 'data_cleaning', 
        'sentiment_analysis', 'topic_modeling', 'classification'
    ], help='要跳过的步骤')
    
    args = parser.parse_args()
    
    # 创建流水线实例
    pipeline = TweetSentimentAnalysisPipeline()
    
    try:
        if args.step == 'all':
            # 运行完整流水线
            final_report = pipeline.run_full_pipeline(skip_steps=args.skip)
        else:
            # 运行单个步骤
            step_mapping = {
                'extraction': pipeline.run_data_extraction,
                'ingestion': pipeline.run_data_ingestion,
                'cleaning': pipeline.run_data_cleaning,
                'sentiment': pipeline.run_sentiment_analysis,
                'topic': pipeline.run_topic_modeling,
                'classification': pipeline.run_classification
            }
            
            if args.step in step_mapping:
                success = step_mapping[args.step]()
                if success:
                    logger.info(f"✅ 步骤 {args.step} 执行成功")
                else:
                    logger.error(f"❌ 步骤 {args.step} 执行失败")
                    sys.exit(1)
            else:
                logger.error(f"未知步骤: {args.step}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行过程中出现未预期错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 