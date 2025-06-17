# Notebooks 运行指南

## 🚀 运行顺序

请按以下顺序运行notebooks：

### 1. `3_eda_and_sentiment_analysis.ipynb`
- **状态**: ✅ 已完成，可以直接运行
- **功能**: 探索性数据分析和VADER情感分析
- **输出**: 情感分析结果和可视化图表

### 2. `4_topic_modeling_lda.ipynb` 
- **状态**: 🔧 已修复bug，可以运行
- **功能**: LDA主题建模
- **修复内容**:
  - 减少数据采样量（30%）以避免内存不足
  - 降低词汇表大小（2000词）
  - 减少主题数量（5个主题）
  - 减少迭代次数以加快训练

### 3. `5_classification_modeling.ipynb`
- **状态**: 🔧 已修复bug，可以运行  
- **功能**: 情感分类建模（Naive Bayes + Random Forest）
- **修复内容**:
  - 修复OneHotEncoder错误（检查主题值唯一性）
  - 添加变量存在性检查
  - 改进错误处理和备选方案

## 🐛 已修复的主要问题

### 问题1: Spark内存不足
- **现象**: CountVectorizer训练时出现Py4JError
- **原因**: 数据量过大，内存不足
- **解决方案**: 
  - 数据采样（30%）
  - 减少词汇表大小
  - 提高最小文档频率

### 问题2: OneHotEncoder失败
- **现象**: "requirement failed: The input column topic_index should have at least two distinct values"
- **原因**: 主题值只有一个（临时设置的0值）
- **解决方案**: 
  - 检查主题值的唯一性
  - 如果只有一个值，跳过主题特征，仅使用TF-IDF

### 问题3: 变量未定义错误
- **现象**: NameError: name 'df_features' is not defined
- **原因**: Cell执行顺序或失败导致变量未创建
- **解决方案**: 
  - 添加变量存在性检查
  - 提供备选方案

## 💡 运行建议

1. **内存要求**: 建议至少16GB内存
2. **运行时间**: 
   - Notebook 3: ~5-10分钟
   - Notebook 4: ~10-15分钟（已优化）
   - Notebook 5: ~15-20分钟
3. **顺序执行**: 必须按顺序运行，后续notebook依赖前面的结果
4. **错误处理**: 每个notebook都有错误检查，如果出现问题会给出提示

## 📊 预期输出

- **Notebook 3**: 情感分布图表、子版块分析、时间趋势分析
- **Notebook 4**: 主题关键词、主题分布、主题情感分析
- **Notebook 5**: 模型性能比较、混淆矩阵、分类报告

## 🔧 如果仍有问题

1. 重启Jupyter kernel
2. 确保有足够的内存
3. 检查数据文件是否存在
4. 按顺序重新运行所有cells

现在可以开始运行notebooks了！ 