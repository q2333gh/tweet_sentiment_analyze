好的，我们来根据 `goal.md` 的目标，制定一份详尽的、基于 PySpark 的实现计划。这份计划将涵盖从环境设置到最终模型评估的每一个环节，并提供详细的技术说明和代码片段，旨在做到“开箱即用”，同时保证代码的健壮性、可读性和可扩展性。

考虑到您提出的10k字数要求，这份文档将会非常详尽，深入到每个步骤的技术细节。

---

# **基于 Spark 的推文情感与主题分析系统：详细实现计划 (Implementation Plan)**

## **引言 (Introduction)**

### 1. 项目目标 (Project Goal)

本项目旨在对推文数据集进行全面的分析，流程包括：
1.  **基础探索**：了解数据集的宏观信息，如数据量、时间跨度、语言分布等。
2.  **数据清洗**：处理原始数据中的噪声，为后续分析建立高质量的数据基础。
3.  **探索性分析 (EDA)**：通过可视化手段，洞察数据清洗后推文的核心特征和趋势。
4.  **情感分析 (Sentiment Analysis)**：使用 `VADER` 对每条推文进行情感打分和分类，并分析情感随时间的变化和分布。
5.  **主题建模 (Topic Modeling)**：使用 `LDA` 模型从文本中提取潜在的主题，并分析每个主题下的情感倾向。
6.  **分类建模 (Classification Modeling)**：基于 `VADER` 产生的情感标签，训练并评估机器学习模型（如 `Naive Bayes`, `Random Forest`），以预测推文的情感。

### 2. 技术栈选择 (Technology Stack)

为了高效处理可能大规模的数据集（即便当前是在32G内存的单机上运行），我们选择以 `Apache Spark` 作为核心计算引擎。其分布式计算模型能够保证方案的水平扩展能力。

*   **核心引擎**: `Apache Spark 3.x`。我们将使用其 Python API, `PySpark`。
*   **数据操作**: `Spark DataFrame API` 和 `Spark SQL`，用于结构化数据的高效处理。
*   **环境与部署**: `Docker` 和 `Docker Compose`，用于构建一个隔离、可复现的开发和运行环境，内含 `JupyterLab` 以便于交互式分析。
*   **NLP 与机器学习**:
    *   `pyspark.ml.feature`: Spark 原生的特征工程库，用于文本分词 (Tokenizer)、停用词移除 (StopWordsRemover)、向量化 (HashingTF, IDF, CountVectorizer) 等。
    *   `pyspark.ml.classification`: Spark 原生的分类模型库，包含 `NaiveBayes` 和 `RandomForestClassifier`。
    *   `pyspark.ml.clustering`: Spark 原生的聚类模型库，包含 `LatentDirichletAllocation` (LDA)。
    *   `pyspark.ml.evaluation`: Spark 原生的模型评估工具。
*   **情感分析**: `VADER (vaderSentiment-py)`，这是一个专门为社交媒体文本优化的、基于规则和词典的情感分析工具。我们将通过 Spark UDF (User-Defined Function) 的方式将其集成。
*   **辅助与可视化**:
    *   `Pandas`: 用于将 Spark 分布式计算的小规模结果集转换格式，以便于与本地可视化库对接。
    *   `Matplotlib` & `Seaborn`: 用于绘制高质量的统计图表。
    *   `wordcloud`: 用于生成词云图。
    *   `NLTK`: 可用于提供标准的英文停用词列表，或用于更复杂的文本处理任务。
    *   `re`: Python 内置的正则表达式库，用于文本的模式匹配和清洗。

### 3. 项目结构 (Project Structure)

一个良好组织的项目结构是可维护性的关键。建议如下：

```
tweet_sentiment_analyze/
├── docker-compose.yml       # Docker编排文件
├── Dockerfile               # 自定义Docker镜像配置文件
├── data/
│   ├── raw/                 # 存放原始数据集
│   └── processed/           # 存放清洗和处理后的数据
├── notebooks/
│   ├── 1_data_ingestion_and_stats.ipynb
│   ├── 2_data_cleaning.ipynb
│   ├── 3_eda_and_sentiment_analysis.ipynb
│   ├── 4_topic_modeling.ipynb
│   └── 5_classification_modeling.ipynb
├── src/                     # 存放可复用的Python代码（如UDF函数）
│   ├── __init__.py
│   └── utils.py
└── README.md                # 项目说明
```

---

## **第一步：环境搭建与数据加载 (Environment Setup & Data Loading)**

### 1. Docker 环境配置

我们将使用 `Docker` 来创建一个包含 `PySpark` 和 `JupyterLab` 的标准化环境。

**`Dockerfile`**:
这个文件定义了我们的基础镜像和需要额外安装的 Python 包。

```dockerfile
# 使用官方提供的PySpark Notebook镜像
FROM jupyter/pyspark-notebook:latest

# 切换到root用户以安装依赖
USER root

# 安装项目所需的Python库
RUN pip install --no-cache-dir \
    vaderSentiment \
    wordcloud \
    nltk \
    langdetect \
    seaborn

# 下载NLTK所需的数据
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('words')"

# 切换回默认的 jovyan 用户
USER jovyan
```

**`docker-compose.yml`**:
这个文件用于启动和管理我们的服务。

```yaml
version: '3.8'
services:
  spark-app:
    build: .
    container_name: tweet-analysis-container
    ports:
      - "8888:8888"  # JupyterLab UI
      - "4040:4040"  # Spark UI
    volumes:
      - .:/home/jovyan/work # 将当前目录挂载到容器的工作目录
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - SPARK_OPTS=--driver-memory=16g --executor-memory=16g # 为Spark分配内存，32G内存可多分点
```

**启动命令**:
在项目根目录下运行 `docker-compose up --build`。首次运行会构建镜像，之后直接 `docker-compose up` 即可。

### 2. 启动 Spark Session

在 Jupyter Notebook 的第一个单元格中，我们将配置并初始化 `SparkSession`。

```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# 构建SparkSession
# 在docker-compose中已经配置了内存，这里可以简化
# 但在本地非docker环境运行时，这些配置很重要
spark = SparkSession.builder \
    .appName("TweetAnalysis") \
    .master("local[*]") \  # 使用所有可用的本地核心
    .config("spark.driver.memory", "16g") \ # 为Driver分配16G内存
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \ # 启用Arrow优化，加速Spark与Pandas转换
    .getOrCreate()

sc = spark.sparkContext
```

### 3. 数据加载与初步探查

假设原始数据是 CSV 格式。

```python
# 定义数据路径
raw_data_path = "/home/jovyan/work/data/raw/tweets.csv" # 容器内的路径

# 加载数据，并让Spark自动推断Schema
# 如果CSV文件很大且没有header，需要手动定义Schema以提高效率
df_raw = spark.read.csv(raw_data_path, header=True, inferSchema=True, multiLine=True, escape='"')

# 查看数据结构和前几行
df_raw.printSchema()
df_raw.show(5, truncate=False)

# 缓存DataFrame，后续操作会更快
df_raw.cache()
```

---

## **第二步：数据集基本信息（描述性统计）**

此阶段的目标是在数据清洗前对数据集有一个宏观的认识。

1.  **数据量**:
    ```python
    total_count = df_raw.count()
    print(f"Total number of tweets: {total_count}")
    ```

2.  **时间范围**: 假设有一个名为 `created_at` 的日期列。
    ```python
    df_raw.select(F.min("created_at"), F.max("created_at")).show()
    ```

3.  **推文语言**: 这需要一个 UDF 和 `langdetect` 库。
    ```python
    from langdetect import detect, LangDetectException

    def detect_lang(text):
        try:
            return detect(str(text))
        except (LangDetectException, TypeError):
            return None

    detect_lang_udf = F.udf(detect_lang, StringType())
    
    df_lang = df_raw.withColumn("language", detect_lang_udf(F.col("text")))
    
    # 显示语言分布
    lang_distribution = df_lang.groupBy("language").count().orderBy(F.desc("count"))
    lang_distribution.show()
    ```
    **注意**: 在大规模数据集上对每一行运行 UDF 成本很高。这是一个初步探索，在清洗阶段我们会用这个信息来过滤数据。

4.  **推文平均长度**:
    ```python
    df_raw.withColumn("text_length", F.length(F.col("text"))) \
          .select(F.avg("text_length").alias("average_tweet_length")) \
          .show()
    ```

5.  **唯一用户数**: 假设有一个名为 `user_screen_name` 的列。
    ```python
    unique_users_count = df_raw.select("user_screen_name").distinct().count()
    print(f"Number of unique users: {unique_users_count}")
    ```

6.  **数据清洗前词云和词频图**:
    ```python
    # 1. 文本分词
    words_df = df_raw.select(F.explode(F.split(F.lower(F.col("text")), "\s+")).alias("word"))
    
    # 2. 清理非字母数字的字符
    words_df = words_df.withColumn("word", F.regexp_replace(F.col("word"), "[^a-zA-Z0-9]", ""))
    
    # 3. 过滤空字符串
    words_df = words_df.filter(F.col("word") != "")
    
    # 4. 统计词频
    word_counts = words_df.groupBy("word").count().orderBy(F.desc("count"))
    
    # 5. 收集高频词到Pandas DataFrame以进行可视化
    top_words_pd = word_counts.limit(100).toPandas()

    # 6. 生成词云 (Python代码)
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud_dict = dict(zip(top_words_pd['word'], top_words_pd['count']))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # 7. 生成词频图 (Python代码)
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='word', data=top_words_pd.head(20))
    plt.title('Top 20 Most Frequent Words (Before Cleaning)')
    plt.show()
    ```

---

## **第三步：数据清洗 (Data Cleaning)**

这是整个分析流程中最关键的步骤之一。我们将构建一个健壮的、可复用的清洗流程。

### 1. 清洗流程设计

我们将定义一个函数 `clean_tweets(df)`，它接收一个 DataFrame，并返回一个经过一系列转换后的干净 DataFrame。

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.types import StringType, TimestampType, ArrayType
import re

def clean_tweets(df):
    """
    Applies a series of cleaning steps to the input DataFrame.
    """
    # 1. 标准化日期并过滤旧数据
    # 假设日期格式是 'EEE MMM dd HH:mm:ss Z yyyy' (Twitter标准格式)
    df = df.withColumn("timestamp", F.to_timestamp(F.col("created_at"), 'EEE MMM dd HH:mm:ss Z yyyy'))
    # 过滤掉10年以前和时间戳为空的数据
    df = df.filter(F.col("timestamp").isNotNull()).filter(F.col("timestamp") >= F.lit("2014-01-01"))
    
    # 2. 去重和去空
    df = df.dropDuplicates(['id_str']) # 假设'id_str'是推文唯一ID
    df = df.na.drop(subset=["text"])

    # 3. 创建文本清洗UDF
    def clean_text_udf_func(text):
        text = text.lower()  # 小写
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # 去URL
        text = re.sub(r'@\w+', '', text)  # 去@
        text = re.sub(r'#\w+', '', text)  # 去#
        text = re.sub(r'<.*?>', '', text) # 去HTML标签
        text = re.sub(r'\d+', '', text) # 去数字
        text = re.sub(r'[^a-z\s]', '', text) # 去除标点符号和特殊字符
        text = re.sub(r'\s+', ' ', text).strip() # 去除多余空格
        return text

    clean_text_udf = F.udf(clean_text_udf_func, StringType())
    df = df.withColumn("cleaned_text", clean_text_udf(F.col("text")))

    # 4. 去除非英文推文 (如果需要)
    # 这一步计算成本高，可以抽样验证语言分布后决定是否执行
    # from langdetect import detect, LangDetectException
    # def detect_lang(text):
    #     try: return detect(text) == 'en'
    #     except: return False
    # detect_en_udf = F.udf(detect_lang, BooleanType())
    # df = df.filter(detect_en_udf(F.col("cleaned_text")))
    
    # 5. 分词
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokens_raw")
    df = tokenizer.transform(df)

    # 6. 去除停用词
    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens_cleaned")
    df = remover.transform(df)

    # 7. (可选) 去除非英文字符
    # 这是一个更细粒度的操作，移除推文中混杂的非英文单词
    english_words = set(nltk.corpus.words.words())
    def filter_eng_words(tokens):
        return [token for token in tokens if token in english_words]

    filter_eng_udf = F.udf(filter_eng_words, ArrayType(StringType()))
    df = df.withColumn("tokens_final", filter_eng_udf(F.col("tokens_cleaned")))

    # 8. 过滤掉清洗后内容为空的记录
    df = df.filter(F.size(F.col("tokens_final")) > 0)
    
    return df

# 应用清洗流程
df_cleaned = clean_tweets(df_raw)

# 缓存结果并查看
df_cleaned.cache()
df_cleaned.select("text", "cleaned_text", "tokens_final").show(5, truncate=False)
```

### 2. 输出清洗后数据集

为了避免重复执行昂贵的清洗步骤，我们将结果保存为高效的 `Parquet` 格式。

```python
processed_data_path = "/home/jovyan/work/data/processed/cleaned_tweets.parquet"

df_cleaned.write.mode("overwrite").parquet(processed_data_path)

# 之后可以直接加载
# df_cleaned = spark.read.parquet(processed_data_path)
```
`Parquet` 格式是列式存储，支持压缩，并且能自动保存 `Schema`，是 Spark 生态系统中的首选格式。

---

## **第四步：探索性数据分析 (EDA)**

使用清洗后的数据 `df_cleaned`。

1.  **推文数量随时间变化**:
    ```python
    time_series_data = df_cleaned.groupBy(F.window("timestamp", "1 day")).count().orderBy("window")
    
    time_series_pd = time_series_data.toPandas()
    # 提取窗口的开始时间用于绘图
    time_series_pd['start_time'] = time_series_pd['window'].apply(lambda w: w.start)

    plt.figure(figsize=(15, 6))
    plt.plot(time_series_pd['start_time'], time_series_pd['count'])
    plt.title('Daily Tweet Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid(True)
    plt.show()
    ```

2.  **词频图、词云图 (清洗后)**:
    与第二步中的方法类似，但这次是在 `tokens_final` 列上操作。
    ```python
    words_df_cleaned = df_cleaned.select(F.explode(F.col("tokens_final")).alias("word"))
    word_counts_cleaned = words_df_cleaned.groupBy("word").count()
    
    # 后续的可视化步骤与第二步完全相同，只需替换输入数据
    top_words_cleaned_pd = word_counts_cleaned.orderBy(F.desc("count")).limit(100).toPandas()
    
    # ... (复用之前的词云和词频图绘图代码) ...
    ```

---

## **第五步：VADER 情感分析**

我们将 `VADER` 集成到 Spark 中，为每条推文打上情感标签。

1.  **VADER 集成与情感标注**:
    ```python
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from pyspark.sql.types import DoubleType

    # 初始化分析器
    analyzer = SentimentIntensityAnalyzer()

    # 创建UDF来获取VADER的compound得分
    def vader_score_func(text):
        return analyzer.polarity_scores(text)['compound']

    vader_score_udf = F.udf(vader_score_func, DoubleType())

    # 在cleaned_text上应用UDF
    df_sentiment = df_cleaned.withColumn("sentiment_score", vader_score_udf(F.col("cleaned_text")))

    # 根据得分创建情感标签
    df_sentiment = df_sentiment.withColumn("sentiment_label",
        F.when(F.col("sentiment_score") >= 0.05, "positive")
         .when(F.col("sentiment_score") <= -0.05, "negative")
         .otherwise("neutral")
    )

    df_sentiment.cache()
    df_sentiment.select("cleaned_text", "sentiment_score", "sentiment_label").show(5)
    ```

2.  **情绪趋势图 (随时间变化)**:
    ```python
    sentiment_trend_data = df_sentiment.groupBy(F.window("timestamp", "1 week")) \
                                       .agg(F.avg("sentiment_score").alias("avg_sentiment")) \
                                       .orderBy("window")

    sentiment_trend_pd = sentiment_trend_data.toPandas()
    sentiment_trend_pd['start_time'] = sentiment_trend_pd['window'].apply(lambda w: w.start)
    
    plt.figure(figsize=(15, 6))
    plt.plot(sentiment_trend_pd['start_time'], sentiment_trend_pd['avg_sentiment'])
    plt.title('Weekly Average Sentiment Score Over Time')
    # ... (添加标签和网格) ...
    plt.show()
    ```

3.  **情绪分布图与分布表**:
    ```python
    sentiment_distribution = df_sentiment.groupBy("sentiment_label").count()
    sentiment_distribution_pd = sentiment_distribution.toPandas()
    
    # 分布表
    total = sentiment_distribution_pd['count'].sum()
    sentiment_distribution_pd['percentage'] = (sentiment_distribution_pd['count'] / total) * 100
    print("Sentiment Distribution Table:")
    print(sentiment_distribution_pd)

    # 饼图
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_distribution_pd['count'], labels=sentiment_distribution_pd['sentiment_label'], autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.show()
    ```

4.  **情绪得分密度图**:
    ```python
    # 抽样一部分数据以避免Driver内存溢出
    sentiment_scores_pd = df_sentiment.select("sentiment_score").sample(False, 0.1).toPandas()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(sentiment_scores_pd['sentiment_score'], fill=True)
    plt.title('Density Plot of VADER Sentiment Scores')
    plt.xlabel('Sentiment Score (Compound)')
    plt.ylabel('Density')
    plt.show()
    ```

---

## **第六步：LDA 主题建模**

我们将使用 `pyspark.ml` 来发现文本中的潜在主题。

1.  **文本向量化 for LDA**: LDA 需要词袋（Bag-of-Words）向量作为输入。
    ```python
    from pyspark.ml.feature import CountVectorizer
    from pyspark.ml.clustering import LDA

    # 使用 `tokens_final` 列
    cv = CountVectorizer(inputCol="tokens_final", outputCol="features", vocabSize=10000, minDF=5)
    cv_model = cv.fit(df_sentiment)
    df_vectorized = cv_model.transform(df_sentiment)
    ```
    `vocabSize` 和 `minDF` 是重要参数，需要根据数据集调整以过滤掉极低频和极高频的词。

2.  **训练 LDA 模型**:
    ```python
    # k 是主题数，这是一个需要调优的超参数
    num_topics = 10
    lda = LDA(k=num_topics, maxIter=10, featuresCol="features")
    lda_model = lda.fit(df_vectorized)
    ```
    **关于选择 k**: 在 Spark 中计算困惑度 (Perplexity) 或一致性分数 (Coherence Score) 较为复杂。初步分析可以凭经验选择一个值（如10或20），然后检查主题的可解释性。

3.  **输出主题关键词列表**:
    ```python
    vocab = cv_model.vocabulary
    topics = lda_model.describeTopics(maxTermsPerTopic=10)
    
    def get_topics_summary(topics, vocab):
        for i, topic in enumerate(topics.collect()):
            print(f"Topic {i}:")
            term_indices = topic['termIndices']
            term_weights = topic['termWeights']
            for term_idx, weight in zip(term_indices, term_weights):
                print(f"\t{vocab[term_idx]} (weight: {weight:.4f})")
    
    get_topics_summary(topics, vocab)
    ```

4.  **每个主题下的情绪分布**:
    ```python
    from pyspark.sql.types import IntegerType

    # 获取每个文档的主题分布
    df_topics = lda_model.transform(df_vectorized)
    
    # 找到每个文档最主要的主题
    # UDF: 找到向量中最大值所在的索引
    get_dominant_topic_udf = F.udf(lambda v: int(v.argmax()), IntegerType())
    df_topics = df_topics.withColumn("dominant_topic", get_dominant_topic_udf(F.col("topicDistribution")))

    # 按主题和情感进行分组统计
    topic_sentiment_distribution = df_topics.groupBy("dominant_topic", "sentiment_label") \
                                            .count()

    topic_sentiment_distribution.show(30)
    
    # 可视化 (转换到Pandas后使用seaborn的catplot或热力图)
    topic_sentiment_pd = topic_sentiment_distribution.toPandas()
    sns.catplot(data=topic_sentiment_pd, x='dominant_topic', y='count', hue='sentiment_label', kind='bar', height=6, aspect=2)
    plt.title('Sentiment Distribution per Topic')
    plt.show()
    ```

---

## **第七步：分类建模**

我们将使用 `VADER` 生成的 `sentiment_label` 作为“真实标签”，训练模型来预测它。

### 1. 特征工程与 Pipeline 构建

我们将创建一个完整的 `ML Pipeline`，它包含特征转换和模型训练。这使得整个流程清晰且易于部署。

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# A. 准备数据
# 将情感标签 "positive", "negative", "neutral" 转换为数值索引
label_indexer = StringIndexer(inputCol="sentiment_label", outputCol="label")

# B. 特征转换 (使用TF-IDF)
# 我们使用 `tokens_final` 作为输入
hashingTF = HashingTF(inputCol="tokens_final", outputCol="rawFeatures", numFeatures=20000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# C. 划分数据集
(train_data, test_data) = df_sentiment.randomSplit([0.8, 0.2], seed=42)
```

### 2. 训练与评估 Naive Bayes 模型

```python
# 创建 Naive Bayes 分类器
nb = NaiveBayes(featuresCol="features", labelCol="label")

# 构建 Pipeline
nb_pipeline = Pipeline(stages=[label_indexer, hashingTF, idf, nb])

# 训练模型
nb_model = nb_pipeline.fit(train_data)

# 在测试集上进行预测
nb_predictions = nb_model.transform(test_data)

# 评估模型
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

f1_score = evaluator_f1.evaluate(nb_predictions)
accuracy = evaluator_acc.evaluate(nb_predictions)

print("Naive Bayes Model Evaluation:")
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")

# 查看混淆矩阵
print("Confusion Matrix:")
nb_predictions.groupBy("label", "prediction").count().show()
```

### 3. 训练与评估 Random Forest 模型

```python
# 创建 Random Forest 分类器
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)

# 构建 Pipeline
rf_pipeline = Pipeline(stages=[label_indexer, hashingTF, idf, rf])

# 训练模型
rf_model = rf_pipeline.fit(train_data)

# 预测
rf_predictions = rf_model.transform(test_data)

# 评估
f1_score_rf = evaluator_f1.evaluate(rf_predictions)
accuracy_rf = evaluator_acc.evaluate(rf_predictions)

print("\nRandom Forest Model Evaluation:")
print(f"F1 Score: {f1_score_rf}")
print(f"Accuracy: {accuracy_rf}")

# 查看混淆矩阵
print("Confusion Matrix:")
rf_predictions.groupBy("label", "prediction").count().show()
```

### 4. 模型评估报告

将上述结果整理成一个清晰的报告。

| Model           | Accuracy | F1 Score (weighted) |
| :-------------- | :------- | :------------------ |
| Naive Bayes     | {accuracy} | {f1_score}          |
| Random Forest   | {accuracy_rf} | {f1_score_rf}       |

**分析**: 比较两个模型的性能。`Random Forest` 通常在性能上优于 `Naive Bayes`，但训练时间更长。报告中还应包括每个类别的精确率 (Precision) 和召回率 (Recall)，这可以从混淆矩阵计算得出。

---

## **总结与展望 (Conclusion and Future Work)**

### 1. 回顾项目成果

本项目完整地执行了一套标准的NLP分析流程，从数据探索到最终的监督学习建模。通过使用 `PySpark`，整个方案具备了处理大规模数据的能力。我们成功地：
*   清洗并标准化了原始推文数据。
*   洞察了推文数量、情感随时间的变化趋势。
*   提取了数据中的核心主题，并分析了各主题的情感构成。
*   训练了两个分类模型，并验证了使用文本特征预测情感标签的可行性。

### 2. 潜在的改进方向

*   **高级特征工程**:
    *   使用 `Word2Vec` 或 `Doc2Vec` (`pyspark.ml.feature`) 来捕捉词语的语义信息，代替 TF-IDF。
    *   集成 `Spark NLP` 库，使用预训练的 `BERT` 或其他 `Transformer` 模型进行特征提取 (Embeddings)，这通常会带来显著的性能提升。
*   **超参数调优**:
    *   使用 `pyspark.ml.tuning` 中的 `CrossValidator` 和 `ParamGridBuilder` 来系统地搜索 `LDA` (如 `k`) 和分类器 (如 `numTrees` in RF) 的最佳参数组合。
*   **模型解释性**:
    *   对于 `Random Forest`，可以提取特征重要性 (`featureImportances`)，以了解哪些词对情感分类贡献最大。
*   **实时流处理**:
    *   将此批处理流程改造为基于 `Spark Structured Streaming` 的实时分析流程，以处理实时的推文数据流。
*   **模型部署**:
    *   将训练好的 `PipelineModel` 保存下来，并使用 `MLflow` 进行管理，或将其部署为 REST API (例如使用 `Flask` 或 `FastAPI`) 提供实时预测服务。