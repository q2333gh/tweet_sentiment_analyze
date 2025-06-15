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