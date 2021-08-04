FROM python:3.7
EXPOSE 8501
WORKDIR /app
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en_core_web_sm
CMD ["streamlit", "run", "app.py"]
