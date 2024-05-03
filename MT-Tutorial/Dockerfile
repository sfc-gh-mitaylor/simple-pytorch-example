ARG BASE_IMAGE=continuumio/miniconda3:4.12.0   
FROM $BASE_IMAGE
RUN conda install python=3.8 && \
    pip install --upgrade pip && \
    pip install torch==2.0.0 && \
    pip install torchdata==0.6.0 && \
    pip install snowflake-ml-python==1.0.12 && \
    pip install snowflake-snowpark-python==1.9.0 \
    pip install accelerate==0.29.3

COPY main.py ./
ENTRYPOINT ["python3", "main.py"]
