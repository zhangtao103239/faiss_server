FROM python:bullseye
WORKDIR /app
# COPY bge-base-zh-v1.5 /app/bge-base-zh-v1.5
COPY tini-amd64 /app/tini
RUN chmod +x /app/tini
COPY requirements.txt /app/requirements.txt
# RUN  pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN  pip install -r /app/requirements.txt
COPY app.py /app/
EXPOSE 8000
ENTRYPOINT [ "/app/tini", "--" ]
CMD [ "python", "/app/app.py" ]
