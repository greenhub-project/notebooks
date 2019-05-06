FROM python:3.7-slim

COPY ./requirements.txt /tmp/requirements.txt

RUN mkdir /opt/notebooks

RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

CMD [ "bash", "-c", "jupyter notebook --ip='0.0.0.0' --port=8888 --notebook-dir=/opt/notebooks --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''" ]
