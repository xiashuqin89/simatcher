FROM python:3.10.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN rm -f /etc/localtime \
&& ln -sv /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
&& echo "Asia/Shanghai" > /etc/timezone

# ENTRYPOINT ["uvicorn", "main:app", "--reload", "--host=0.0.0.0", "--port=8000"]
ENTRYPOINT ["python", "main.py"]
