FROM ${BASE_IMAGE}

WORKDIR /app

ENV BKCHAT_APP_ID=${BKCHAT_APP_ID}
ENV BKCHAT_APP_SECRET=${BKCHAT_APP_SECRET}
ENV BKCHAT_APIGW_ROOT=${BKCHAT_APIGW_ROOT}
ENV BK_SUPER_USERNAME=${BK_SUPER_USERNAME}
ENV ACCESS_TOKEN=${ACCESS_TOKEN}

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN rm -f /etc/localtime \
&& ln -sv /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
&& echo "Asia/Shanghai" > /etc/timezone

ENTRYPOINT ["python", "main.py"]
