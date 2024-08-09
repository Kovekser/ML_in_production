FROM python:3.12

WORKDIR /var/app

RUN python -m pip install pipenv

COPY Pipfile Pipfile
RUN pipenv lock
RUN pipenv sync
RUN pipenv sync --system

COPY docker-entrypoint.sh .
COPY ./app ./app

ENV HOST="0.0.0.0"
ENV PORT=80
ENV WORKERS=1
ENV LOOP=asyncio

EXPOSE 80

CMD ["./docker-entrypoint.sh"]