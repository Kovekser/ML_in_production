#!/bin/bash
cd app || exit
pipenv run uvicorn \
  --host ${HOST} \
  --port ${PORT} \
  --loop ${LOOP} \
  --workers ${WORKERS} \
  main:app
