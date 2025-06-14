FROM python:3.11

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . ./

EXPOSE 2000

CMD ["gunicorn", "-b", "0.0.0.0:2000", "--access-logfile", "-", "api:app"]