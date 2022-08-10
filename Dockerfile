FROM waggle/plugin-opencv:4.1.1

RUN pip3 install xarray

COPY app/ /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]