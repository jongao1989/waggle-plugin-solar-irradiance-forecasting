FROM waggle/plugin-tensorflow:2.0.0

RUN pip3 install xarray numpy sklearn

COPY app/ /app/
WORKDIR /app

ENTRYPOINT ["/usr/bin/python3", "/app/app.py"]