FROM python:3.6.9-stretch

# ------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.3.0.poc.1 && pip install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.8 && pip install .

# install utilities binaries
RUN chmod +x /neubiaswg5-utilities/bin/*
RUN cp /neubiaswg5-utilities/bin/* /usr/bin/ && \
    rm -r /neubiaswg5-utilities

# ------------------------------------------------------------------------------

RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install scikit-image
RUN pip install keras
RUN pip install gdown

RUN mkdir /app
RUN cd /app && gdown -O model.h5 https://drive.google.com/file/d/1lZVDnCPmw9GT90oxRe_FIGgC19mKIvV7
ADD cell_tracker.py /app/cell_tracker.py
ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
