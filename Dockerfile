FROM python:3.6.9-stretch

# ------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.7.3 && pip install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/biaflows-utilities.git && \
    cd /biaflows-utilities/ && git checkout tags/v0.9.1 && pip install .

# install utilities binaries
RUN chmod +x /biaflows-utilities/bin/*
RUN cp /biaflows-utilities/bin/* /usr/bin/ && \
    rm -r /biaflows-utilities

# ------------------------------------------------------------------------------

RUN pip install opencv-python
RUN pip install tensorflow
RUN pip install scikit-image
RUN pip install keras

RUN mkdir /app
RUN wget https://biaflows.neubias.org/workflow-files/MU-Lux-CZ-model.h5 -O /app/model.h5
ADD cell_tracker.py /app/cell_tracker.py
ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
