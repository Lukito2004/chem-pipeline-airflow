FROM apache/airflow:2.9.3
COPY requirements.txt /requirements.txt
# Installing the CPU-only PyTorch FIRST so chemprop's torch dependency is satisfied
# and pip never pulls the multi-GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /requirements.txt