FROM rust:latest

# install requirements
RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install python3 python3-pip python3-venv build-essential gfortran libopenblas-base libopenblas-dev gcc -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app/
COPY . /app

VOLUME /results

RUN python3 -m venv .venv
ENV PATH=/app/.venv/bin:$PATH

RUN make build
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
