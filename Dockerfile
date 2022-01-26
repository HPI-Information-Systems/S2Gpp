FROM mut:5000/akita/rust-base:latest

LABEL maintainer="phillip.wenig@hpi.de"

# install requirements
RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install jq build-essential gfortran libopenblas-base libopenblas-dev gcc -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV ALGORITHM_MAIN="/app/scripts/start_timeeval.sh"
COPY . /app/

# build source code to executable
RUN cargo build --release --package s2gpp --bin s2gpp
