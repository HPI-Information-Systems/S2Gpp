FROM sopedu:5000/akita/rust-base:latest as build

# install requirements
RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install build-essential gfortran libopenblas-base libopenblas-dev gcc -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./src /app/src
COPY ./Cargo.toml /app/Cargo.toml

# build source code to executable
RUN cargo build --release --package s2gpp --bin s2gpp


FROM sopedu:5000/akita/rust-base:latest

LABEL maintainer="phillip.wenig@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install jq gfortran -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=build /app/target/release/s2gpp /app/s2gpp
COPY ./scripts/start_timeeval.sh /app/start_timeeval.sh

EXPOSE 8000

ENV ALGORITHM_MAIN="/app/start_timeeval.sh"
