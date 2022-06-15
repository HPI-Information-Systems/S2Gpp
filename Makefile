.DEFAULT_GOAL := help
RUSTFLAGS = "-l static=gfortran -L native="$$(find /usr/lib/gcc/*/*/libgfortran.a | sed 's/libgfortran.a//' | tail -n 1)

.PHONY: help
help: ## Generate list of targets with descriptions
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sed -n 's/^\(.*\): \(.*\)##\(.*\)/\1~\3/p' \
	| column -t -s "~"

.PHONY: install
install: build ## Build and install s2gpp to current Python environment
	cd wheels && pip install --force-reinstall -U s2gpp-*.whl && cd ..

.PHONY: build
build: ## Build s2gpp Python package with RUSTFLAGS
	@pip install -r requirements.txt
	@RUSTFLAGS=$(RUSTFLAGS) maturin build --release --cargo-extra-args="--features python" -o wheels -i $$(which python)

.PHONY: clean
clean: ## Uninstall s2gpp and its requirements and delete wheels
	pip uninstall -y s2gpp
	pip uninstall -y -r requirements.txt
	rm -r ./wheels

.PHONY: build-container
build-container: ## Build Docker container to build s2gpp Python package
	docker build -t s2gpp-python-build -f Python.Dockerfile .

.PHONY: build-docker
build-docker: build-container ## Build s2gpp Python package in Docker container
	docker run --rm -v $(pwd)/wheels:/results/wheels s2gpp-python-build build

.PHONY: deploy-docker
deploy-docker: build-container ## Build s2gpp Python package in Docker container and deploy to PYPI
	docker run --rm s2gpp-python-build deploy

.PHONY: deploy
deploy: ## Upload s2gpp wheels to PYPI
	maturin upload ./wheels/s2gpp-*.whl
