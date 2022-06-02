install:
	pip install -r requirements.txt
	bash ./tasks.sh release-install

build:
	pip install -r requirements.txt
	bash ./tasks.sh release-build

clean:
	pip uninstall -y s2gpp
	pip uninstall -y -r requirements.txt
	rm -r ./wheels

build-docker:
	docker build -t s2gpp-python-build -f Python.Dockerfile .
	docker run --rm -v $(pwd)/wheels:/results/wheels s2gpp-python-build build

deploy-docker:
	docker build -t s2gpp-python-build -f Python.Dockerfile .
	docker run --rm s2gpp-python-build deploy

deploy:
	maturin upload ./wheels/s2gpp-*.whl
