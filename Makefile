.PHONY: install infra-up infra-down train package deploy status rollback list test clean

install:
	pip install -r requirements.txt
	pip install -e .

infra-up:
	docker compose up minio registry -d

infra-full:
	docker compose up -d

infra-down:
	docker compose down

train:
	python examples/fraud_detector/train_model.py
	python examples/sentiment_classifier/train_model.py

package:
	capsule package --manifest examples/fraud_detector/capsule.yaml

package-v2:
	capsule package --manifest examples/fraud_detector/capsule_v2.yaml

deploy:
	capsule deploy fraud-detector:1.0

deploy-canary:
	capsule deploy fraud-detector:2.0 --canary 10

status:
	capsule status fraud-detector

rollback:
	capsule rollback fraud-detector --yes

list:
	capsule list

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=capsule --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
