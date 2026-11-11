#!/bin/bash
set -e
export PYTHONPATH=src
python -m qa_baseline.train --config configs/baseline.json --data data/raw/train-v1.1.json --output runs/baseline
python -m qa_baseline.predict --data runs/baseline/validation-v1.1.json --model runs/baseline/model --output runs/baseline/validation-predictions.json
python -m qa_baseline.evaluate --data runs/baseline/validation-v1.1.json --predictions runs/baseline/validation-predictions.json --output runs/baseline/validation-metrics.json
cat runs/baseline/validation-metrics.json
