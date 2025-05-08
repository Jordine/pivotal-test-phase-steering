WIP

Project: Probing and steering evaluation awareness. Training probes on llama 3.3 70b instruct to see whether there's internal distinction between test vs deploy prompts, and steering with SAE features to detect sandbagging. This repo is a minimal implementation of probe training and evaluation.

Project draft here.

Results: Linear probes trained with simple contrastive data generalises to distinguishing real eval and deploy prompts.


<img width="228" alt="poster - roc" src="https://github.com/user-attachments/assets/281173f6-13b2-4732-a0fb-419ed1fc027f" />


**Usage**

1. Clone the repo
2. Get packages ```pip install -r requirements.txt```
3. Generate probes ```python scripts/generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR```
4. Visualise ```python scripts/analyze_probe.py --model MODEL_PATH --vectors VECTORS_DIR --data TEST_DATA --output RESULTS_DIR --visualize```
