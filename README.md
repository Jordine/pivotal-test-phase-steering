WIP

Project: Probing and steering evaluation awareness. Training probes on llama 3.3 70b instruct to see whether there's internal distinction between test vs deploy prompts, and steering to detect sandbagging. This repo is a minimal implementation of probe training and evaluation.

[Project draft here](https://docs.google.com/document/d/1SEgV-resU_MQcjMiGy5Hge0vqshz165YtZU2BdV4ktI/edit?tab=t.0).

Results: Linear probes trained with simple contrastive data generalises to distinguishing real eval and deploy prompts.

[roc_curve.pdf](https://github.com/user-attachments/files/19637508/roc_curve.pdf)

Probe classifies deceptive eval prompts as test-like.

[violin_plot (1).pdf](https://github.com/user-attachments/files/19637515/violin_plot.1.pdf)


**Usage**
0. Clone the repo
1. Get packages ```pip install -r requirements.txt```
2. Generate probes ```python scripts/generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR```
3. Visualise ```python scripts/analyze_probe.py --model MODEL_PATH --vectors VECTORS_DIR --data TEST_DATA --output RESULTS_DIR --visualize```
