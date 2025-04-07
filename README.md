Project: Probing and steering evaluation awareness. Training probes on llama 3.3 70b instruct to see whether there's internal distinction between test vs deploy prompts.

[Project draft here](https://docs.google.com/document/d/1SEgV-resU_MQcjMiGy5Hge0vqshz165YtZU2BdV4ktI/edit?tab=t.0).

**Usage**
1. Get packages ```pip install -r requirements.txt```
2. Generate probes ```python scripts/generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR```
3. Visualise ```python scripts/analyze_probe.py --model MODEL_PATH --vectors VECTORS_DIR --data TEST_DATA --output RESULTS_DIR --visualize```
