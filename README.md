
# DeepSeek-OCR — Fox Benchmark Evaluation

This repository provides a reproducible pipeline to evaluate **DeepSeek-OCR** on the **Fox Benchmark** using **VLLM**.  
It includes dataset setup, environment installation, the evaluation script, and metric definitions used to reproduce the results from the DeepSeek-OCR paper.



## 1. Installation

Clone the DeepSeek-OCR repository:

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR
cd DeepSeek-OCR
````

Install all required dependencies:

```bash
pip install -r requirements.txt
```

If `vllm` is not installed, install it manually:

```bash
pip install vllm
```


## 2. Setting Up the Fox Benchmark

Download the Fox Benchmark dataset from HuggingFace:

Dataset link:
[https://huggingface.co/datasets/ucaslcl/Fox_benchmark_data](https://huggingface.co/datasets/ucaslcl/Fox_benchmark_data)

Place the downloaded contents into:

```
DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/
```

Your folder structure should look like:

```
DeepSeek-OCR-vllm/
├── focus_benchmark_test/
│   ├── en_page_ocr.json
│   ├── en_pdf_png/
│   │   ├── page_0001.png
│   │   ├── page_0002.png
│   │   └── ...
```


## 3. Running the Evaluation

Place the evaluation script into:

```
DeepSeek-OCR-vllm/run_fox_eval.py
```

Run it:

```bash
python run_fox_eval.py
```

The script will:

* Load Fox benchmark metadata (`en_page_ocr.json`)
* Load each page image
* Run DeepSeek-OCR inside VLLM
* Compute metrics:

  * **Precision (%) = 1 - CER**
  * **Compression × = text_tokens / vision_tokens**
* Aggregate results into text-token bins
* Save the final table to `fox_results_table.json`


## 4. Metrics

### **Compression per Page**

```
compression_per_page = num_text_tokens / vision_tokens
```

### **Average Compression for Each Bin**

```
avg_compression = sum(compression_per_page for all pages) / number_of_pages
```

### **Precision (%) (CER-based)**

```
precision = 1 - (Levenshtein.distance(pred_text, gt_text) / max(len(gt_text), 1))
```


## 5. Example Results

| Text Tokens | Vision Tokens | Precision (%) | Compression (×) |
| ----------- | ------------- | ------------- | --------------- |
| 600–700     | 64            | 97.7          | 10.49           |
| 600–700     | 100           | 97.7          | 6.73            |
| 700–800     | 64            | 95.78         | 11.75           |
| 700–800     | 100           | 95.78         | 7.52            |
| 800–900     | 64            | 97.30         | 13.21           |
| 800–900     | 100           | 97.30         | 8.45            |
| 900–1000    | 64            | 97.48         | 15.14           |
| 900–1000    | 100           | 97.48         | 9.69            |
| 1000–1100   | 64            | 96.50         | 16.55           |
| 1000–1100   | 100           | 96.50         | 10.58           |
| 1100–1200   | 64            | 97.32         | 17.68           |
| 1100–1200   | 100           | 97.32         | 11.31           |
| 1200–1300   | 64            | 83.50         | 19.70           |
| 1200–1300   | 100           | 83.50         | 12.60           |

These results follow the same precision–compression trend reported in the DeepSeek-OCR paper.


## 6. Output Format

The script generates a JSON file:

```
fox_results_table.json
```

Example entry:

```json
[
  {
    "Text Tokens": "600-700",
    "Vision Tokens": 64,
    "Precision (%)": 97.7,
    "Compression (×)": 10.49,
    "Pages": 14
  }
]
```


## 7. Notes

* Vision token counts (e.g., 64, 100, 1000) are manually selected for benchmarking.
* Patch size is fixed to `16`, matching the DeepSeek-OCR paper assumptions.
* Longer pages (>1200 text tokens) naturally show lower precision, consistent with the paper.
* You can modify:

  * Token budgets
  * Patch size
  * N-gram constraint window
  * Sampling settings

---

## 8. Summary

This pipeline reproduces the Fox Benchmark evaluation from the DeepSeek-OCR paper:

* Loads Fox data
* Runs inference via VLLM
* Computes CER-based precision
* Computes compression ratio
* Outputs aggregated tables matching the paper’s results

You can extend this evaluation to compare:

* Fine-tuned variants
* Alternative VLLM configurations
* Different vision token budgets
* Patch-size sweeps


