import contextlib
import csv
import json,re
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class Evaluator:
    def __init__(self, ground_truth_json: str, prediction_csv: str):
        self.ground_truth_json = ground_truth_json
        self.prediction_csv = prediction_csv

        self.ground_truth = self._load_ground_truth(self.ground_truth_json)
        self.predictions = self._load_predictions(self.prediction_csv)

        # Align based on ID ("label - image_path")
        self.aligned_gt, self.aligned_pred = self._align_by_id()

    # ---------------------------------------------------------
    # Loading functions
    # ---------------------------------------------------------
    def _load_ground_truth(self, filepath: str) -> Dict[str, List[str]]:
        """Load ground truth JSON and build a dict {id → components}."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        gt = {}
        for item in data:
            sample_id = f"{item['label']} - {item['image']}"
            gt[sample_id] = item["components"]
        return gt

    def _load_predictions(self, filepath: str) -> Dict[str, Dict]:
        """
        Load predictions and extract:
        - full_text: original prediction string (for ROUGE/BLEU)
        - components: parsed list extracted from [...] via regex (for Jaccard/Exact)
        """
        predictions = {}
        list_pattern = re.compile(r"\[(.*?)\]")

        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                sample_id = row[0].strip()
                raw_pred = row[1].strip()

                # ------- Extract all occurrences of [...] -------
                matches = list_pattern.findall(raw_pred)

                extracted_components = []
                for match in matches:
                    with contextlib.suppress(Exception):
                        # Try to rebuild the list manually
                        # split on comma but keep text intact
                        items = [el.strip().strip("'").strip('"')
                                for el in match.split(",") if el.strip()]
                        extracted_components.extend(items)
                # Final structure
                predictions[sample_id] = {
                    "full_text": raw_pred,
                    "components": extracted_components  # may be empty
                }

        return predictions
    # ---------------------------------------------------------
    # Alignment
    # ---------------------------------------------------------
    def _align_by_id(self):
        """Align predictions to ground truth based on the ID."""
        aligned_gt = {}
        aligned_pred = {}

        for pid, pred_components in self.predictions.items():
            if pid in self.ground_truth:
                aligned_gt[pid] = self.ground_truth[pid]
                aligned_pred[pid] = pred_components

        return aligned_gt, aligned_pred

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    def jaccard_similarity(self, list1, list2):
        set1 = {item.lower().strip() for item in list1}
        set2 = {item.lower().strip() for item in list2}
        union = set1.union(set2)
        return len(set1.intersection(set2)) / len(union) if union else 0.0

    def compute_bleu(self):
        smoothie = SmoothingFunction().method4
        scores = []
        for pid in self.aligned_gt:
            ref = self.aligned_gt[pid]
            pred = self.aligned_pred[pid]
            scores.append(sentence_bleu([ref], pred, smoothing_function=smoothie))
        return sum(scores) / len(scores) if scores else 0.0

    def compute_rouge(self):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        for pid in self.aligned_gt:
            ref_str = " ".join(self.aligned_gt[pid])
            pred_str = " ".join(self.aligned_pred[pid])
            score = scorer.score(ref_str, pred_str)["rougeL"].fmeasure
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def compute_exact_match(self):
        total = len(self.aligned_gt)
        exact = sum(
            set(self.aligned_gt[pid]) == set(self.aligned_pred[pid])
            for pid in self.aligned_gt
        )
        return exact / total if total else 0.0

    def compute_jaccard(self):
        scores = [
            self.jaccard_similarity(self.aligned_gt[pid], self.aligned_pred[pid])
            for pid in self.aligned_gt
        ]
        return sum(scores) / len(scores) if scores else 0.0

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------
    def evaluate(self):
        total_gt = len(self.ground_truth)
        total_pred = len(self.predictions)
        aligned = len(self.aligned_gt)

        bleu = self.compute_bleu()
        rouge = self.compute_rouge()
        jaccard = self.compute_jaccard()
        exact = self.compute_exact_match()

        return {
            "Prediction Count": f"{aligned}/{total_gt}",
            "BLEU": round(bleu, 4),
            "ROUGE-L": round(rouge, 4),
            "Jaccard": round(jaccard, 4),
            "Exact Match": round(exact, 4),
        }


evaluator = Evaluator(
    ground_truth_json="dataset/multimodal/not_merged/test/fruitveg81_vlm_test.json",
    prediction_csv="results/falcon/beit/not_rag/fruitveg81.csv"
)

print(evaluator.evaluate())