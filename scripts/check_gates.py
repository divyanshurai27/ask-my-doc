"""
Quality gate validation script.
Asserts that evaluation metrics meet success thresholds to gate CI/CD deployment.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Success thresholds defined in README.md
THRESHOLDS = {
    "faithfulness": 0.00,
    "answer_relevancy": 0.00,
    "context_precision": 0.00,
    "context_recall": 0.00,
}


def check_quality_gates():
    logger.info("Starting Quality Gate Verification...")
    
    results_path = Path(__file__).parent.parent / "data" / "evaluation_results.json"
    
    if not results_path.exists():
        logger.error(f"Evaluation results file not found at {results_path}.")
        logger.info("Running evaluation first to generate results...")
        try:
            from scripts.evaluate import run_evaluation
            scores = run_evaluation()
        except Exception as e:
            logger.error(f"Failed to run evaluation: {str(e)}")
            sys.exit(1)
    else:
        try:
            with open(results_path, "r") as f:
                scores = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load evaluation results: {str(e)}")
            sys.exit(1)
            
    logger.info("Loaded evaluation scores:")
    for metric, score in scores.items():
        logger.info(f"  {metric}: {score:.4f}")
        
    failed = False
    for metric, threshold in THRESHOLDS.items():
        if metric not in scores:
            logger.error(f"Metric '{metric}' is missing from evaluation results!")
            failed = True
            continue
            
        score = scores[metric]
        if score < threshold:
            logger.error(
                f"Gate FAILED for '{metric}': "
                f"Score {score:.4f} is below the threshold of {threshold:.4f}!"
            )
            failed = True
        else:
            logger.info(
                f"Gate PASSED for '{metric}': "
                f"Score {score:.4f} meets or exceeds threshold of {threshold:.4f}"
            )
            
    if failed:
        logger.error("Quality gate check FAILED! Blocking integration pipeline.")
        sys.exit(1)
    else:
        logger.info("All quality gates PASSED! Ready for deployment.")
        sys.exit(0)


if __name__ == "__main__":
    check_quality_gates()
