# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

from typing import Dict, List, Tuple
import json
import math

import numpy as np
from tqdm import tqdm

from serve.utils_llm import get_llm_output


class LLMEvaluator:
    """
    Ask LLM if the hypothesis is true or false.
    """

    system_prompt = """
# Identity

You are an expert evaluator for machine learning research that scores how well a given prediction describes the difference between two groups of images, Group A and Group B. The goal is to find a concept that is more true for Group A than Group B.

# Instructions

Given a description of Group A and Group B, output whether a given prediction aligns with the description of Group A. Answer with a 2 (fully aligned), 1 (somewhat aligned), or 0 (not aligned). A score of 1 should be given if the prediction is more true for A than B, but is a superset or a subset of the most correct difference.
Only respond with the score as a single integer (0, 1, or 2) and nothing else.

## Additional rules for scoring

In edge cases where the evaluation is not straightforward, please follow the following additional rules. However, the focus of the evaluation should always be to determine whether the prediction accurately captures the difference between Group A and Group B (concept that is more true for Group A than Group B)

### Common element in descriptions of Group A and Group B

A common element is an element that is present in both Group A and Group B descriptions. For example, if Group A is "images of dogs in the snow" and Group B is "images of dogs next to cats", then "images of dogs" or "dogs" is a common element.

The following rules apply if the description of Group A and Group B share a common element:
* Common element missing in prediction: score as if the common element was correctly described, i.e. score based only on the part describing the actual difference between Group A and Group B. For above example, the prediction "in the snow" should be given a score of 2, since it correctly describes the difference (snow vs. next to cats). The predictions "next to cats" or "in the rain" should be given a score of 0, since they incorrectly describes the difference. The prediction "in precipitation" should be given a score of 1, since it is a superset of the actual difference ("snow" is a type of "precipitation").
* Superset of common element in prediction: as above, score as if the common element was correctly described, i.e. score based only on the part describing the actual difference between Group A and Group B. For above example, the prediction "animals in the snow" should be given a score of 2, since it correctly describes the difference (snow vs. next to cats). The predictions "animals next to cats" or "animals in the rain" should be given a score of 0, since they incorrectly describes the difference. "Animals in precipitation" should be given a score of 1, since it is a superset of the actual difference.
* Incorrect common element in prediction (superset and subset are not considered incorrect): maximum score of 1, even if the part describing the difference between Group A and Group B is correct. For above example, the prediction "giraffes in the snow" should be given a score of 1, since the common element "dogs" is incorrectly described as "giraffes", even though the difference part "in the snow" is correct. The prediction "giraffes next to cats" should be given a score of 0, since both the common element and the difference part are incorrect. The prediction "giraffes in precipitation" should be given a score of 1, since the common element is incorrect but the difference part is a superset of the actual difference.
* Subset of common element in prediction: maximum score of 1, except if the subset is expected to be present in almost all images of Group A and Group B, in which case the common element should be ignored and the prediction should be evaluated solely based on the part describing the difference between Group A and Group B. For above example, the prediction "puppies in the snow" should be given a score of 1, since "puppies" is a subset of "dogs" but not expected to be present in almost all images of dogs. The prediction "puppies next to cats" should be given a score of 0, since both the common element and the difference part are incorrect. The prediction "puppies in precipitation" should be given a score of 1, since the common element is a subset but the difference part is a superset of the actual difference. On the other hand, the prediction "dogs with hair" should be given a score of 2, since "dogs with hair" is a subset of "dogs" but is expected to be present in almost all images of dogs, as hairless dogs are very rare. The prediction "dogs with hair next to cats" should be given a score of 0, since the difference part is incorrect. The prediction "dogs with hair in precipitation" should be given a score of 1, since the difference part is a superset of the actual difference.

A score of 0 should be given only if the part of the prediction describing the difference between Group A and Group B is incorrect or if the prediction only describes a common element.

### Multiple distinct differences in descriptions of Group A and Group B

If the description of Group A and Group B contain multiple distinct differences, the prediction should be evaluated based on whether it accurately describes at least one of the differences (score based on the most accurate description of any of the differences).
However, if the prediction contains a correct (possibly a subset or superset) description of one of the differences, and an incorrect description of another difference or a subset of another difference, it should receive a score of 1.
For example, if Group A is "dog in a forest" and Group B is "cat in a city", then the prediction "dog in a city" should be given a score of 1, since it correctly describes the difference in animal (dog vs. cat) but incorrectly describes the environment (city vs. forest).
For the same example, the prediction "forest" should be given a score of 2, since it correctly describes the environment difference (forest vs. city) and does not incorrectly describe the animal difference.

### Prediction is a superset or subset of the actual difference

As already mentioned in the main instructions, a score of 1 should be given if the prediction is a superset or a subset of the most correct difference.
This refers to the part of the prediction describing the difference between Group A and Group B (i.e. ignoring any common elements as per the rules above).

#### Expectable subset predictions

However, if the part of the prediction describing the difference between Group A and Group B is strictly speaking a subset of the actual difference, but an image of Group A is expected to almost exlusively focus on the subset, then a score of 2 should be given.
For example, if Group A is "reporter" and Group B is "people", then the prediction "a reporter holding a microphone" should be given a score of 2, since almost all images of reporters are expected to show them holding a microphone.

### Additional elements in predictions not part of Group A or Group B

If the prediction contains elements not part of either Group A or Group B, please follow the following rules:
* Elements of the prediction not describing the content of the images should be ignored. For example, if the prediction is "a photo of dogs in the snow", the phrase "a photo of" should be ignored and the prediction should be evaluated based on "dogs in the snow". Other examples of such elements include "a picture of", "a drawing of", "a painting of", "a close-up of", "gray-scale", "a panoramic view of", "in a stock photo", etc.
* If this element of the prediction it to be expected to be present in almost all images in one or both groups, it should be ignored (same score as if the element was not present in the prediction), unless it is crucial to the difference being described. If it is crucial to the difference being described, it should be evaluated as part of the difference. If it is crucial to the difference being described and no further description of the difference is given, it should be evaluated as a correct description of the difference (score of 2).
* Otherwise, the maximum achievable score is reduced to 1 due to the presence of the unexpected/incorrect element in the prediction.

For example, if Group A is "sailboat" and Group B is "fishing boat", the prediction "a sailboat on the water" should be evaluated based on "a sailboat" and not "on the water", since almost all images of boats are expected to be on the water.
For the same example, if the prediction is just "water", the score should be 0, since "water" is not part of the difference between a sailboat and a fishing boat, and also as the prediction is empty if "water" as an expected element is ignored.
For the same example, the prediction "a sail" should not be ignored since a sail is crucial to the difference between a sailboat and a fishing boat. Furthermore, as no further description of the difference is given in the prediction, the score should be 2 in this case as "sail" implies "sailboat", as almost all images of sailboat are expected to show a sail.
For the same example, the predictions "a stock photo of a sailboat" or "close-up of a sailboat" should be evaluated based on "a sailboat", ignoring "a stock photo of" as this element does not describe the content of the image. Thus, the score should be 2.

### Singular and plural forms

Singular and plural forms should be treated as equivalent, unless the distinction is crucial to the difference being described. This includes numerals, collective nouns, and quantifiers (like "group of people" or "a herd of elephants", "multiple dogs", "three cats" etc.).

For instance, "dog" and "dogs" should be considered the same in the context of evaluating the prediction against the descriptions of Group A and Group B in the above example, as should "four dogs" and "dog". If however, the difference being described specifically involves the number of subjects (e.g., Group A "a dog" vs. Group B "dogs"), then the singular and plural forms should be evaluated accordingly.

### If rules conflict or if it is unclear which rule to apply

The overall goal is always to accurately assess how well the prediction captures the unique aspects of Group A in contrast to Group B, giving a score of 2 for a perfect match, 1 for a partial match, and 0 for no match.
If multiple rules seem to conflict or if it is unclear which rule to apply in a specific situation, prioritize the rule that best serves this overall goal.
Ask yourself the question: "Given the prediction, would a user get a correct understanding of what makes Group A different from Group B?"

# Examples

Group A: \"images of dogs in the snow\" and Group B: \"images of dogs next to cats\". Prediction: \"dogs in winter time\"
Response: 2

Group A: \"images of dogs in the snow\" and Group B: \"images of dogs next to cats\". Prediction: \"animals in the snow\"
Response: 1

Group A: \"apples on trees\" and Group B: \"oranges on trees\". Prediction: \"fruit on trees\"
Response: 1

Group A: \"tea cups\" and Group B: \"coffee mugs\". Prediction: \"table\"
Response: 0

Group A: \"classic cars\" and Group B: \"modern cars\". Prediction: \"vintage vehicles\"
Response: 2

Group A: \"person with backpack\" and Group B: \"person with hat\". Prediction: \"person with umbrella\"
Response: 0

Group A: \"pasta dishes\" and Group B: \"salads\". Prediction: \"Italian food\"
Response: 1

Group A: \"umbrellas in the rain\" and Group B: \"umbrellas indoors\". Prediction: \"people holding umbrellas in the rain\"
Response: 2

"""

    user_prompt = """Again, output either a 2, 1, or 0.
Here are the descriptions:

Group A: \"{gt_a}\" and Group B: \"{gt_b}\". Prediction: \"{hypothesis}\"
Response: """

    def __init__(self, args: Dict):
        self.args = args
        self.llm_hostname = args.get("eval_llm_hostname", "localhost")
        print(f"Initialized evaluator with LLM hostname: {self.llm_hostname}")

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
        assert all(isinstance(hypothesis, str) for hypothesis in hypotheses), "All hypotheses must be strings"
        # verify that the hypothesis is true or false
        scores = []
        evaluated_hypotheses = []
        for hypothesis in tqdm(hypotheses[: self.args["n_hypotheses"]]):
            prompt = self.user_prompt.format(hypothesis=hypothesis, gt_a=gt_a, gt_b=gt_b)
            answer, reasoning = get_llm_output(prompt, self.args["model"], vllm_hostname=self.llm_hostname, think=self.args["think"], greedy=self.args["greedy"], from_choices=("0", "1", "2"), system_prompt=self.system_prompt)
            try:
                if answer == None:
                    scores.append(float('nan'))
                else:
                    scores.append(int(answer))
            except ValueError:
                scores.append(0)

            evaluated_hypotheses.append(
                {"hypothesis": hypothesis, "score": scores[-1], "response": answer, "reasoning": reasoning}
            )

        metrics = {
            "acc@1": scores[0] / 2,
            "acc@5": np.max(scores[:5]) / 2,
            "acc@N": np.max(scores[: self.args["n_hypotheses"]]) / 2,
        }
        return metrics, evaluated_hypotheses


class NullEvaluator:
    def __init__(self, args: Dict):
        self.args = args

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
        return {}, [{}]

class ExportEvaluator:
    """
    Export hypotheses to a jsonl file with the ground truth groups, no evaluation is done.
    Useful to defer evaluation to a later time, e.g. for human evaluation.
    """
    def __init__(self, args: Dict):
        self.args = args

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
        export_path = self.args.get("export_path", "exported_hypotheses.jsonl")
        if self.args.get("n_hypotheses", None) is not None:
            hypotheses = hypotheses[: self.args["n_hypotheses"]]
        with open(export_path, "a") as f:
            for hypothesis in hypotheses:
                f.write(json.dumps({"hypothesis": hypothesis, "group_a": gt_a, "group_b": gt_b}) + "\n")
        print(f"Exported hypotheses to {export_path}")
        return {}, [{}]

class EvaluatorEvaluator:
    """
    Evaluate the evaluator by comparing its scores to human scores in the ranked_hypotheses loaded by the ImportRanker from a jsonl file.
    """
    def __init__(self, args: Dict):
        self.stats_accumulation_file_path = args["stats_accumulation_file_path"]
        self.evaluator = LLMEvaluator(args)

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str, manual_labels: List[int]
    ) -> Tuple[Dict, List[Dict]]:
        assert len(hypotheses) == len(manual_labels)
        metrics, evaluated_hypotheses = self.evaluator.evaluate(hypotheses, gt_a, gt_b)
        assert len(evaluated_hypotheses) == len(hypotheses) and all(evaluated_hypotheses[i]["hypothesis"] == hypotheses[i] for i in range(len(hypotheses))), f"Evaluated hypotheses do not match input hypotheses: {len(evaluated_hypotheses)=} vs {len(hypotheses)=} with {evaluated_hypotheses=} and {hypotheses=}"
        evaluator_scores = [evaluated_hypotheses[i]["score"] for i in range(len(hypotheses))]
        labeled_scores = [manual_labels[i] for i in range(len(hypotheses))]
        for i, hypothesis in enumerate(evaluated_hypotheses):
            hypothesis["manual_label"] = manual_labels[i]
        try:
            with open(self.stats_accumulation_file_path, "r") as f:
                d = json.load(f)
        except FileNotFoundError:
            d = {"evaluator_scores": [], "labeled_scores": [], "accuracy": 0.0, "mean_absolute_error": 0.0}
        d["evaluator_scores"].extend(evaluator_scores)
        d["labeled_scores"].extend(labeled_scores)
        d["accuracy"] = sum(1 for i in range(len(d["evaluator_scores"])) if d["evaluator_scores"][i] == d["labeled_scores"][i]) / len(d["evaluator_scores"])
        d["mean_absolute_error"] = sum(abs(d["evaluator_scores"][i] - d["labeled_scores"][i]) if not math.isnan(d["evaluator_scores"][i]) else (1 if d["labeled_scores"][i] == 1 else 2) for i in range(len(d["evaluator_scores"]))) / len(d["evaluator_scores"])
        with open(self.stats_accumulation_file_path, "w") as f:
            json.dump(d, f)
        return metrics, evaluated_hypotheses
