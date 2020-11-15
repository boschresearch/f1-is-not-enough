# F1 is Not Enough
This repository contains the code of the models and evaluation scores as well as the collected study data of our EMNLP 2020 paper [F1 is Not Enough! Models and Evaluation Towards User-Centered Explainable Question Answering](https://www.aclweb.org/anthology/2020.emnlp-main.575/).

## Evaluation Scores
Coming soon.

## Models
Coming soon.

## User Study Data
We provide the detailed data collected from 40 participants in the human evaluation.
The data can be found at [user_study_data/data.jsonl](user_study_data/data.jsonl) and  loaded using
```python
import json, jsonlines
with jsonlines.open('user_study_data/data.jsonl') as reader:
    data = [e for e in reader]

# A look at the first trial
print(json.dumps(data[0], indent=4))
```
which prints the data of the first trial:
```
{
    "subject_id": 6548563143128909446,
    "condition": "Select & Forget",
    "per_question_ratings": {
        "question_ids": ["5ae0ae4555429945ae959419", ..., "5a82efe355429966c78a6aa1"],
        "certainty_ratings": ["strong_agree", ..., "neutral"],
        "completion_times(ms)": [67659, ..., 69649],
        "answer_correct_ratings": ["yes", ..., "yes"],
        "answer_known_before": ["no", ..., "no"],
        "helpfulness_ratings": ["agree", ..., "neutral"]
    },
    "post_questionnaire_ratings": {
        "trust": "neutral",
        "enough_details": "agree",
        "irrelevant_details": "agree",
        "satisfied": "agree"
    }
}

​
```

## License
The user study data in the folder [user_study_data](user_study_data/) is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0).

## Citation
If you use our models, scores or data in your work, please cite [our paper](https://www.aclweb.org/anthology/2020.emnlp-main.575/):
```
@inproceedings{schuff-etal-2020-f1,
    title = "F1 Is Not Enough! Models and Evaluation towards User-Centered Explainable Question Answering",
    author = "Schuff, Hendrik  and
      Adel, Heike  and
      Vu, Ngoc Thang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.575",
    pages = "7076--7095",
}
```

