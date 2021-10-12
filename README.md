# Hybrid CTC

## Setup

```shell
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run on IAM

```shell
source venv/bin/activate
python run_iam.py --model=model/best_lm=0.5_beams=5/serve --data=data/test.lst
```

## Shared Code

In `src` we share our implementations of the beam-search and the CTC-prefix-scores.
Note, that the full algorithm is build in the Tensorflow graph which is why a simple `model.predict` is sufficient to obtain the decoded sequence.

