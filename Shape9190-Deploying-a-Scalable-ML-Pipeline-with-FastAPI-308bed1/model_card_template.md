# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A modified model. A machine learning model of Census Data. Using scikit-learn hyperpameters, trained with scikit-learn, and RandomForestClassifier model type.

## Intended Use
Model should be used to predict if an individuall's annual income is more or less than $50,000.

## Training Data
Training data consists of 80% of total data

## Evaluation Data
Testing was performed on the remaining 20% of the total data

## Metrics
The metrics used were the precision score, recall score, and Fbeta score.

The datasets results:
Precision: 0.8717 | Recall: 0.7200 | F1: 0.7886

## Ethical Considerations
Risk: The dataset may present an ethical risk of unproportionate category values that may sway the results. 

## Caveats and Recommendations
Recommendation of multiple testing rounds to best produce more accurate metric results.
