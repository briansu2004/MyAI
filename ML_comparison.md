# ML comparison

- [Strengths and Limitations of Linear Regression](#strengths-and-limitations-of-linear-regression)
  - [Strengths of Linear Regression](#strengths-of-linear-regression)
  - [Limitations of Linear Regression](#limitations-of-linear-regression)
- [Strengths and Limitations of Logistic Regression](#strengths-and-limitations-of-logistic-regression)
  - [Strengths of Logistic Regression](#strengths-of-logistic-regression)
  - [Limitations of Logistic Regression](#limitations-of-logistic-regression)
- [Strengths and Limitations of Decision Trees](#strengths-and-limitations-of-decision-trees)

## Strengths and Limitations of Linear Regression

Being aware of a model's strengths and weaknesses allows us to be mindful of its suitability for addressing specific problems. In the case of Linear Regression:

### Strengths of Linear Regression

- Simplicity: It is easy to comprehend and implement.
- Speed: It has quicker computation than some other models.
- Handles continuous data well: It can model the relationship between continuous features and outputs.

### Limitations of Linear Regression

- Sensitivity to extreme values: A single outlier can significantly alter the model.
- Infers linear relationships: It assumes a simple linear correlation, which might not always hold true in real-world data.
- Cannot model complex patterns: Models that can capture complex data relationships perform better.

When working with the Iris dataset, Linear regression could allow us to predict a feature value like petal length or width using other features. However, its assumptions and sensitivity to outliers could impose limitations.

## Strengths and Limitations of Logistic Regression

Like Linear Regression, Logistic Regression also has its own set of strengths and constraints.

### Strengths of Logistic Regression

- Handles categorical data: It's adept at modeling problems with a categorical target variable.
- Provides probabilities: It helps in understanding the level of certainty of the predictions.
- Offers solid inference: Insights into how each feature affects the target variable can be feasibly deduced.

### Limitations of Logistic Regression

- Inefficient with complex relationships: It has trouble capturing complex patterns.
- Assumes linear decision boundaries: It might not always align with complex dataset structures.
- Unsuitable for continuous outcomes: Due to its probabilistic approach, it does not provide continuous outputs.

With the Iris dataset, Logistic Regression works well for predicting the species of a flower based on its features, as it's a classification task. However, its assumptions about data structures and limitations in handling complex relationships might hinder its performance.

## Strengths and Limitations of Decision Trees

Decision Tree models also have unique abilities and setbacks.

Strengths of Decision Trees:

- Transparent: They are easy to understand and interpret.
- Handles categorical and numerical data: They can conveniently work with a mix of categorical and numerical features.
- Can capture complex patterns: They are capable of fitting highly complex datasets.

Limitations of Decision Trees:

- Prone to overfitting: They might create an overly complex model that does not generalize well.
- Sensitivity to data tweaks: Small changes in data could lead to different trees.
- Biased for the dominating class: If one class outnumbers other classes, the decision tree might create a biased tree.

In the context of the Iris dataset, Decision Trees could show excellent performance as they handle the variation of features well and interpret the relationships between different species and features. However, because of their tendency to overfit and sensitivity to minute changes, one must exercise caution while working with them.
