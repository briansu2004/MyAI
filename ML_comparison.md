# ML comparison

- [Strengths and Limitations of Linear Regression](#strengths-and-limitations-of-linear-regression)
  - [Strengths of Linear Regression](#strengths-of-linear-regression)
  - [Limitations of Linear Regression](#limitations-of-linear-regression)
- [Strengths and Limitations of Logistic Regression](#strengths-and-limitations-of-logistic-regression)
  - [Strengths of Logistic Regression](#strengths-of-logistic-regression)
  - [Limitations of Logistic Regression](#limitations-of-logistic-regression)

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
