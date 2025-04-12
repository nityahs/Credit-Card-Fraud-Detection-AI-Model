# Credit-Card-Fraud-Detection-AI-Model
## System Architechture
![Sys Arch 1](https://github.com/user-attachments/assets/7835a72c-67c4-4e3b-9f9f-cd73282e5bbc)
### Data Ingestion
The dataset typically includes several fields: 
● Transaction Time: A time index relative to the first transaction. 
● Transaction Amount: The monetary value of the transaction. 
● Merchant Details: Information about where the transaction occurred. 
● Category: Type of transaction or purchase category. 
● Label: A binary indicator where 0 signifies a legitimate transaction and 1 represents a 
fraudulent transaction. 
The ingestion module utilizes Python's pandas library to load the dataset into a DataFrame—a 
powerful twodimensional data structure. This format provides high flexibility for slicing, 
filtering, aggregating, and transforming the data. Logging mechanisms are built into this stage to 
track whether the data has been ingested correctly. They record critical information like file read 
times, number of rows ingested, schema conformity, and missing fields. In the case of corrupted 
files or incompatible formats, the system is equipped to raise exceptions and halt downstream 
processing, thus ensuring data integrity.

Dataset used - https://www.kaggle.com/datasets/kartik2112/fraud-detection/code
### Data PreProcessing
Preprocessing is the transformation phase where raw data is cleaned, curated, and made ready for 
machine learning algorithms. Poor quality or inconsistent data can significantly degrade model 
performance, making this stage essential. This step includes several subprocesses: 
1. Dropping Redundant Columns 
Data often contains information irrelevant to the prediction task. In credit card transactions, 
fields like customer names, email addresses, street locations, or even detailed merchant 
descriptors can be considered redundant. These fields are removed for multiple reasons: 
● Privacy: Reduces the risk of exposing Personally Identifiable Information (PII).
● Noise Reduction: Eliminates irrelevant features that may distract the model.
● Dimensionality Reduction: Fewer features often lead to simpler and more interpretable 
models.
### Handling Missing Values 
Missing data is a frequent occurrence in real world datasets and can arise from several factors 
such as system errors, manual entry mistakes, or incomplete integration pipelines. 
However, in the context of our project, we were fortunate to work with a dataset that was 
remarkably clean and well-prepared, with no missing values present across any of the features. 
This eliminated the need for additional imputation or data removal strategies, thereby 
simplifying the preprocessing pipeline and allowing us to focus more directly on encoding, 
normalization, and model development.  
3. Encoding Categorical Variables 
Most machine learning algorithms require numerical inputs, but transactional data often includes 
categorical variables such as gender, merchant category, or location. To make these features 
usable: 
● Label Encoding assigns a unique integer to each category. For example, “Grocery” = 0, 
“Electronics” = 1, “Fuel” = 2, etc. 
● OneHot Encoding is also considered in some cases, especially for non ordinal 
categories. It creates binary vectors for each category but can lead to a large number of 
dimensions. 
Choosing the appropriate encoding method depends on the algorithm and the volume of 
categories. 
4. Normalization 
Normalization ensures that numerical features are brought onto a comparable scale. This is 
especially critical for algorithms like Logistic Regression, which are sensitive to the magnitude 
of feature values. In this system: 
● StandardScaler from scikitlearn is used, which transforms features to have a mean of 0 
and a standard deviation of 1. This transformation helps the learning algorithms converge 
faster and avoids bias towards features with higher absolute values. 
### Model Training 
Model training is the core computational stage where balanced and preprocessed data is used to 
build predictive models. This project explores multiple algorithms to identify which one yields 
the best performance in terms of fraud detection. Each model is encapsulated in its own training 
pipeline. Below are the architectures used: 
1. Logistic Regression 
Logistic Regression is a fundamental linear model used for binary classification problems. Its 
architecture includes: 
● Linear Decision Boundary: Computes a weighted sum of input features. 
● Sigmoid Function: Converts output to a probability score between 0 and 1. 
● Loss Function: Optimizes the model using Binary CrossEntropy or Log Loss. 
● Regularization: L2 regularization (Ridge) is added to penalize large coefficients, helping 
to reduce overfitting. 
Despite its simplicity, Logistic Regression is highly interpretable and often provides competitive 
baselines. It is especially useful when model transparency is a priority. 
2. Random Forest 
Random Forest is an ensemble method based on decision trees. Its architecture comprises: 
● Bootstrap Sampling: Each tree is trained on a random sample of the data. 
● Feature Bagging: Only a subset of features is used at each split, increasing diversity 
among trees. 
● Majority Voting: Final classification is based on the most frequent output from 
individual trees. 
Random Forests are robust to overfitting and handle nonlinearities well. Additionally, they 
provide feature importance scores, helping analysts understand which features are most 
predictive of fraud. 
3. Naive Bayes 
Naive Bayes is a probabilistic model rooted in Bayes' theorem. Its key features include: 
● Conditional Independence: Assumes all features are independent given the target class, 
which simplifies computation. 
● Gaussian Naive Bayes: For continuous variables, it assumes data follows a normal 
distribution. 
● Multinomial/Bernoulli Variants: Useful for categorical or binary features. 
Naive Bayes is extremely fast and performs well in highdimensional spaces, making it suitable 
for realtime inference scenarios despite its strong assumptions. 
### Evaluation 
Once the models are trained, they are evaluated to measure performance. Evaluation is 
performed on a separate test set not seen during training, ensuring unbiased results. The 
following metrics are considered: 
1. Accuracy 
While accuracy is the most common metric, it is less informative in imbalanced datasets. For 
example, if 99.8% of transactions are legitimate, a model that always predicts “legitimate” would 
have 99.8% accuracy but detect no fraud at all. 
2. Precision and Recall 
● Precision: The ratio of true positives (correctly identified frauds) to all positive 
predictions. High precision means fewer false alarms. 
● Recall (Sensitivity): The ratio of true positives to all actual fraud cases. High recall 
means fewer missed frauds. 
In fraud detection, recall is often more critical than precision, since missing a fraud can be more 
costly than investigating a false alarm. 
3. F1 Score 
F1 Score is the harmonic mean of precision and recall, providing a balanced measure especially 
useful when classes are imbalanced. It penalizes extreme values of either metric, offering a more 
nuanced view of model performance. 
4. ROC-AUC 
The Receiver Operating Characteristic (ROC) curve plots True Positive Rate vs. False Positive 
Rate at different thresholds. The Area Under the Curve (AUC) summarizes this curve into a 
single number, with values closer to 1 indicating better performance. ROCAUC is 
thresholdindependent, making it ideal for comparing models. 
