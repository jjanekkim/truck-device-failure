# Truck Device Failure Classification

## Summary

This project was one of my early data science boot camp projects. The goal was to address class imbalance and build a machine-learning model with both precision and recall scores above 60%. A key challenge in the project was that the feature names were anonymized, making it difficult to understand their context or purpose.

During my initial analysis, I realized I had missed a crucial step—sorting the features into their correct data types. To establish a baseline, I classified features as categorical if they had around 100 unique values and as numerical if they had over 10,000. However, there was one feature with about 500 unique values that I wasn’t sure how to categorize, so I initially treated it as numerical.

After organizing the features, I used a correlation map to detect identical features and removed one of the duplicates. Next, I moved on to feature engineering. I extracted the 'month' from the 'date' feature and created two new features: 'quarter' and 'work_day' (which counted the number of days the device was operational).

Once feature engineering was complete, I applied a log transformation to features with high skewness, scaled the numerical data using a MinMax scaler, and one-hot encoded the categorical data. To address the class imbalance, I chose under-sampling due to computational constraints and time efficiency.

I selected five models for training: Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and CatBoost. The results of the model evaluation are shown below.

![image](https://github.com/user-attachments/assets/bc1433b2-084c-4e4c-9c65-6489ba67bef7)

The best-performing model was the RandomForestClassifier, which was surprising since XGBoost typically outperforms other models in most cases. To further analyze the results, I created a confusion matrix for the RandomForest model.

![image](https://github.com/user-attachments/assets/cc1287cc-933b-4f16-8fbd-bc3a140e1f30)

There was one false positive and six false negatives. Since this model predicts truck device failures, reducing the number of false negatives is crucial, as it is more costly to detect a failure after it has already occurred than to re-check a device that is functioning correctly.

The important features identified by the RandomForestClassifier were as follows:

![image](https://github.com/user-attachments/assets/f1ae02db-44e1-45a8-8135-cbb73a5139c8)

From the top 10 feature importance chart, I noticed that features I created, like 'work_day' and 'quarter,' were determined to be significant.

This project was especially meaningful because it taught me the importance of correctly sorting features into numerical and categorical types, particularly when feature names are anonymized. This experience gave me the confidence to work with large, imbalanced datasets, knowing that there are always ways to improve the model by correctly classifying data types and generating meaningful features.
