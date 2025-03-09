# FindDefault (Prediction of Credit Card fraud) - Capstone Project

## Problem Statement:
A credit card is one of the most widely used financial products for online purchases and payments. While credit cards offer convenience in managing finances, they also come with risks. Credit card fraud involves the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash. It is crucial for credit card companies to identify fraudulent transactions to ensure customers are not charged for items they did not purchase.

## About Credit Card Fraud Detection:
- **What is credit card fraud detection?**

Credit card fraud detection encompasses the policies, tools, methodologies, and practices that credit card companies and financial institutions use to combat identity fraud and prevent fraudulent transactions.

In recent years, with the explosion of data and the surge in payment card transactions, credit fraud detection has largely become digitized and automated. Modern solutions primarily utilize artificial intelligence (AI) and machine learning (ML) to manage data analysis, predictive modeling, decision-making, fraud alerts, and remediation activities when instances of credit card fraud are detected. 

- **Anomaly detection**

Anomaly detection involves analyzing vast amounts of data points from both internal and external sources to establish a framework of “normal” activity for each user, thereby identifying regular patterns in their behavior.

Data used to create the user profile includes:

- Purchase history and other historical data

- Location

- Device ID

- IP address

- Payment amount

- Transaction information

When a transaction deviates from the established normal activity, the anomaly detection tool alerts the card issuer and, in some cases, the user. Based on the transaction details and the assigned risk score, these fraud detection systems may flag the transaction for review or place a hold on it until the user verifies their activity.

- **What can be an anomaly?**
  - A sudden increase in spending
  - Purchase of a large ticket item
  - A series of rapid transactions
  - Multiple transactions with the same merchant
  - Transactions that originate in an unusual location or foreign country
  - Transactions that occur at unusual times

  If the anomaly detection tool leverages ML, the models can also be self-learning, meaning that they will constantly gather and analyze new data to update the existing model and provide a more precise scope of acceptable activity for the user.

 
## Project Introduction: 
The dataset contains transactions made by European cardholders in September 2013. This dataset includes transactions over two days, with 492 frauds out of 284,807 transactions, highlighting a highly imbalanced nature where fraudulent transactions account for only 0.172% of all transactions.

In this project, we aim to build a classification model to predict whether a transaction is fraudulent. We will employ various predictive models to assess their accuracy in distinguishing between normal and fraudulent transactions.

## Project Outline:
- **Exploratory Data Analysis:** Analysing and understanding the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations.
- **Data Cleaning:** Checking for the data quality, handling the missing values and outliers in the data.
- **Dealing with Imbalanced data:** This data set is highly imbalanced. The data should be balanced using the appropriate Resampling Techniques (NearMiss Undersampling, SMOTETomek) before moving onto model building.
- **Feature Engineering:** Transforming the existing features for better performance of the ML Models. 
- **Model Training:** Splitting the data into train & test sets and use the train set to estimate the best model parameters.
- **Model Validation:** Evaluating the performance of the models on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting.
- **Model Selection:** Choosing the most appropriate model that can be used for this project.
- **Model Deployment:** Model deployment is the process of making a trained machine learning model available for use in a production environment.

## Project Work Overview:
Our dataset exhibits significant class imbalance, with the majority of transactions being non-fraudulent (99.82%). This presents a challenge for predictive modeling, as algorithms may struggle to accurately detect fraudulent transactions amidst the overwhelming number of legitimate ones. To address this issue, we employed various techniques such as undersampling, oversampling, and synthetic data generation.
1. **Undersampling:** We utilized the NearMiss technique to balance the class distribution by reducing the number of instances of non-fraudulent transactions to match that of fraudulent transactions. This approach helped in mitigating the effects of class imbalance. Our attempt to address class imbalance using the NearMiss technique did not yield satisfactory results. Despite its intention to balance the class distribution, the model's performance was suboptimal. This could be attributed to the loss of valuable information due to the drastic reduction in the majority class instances, leading to a less representative dataset. As a result, the model may have struggled to capture the intricacies of the underlying patterns in the data, ultimately affecting its ability to accurately classify fraudulent transactions.
2. **Oversampling:** To further augment the minority class, we applied the SMOTETomek method with a sampling strategy of 0.75. This resulted in a more balanced dataset, enabling the models to better capture the underlying patterns in fraudulent transactions.
3. **Machine Learning Models:** After preprocessing and balancing the dataset, we trained several machine learning models, including:

   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier
   - AdaBoost Classifier
   - XGBoost Classifier
4. **Evaluation Metrics:** We evaluated the performance of each model using various metrics such as accuracy, precision, recall, and F1-score. Additionally, we employed techniques like cross-validation and hyperparameter tuning to optimize the models' performance.
5. **Model Selection:** Among the various models and balancing methods experimented with, the XGBoost model stands out as the top performer when using oversampling techniques. Despite the inherent challenges posed by imbalanced datasets, the XGBoost algorithm demonstrates robustness and effectiveness in capturing the underlying patterns associated with fraudulent transactions. By generating synthetic instances of the minority class through oversampling methods like SMOTETomek, the XGBoost model achieves a more balanced representation of the data, enabling it to learn and generalize better to unseen instances. This superior performance underscores the importance of leveraging advanced ensemble techniques like XGBoost, particularly in the context of imbalanced datasets characteristic of credit card fraud detection.
   
In summary, our approach involved preprocessing the imbalanced dataset using undersampling and oversampling techniques, followed by training and evaluating multiple machine learning models. By systematically exploring different methodologies and algorithms, we aimed to develop robust fraud detection XGBoost model capable of accurately identifying fraudulent transactions while minimizing false positives.

## Future Work:
Anomaly detection techniques, including isolation forests and autoencoders, offer specialized capabilities for identifying outliers and anomalies within datasets. By incorporating these methods alongside traditional classification approaches, we can enhance the effectiveness of fraud detection systems. Isolation forests excel at isolating anomalies by randomly partitioning data points, making them particularly useful for detecting fraudulent transactions that deviate from normal patterns. Autoencoders, on the other hand, leverage neural networks to reconstruct input data, effectively learning representations of normal behavior and flagging deviations as potential anomalies.

Exploring the integration of advanced deep learning models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) alongside traditional machine learning techniques holds significant promise for enhancing fraud detection systems. These neural network architectures offer unique capabilities for processing sequential and structured data, which are crucial in identifying anomalous patterns indicative of fraudulent activities. By leveraging CNNs and RNNs, alongside hybrid models that combine the strengths of both deep learning and traditional algorithms, we can improve accuracy, adaptability, and overall performance in fraud detection. Additionally, techniques such as unsupervised learning, transfer learning, and feature extraction through deep learning can further enhance the efficiency and effectiveness of fraud detection systems. Through these advancements, we aim to bolster our ability to detect and prevent fraudulent transactions, ultimately safeguarding financial systems and protecting consumers from financial losses.


   

