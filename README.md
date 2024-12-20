# AI-ML
## Artificial Intelligence
**Creation of Intelligent Machines.**
- Programming machines to exhibit intelligence.
- Machine identifying patterns and learning by itself.
### Programming machines Examples:
**Knowledge-based AI**\
Expert Systems --> AI Suggesting what to do based on our questions and answers, like a doctor.\
Reasoning.\
**Machine Exhibiting Intelligence** in Uncertain Environments like predicting weather conditions.\
**Search-based AI** --> AI in Google Maps or Solving Puzzles.
## Machine Learning
**A subfield of AI that gives computers the ability to learn itself by identifying patterns from the Data we provide using different ML Algorithms without being explicitly programmed**.

The best example for this is ***Personalized Suggestions, Targeted Advertisements*** in ecommerce websites or social media based on the user likes and interests, Voice recognization and Face recognization also good examples of ML.
## Deep Learning
Deep Learning is part of ML, it deals with huge volumes of data to identify the underlying patterns in it.\
It is also called as Artificial Neural Networks.
- Artificial Neural Networks are based on the ideas of how the human brain works.

Tryout the Google [Teachable Machine](https://teachablemachine.withgoogle.com/) to understand how ML Software takes data and create a program to give the predicted output using prediction function.
## Data Science
Data Science is the science of **extracting knowledge and insights from data**.
### Data for ML
**Data** --> Any information that is stored is called Data.
- Image, Text file, Video, Emails, Messages, etc.

**Machine Learning have certain Algorithms, when these Algorithms written as code is called ML Software**.

Training
- Data --> ML Software --> Intelligent Program (or) ML Model (or) Prediction function.

Prediction
- New Inputs --> Prediction Function --> Predicted Output

“Prediction” refers to the output of an algorithm after it has been trained on a given old dataset and applied to new data when predicting the possibility of a particular outcome.

**Types of Data**
- Structured Data
- Unstructured Data
- Labeled Data
- Unlabeled Data

**Structured Data**
- Any data that can be displayed in rows and columns.
- Easy to analyze, Ex: sensors data, app data, network data, e-commerce data, etc.
- Each row is called as an instance (or) an example (or) a sample.
- Each column is called as Feature (or) Attribute.

**Unstructured Data**
- Images, audio, video, text,...
- Can't be analyzed using regular methods used for structured data.
- 80% of enterprise data is unstructured.
- To make computations easier, unstructured data is transformed into ***matrices***.

**Labeled data**
- Data labeling is the process of identifying raw data (images, text files, videos, etc.) and adding one or more meaningful and informative labels to provide context so that a machine learning model can learn from it. (or) simply, Labeled data is carefully labelled with meaningful tags, or labels, that classify the data's elements or outcomes.
- For example, in a dataset of emails, each email might be labeled as "spam" or "not spam." These labels then provide a clear guide for a machine learning algorithm to learn from.
- The column which has to be predicted is called as Target Feature (or) Target Attribute (or) Target Variable (or) Target Label.
### Types of ML
**Supervised Learning**
- Learning a **Prediction Function** that **maps** *Input* with *Output* using labelled Data.
- **Supervised learning** is a type of machine learning where an algorithm learns from labeled training data to make predictions or decisions. The model is trained on input-output pairs, and its goal is to map input data to the correct output based on the provided examples. This enables the model to generalize its learning to make predictions on new, unseen data.
- Example: Youtube or Instagram recommending shorts or reels based on the data of your previously watched and liked shorts and reels to make you keep watching and spending more time on it so that they will benefit and you will waste your time and ruin your future.
- Supervised Learning is two types --> **Classification** and **Regression**.
- **Classification**: When a Target Feature is *Discrete valued* which means it can be countable.<br/> Ex: Like/Dislike, Yes/No and Image recognization: cat, dog, lion etc.
- **Regression**: When a Target Feature is *Continuous valued*<br/>
Ex: continuously increasing and decreasing of prices of products.
- Supervised learning requires at least one input attribute in the data and at least one output attribute in the data.
- It is usually hard to collect large amount of labelled data. 
- There could be one or multiple targets as the desired output.
- Ex: Spam filtering, Speech to Text, Machine Translation, Online Advertising.

**Unsupervised Learning**
- It is a type of machine learning that looks for patterns in a data set with no pre-existing labels.
- **Clustering**: Grouping similar data or information together. Examples include k-means clustering and hierarchical clustering. A very known example is *Google Photos* grouping similar images.
- **Dimensionality Reduction**: Reducing the number of features or variables in the data while retaining important information. Principal Component Analysis (PCA) is an example.
- **Association Rules**: Discovering relationships or associations between variables. Apriori algorithm for mining association rules is an example.<br/>
Ex: When we search about a sci-fi movie in google it will also recommend other sci-fi movies under "people also watch" (or) in flipcart when we try to buy a mobile it will recommend pouch and screen glass under 'frequently bought together' this is because a lot of users or customers watching those recommend movies or buying those required products together based on that it will start recommending.

**Semi-Supervised Learning**
- Semi-supervised learning is a type of machine learning that falls between supervised learning and unsupervised learning. In semi-supervised learning, the algorithm is trained on a dataset that contains both labeled and unlabeled data. The presence of labeled data provides better guidance for learning, while the unlabeled data allows the algorithm to explore and discover additional patterns or structures within the data.
- Semi-supervised learning is frequently applied to classification tasks, where the algorithm learns to predict the labels of unlabeled data points based on the patterns it has identified in the labeled data.
- One common approach in semi-supervised learning is to use the labeled data to train a model initially and then use the trained model to predict labels for the unlabeled data. These predicted labels are then treated as if they were true labels, and the model is further refined using this ***pseudo-labeled data***.
- In the end we will train the model with ***labeled + pseudo-labeled data*** to get a better model.
- Ex: Natural language processing, image recognition, and healthcare.

**Reinforcement Learning**
- Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent learns to achieve a goal in an complex environment by receiving feedback in the form of rewards or punishments. The primary focus of reinforcement learning is on learning best sequences of actions to maximize cumulative rewards over time.
- **Agent**: The entity or system that takes actions within an environment. The agent is the learner in the RL framework.
- **Environment**: The external system with which the agent interacts. It represents the context or surroundings in which the agent operates.
- **Action**: The set of possible moves or decisions that the agent can take in a given state. Actions are the choices available to the agent.
- **Reward**: A numerical value that the environment provides to the agent as feedback after it takes an action in a particular state. The reward indicates the immediate benefit or cost associated with the action.
- The learning process in reinforcement learning typically involves the agent interacting with the environment, receiving feedback in the form of rewards, updating its knowledge or policy based on this feedback, and iteratively improving its decision-making capabilities.
- Ex: Robotics, game playing, autonomous systems, and optimization problems. Notable algorithms in reinforcement learning include Q-learning, Deep Q Networks (DQN), and Policy Gradient methods.
## ML Project Workflow
### Workflow of Supervised Learning
Why choosing workflow of Supervised Learning?
- A lot of applications thesedays are in the field or subfield of Supervised Learning.
- Most of the things would be similar in all the types of ML so understanding the workflow of Supervised Learning will also helps in understanding the workflow of others like SSL, UL, RL.

Workflow of Supervised Learning Classified into different steps, different people classified it into different no. of steps but widely accepted one is 4 major steps which are
- Domain Problem to ML Problem
- Data Collection & Preparation
- Training
- Deployment
### Domain Problem to ML Problem
Translating a domain problem to an ML problem involves converting a real-world challenge or goal into a specific question that machine learning can answer. It's about defining the problem in a way that allows us to use data and algorithms to find a solution or make predictions. This step clarifies what we want to achieve and sets the stage for applying machine learning techniques to address the problem effectively.

To Understand the Domain we need to ***Talk to domain experts***, Understand more about ***available data & constraints of the problem*** and make sure that ***ML can solve the problem***.

Ex: Domain - User Engagement | Domain Problem - Making Users spend more time on video platform | ML Problem - User Info + Video Info --> Likes?
### Data Collection & Preparation
There are 3 steps 1.Data Collection, 2.Data Preprocessing, 3.Exploratory Data Analysis & Feature Engineering.

**Data Collection** 
- Collecting various data sources and steps to extract data.
- Data can be collected from Sales team, Marketing Team and Other Sources from all these we will have Raw Data and we need to specify the Target Feature.

In Supervised Learning the labeled data can be acquired from
- Manual Labeling
- Observing Behaviours
- Downloading from Websites
- Data Labeling Services & Tools

**Data Preprocessing**
- Integrating the data from various sources.
- Cleaning the data from ***Duplicates & Outliers*** and ***Handling missing values***.

**Exploratory Data Analysis**
- Exploratory Data Analysis is the process of visually and statistically summarizing, interpreting, and understanding the main characteristics of a dataset. The goal is to gain insights into the data, identify patterns, detect anomalies, and inform subsequent steps in the data analysis or machine learning process.
- EDA is an essential step before building machine learning models as it helps practitioners understand the nature of the data they are working with.
- It is used to understand the relationship between different features in the data.
- Tools that we can use are matplotlib, Pandas, Amazon Glue.
- For example in a Users data understanding relations like certain age group might like certain type of advertisements.

**Feature Engineering**
- Feature Engineering is the process of transforming raw data into a format that can better represent the underlying patterns of the problem to the machine learning algorithms. It involves creating new features or modifying existing ones to improve the model's performance.
- Effective feature engineering can significantly impact the performance of machine learning models, making them more accurate and robust. It requires a deep understanding of the data and the problem domain.
- For example in Users data there will be features like name, age, phone number, gender or city in all these features only gender, city and age will be useful to predict which kind of advertisements should show to users this selecting the relevent features and dropping the irrelevent features happen in Feature Engineering
### Training
**Choosing ML Algorithm**

We choose an ML Algorithm based on Influencing Factors
- ML Problem
- Amount of Data
- Accuracy
- Scalability

A few well known ML Algorithms are
- Neural Networks
- Decision Trees 
- Support Vector Machine(SVM)
- Naive Bayes
- k-NN
- Random Forests
- Logistic Regression and many more....

**Training the Model**

After choosing the ML Algorithm, we will convert this algorithm into ML Software which means we will code this algorithm, when coding this algorithm we use tools like
- Pandas
- TensorFlow
- Scikit Learn
- Keras
- PyTorch

with these tools or libraries we don't have to write code for ML Software from scratch.\
And this ML Software with analysing the given data for patterns will give a Model as an output

**Model Evaluation**

After we got Model we will split the data into *Training Data*, *Validation Data* and *Testing Data*.\
To Evaluate the Model we have Evaluation Metrics which are Accuracy, Precision, Recall, F1 score and many more.....

**Hyperparameter Tuning**

Hyperparameter: Every ML algorithm has a set of parameters which are used to define and control the training process.

With Hyperparameters and Training Data we do the training then we will get the Model and we will ***Evaluate*** the Model with Validation Data then we will get the Performance Metric which are Accuracy, Precision etc of the Model.This Process will continue with changing Hyperparameters everytime with changing the Algorithm until we get desired Performance Metric.

What actually happens in the Evaluation process is, in a set of data we have val_x and actual_val_y, now using the val_x with Model we get the predicted_val_y then we compare this predicted_val_y with actual_val_y which is called Evaluation.

(Hyperparameters + Training set)Training --> Model | (Model + Validation set)Evaluation --> Performance Metrics and this cycle will go on until we get desired outcome.

**Training Final Model**
- Here we merge the Training Data & Validation Data and do the training with hyperparameters them we will get the Final Model.
- With Final Model we will do the Evaluation with Testing Data and we will get the final Performence Metrics and the Model is ready to deploy.
### Deployment
Deployment is the process of making software available to be used on a system by users and other programs.

Interpreting the Results

**Explainable AI**
- In some scenarios, it becomes very important to be able to ***explain the predictions*** made by ML Models.

**Packaging and Shipping**
- Deployed Model + Real World Inputs --> Real Life Application --> Predictions in Real World

**Maintenance and Monitoring**
- Review Model's Results
- Re-train the Model
- Sometimes, we might have to even change the Algorithm
### Python Libraries
- [Numpy](https://github.com/SspStark/NumPy-and-Colab)
### Algorithms
- [k-NN Algorithm](https://github.com/SspStark/k-NN-Algorithm)
