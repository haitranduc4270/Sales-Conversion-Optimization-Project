# Sales Conversion Optimization Project

## Project Description

### Overview

The objective of this project was to automate sales conversions for an anonymous organization's social media advertising campaign. The goal is to optimize transformations using a structured pipeline and predictive models.

### 1. Data Exploration and Preprocessing

#### Data Description

The dataset, [conversion_data.csv](https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking/data), includes 1143 observations across 11 variables, detailed as follows:

- ad_id: Unique ID for each ad.
- xyz_campaign_id: ID associated with each ad campaign of XYZ company.
- fb_campaign_id: ID for how Facebook tracks each campaign.
- age: Age of the person to whom the ad is shown.
- gender: Gender of the person to whom the ad is shown.
- interest: Code specifying the category of the person’s interests (based on Facebook profile).
- Impressions: Number of times the ad was shown.
- Clicks: Number of clicks on the ad.
- Spent: Amount paid by company XYZ to Facebook for the ad.
- Total conversion: Total number of inquiries about the product after seeing the ad.
- Approved conversion: Total number of product purchases after seeing the ad.
![Alt text](DE.png)


#### Data Cleaning and Preprocessing

Cleaned and preprocessed the dataset for analysis, identifying and handling missing or inconsistent data. Used the Facade design pattern for data preprocessing:

```python
# Facade design pattern for data preprocessing
class DataPreProcessStrategy(DataStrategy):
    # ... (Code snippet provided in the document)

class DataDivideStrategy(DataStrategy):
    # ... (Code snippet provided in the document)

class DataCleaning:
    # ... (Code snippet provided in the document)
```

### 2. Exploratory Data Analysis (EDA)

Performed statistical analysis to understand distributions and relationships, visualized key metrics, and trends in the data. Various exploratory data analysis techniques were employed, including Box plot, Pair plot, and Correlation Matrix heatmap.

### 3. Feature Engineering

Implemented feature engineering to create new features that might improve model performance.

### 4. Model Development

Developed machine learning models, including GradientBoostingModel, LinearRegressionModel, AdaBoostRegressorModel, RandomForestRegressorModel, and planned to add more models over time. Evaluated model performance using metrics like R2 Score, RMSE, etc.

### 5. MLOps Integration

#### Version Control and Collaboration

Utilized Git and GitHub for version control and collaboration, enabling multiple team members to work on the project simultaneously. This streamlined collaboration, enhanced version control, and provided a centralized platform for project management.

![Git1](Git1 pic link)
![Git2](Git2 pic link)

#### Automated Training and Deployment Pipelines

Streamlined the model development process using ZenML and MLflow for automated training and deployment pipelines. ZenML organized and managed machine learning workflows, while MLflow automated model training and deployment steps. This setup facilitated experimentation with different models, tracking their performance, and deploying the best-performing ones effortlessly.

![ZenML Dashboard](RUN pic link)
![ZenML DAG Visualizer](1pic link)
![MLflow Artifacts](MLflow pic link)

#### Communication and Project Organization

Utilized Discord for seamless and real-time team interaction and Notion for creating a structured timeline and tracking project-related tasks. This combination ensured effective communication and project organization within the team.

![Discord](Discord pic link)
![Notion Roles](Role pic link)
![Notion Timeline](Timeline pic link)

This combination of Discord for communication and Notion for project organization ensured a smooth collaborative environment, keeping everyone informed and organized throughout the development process.