Adult Income Prediction Model
This repository contains the implementation of a Random Forest Classifier model to predict income levels (<=50K or >50K) using the UCI Adult Income dataset, developed and deployed with Azure Machine Learning (AML). The project leverages Python 3.10 in the azureml_py310_sdkv2 environment and includes data preprocessing, model training, evaluation, and deployment as a web service. Last updated: 02:22 AM EDT on Monday, September 22, 2025.
Overview

Dataset: UCI Adult Income dataset with 14 features (e.g., age, education, occupation) and a binary income target.
Model: Random Forest Classifier with 100 estimators, trained on 80% of the data and validated on 20%.
Environment: Azure ML workspace (azurelabsbyme09), Jupyter notebook, dependencies managed via requirements.txt.
Visualizations: Confusion matrix and ROC curve generated with matplotlib (no seaborn dependency).
Deployment: Model registered as adult_income_model and deployed as adult-income-service using ACI.

Installation
Prerequisites

Python 3.10
Conda or virtual environment manager
Azure ML SDK and related libraries
Git for version control

Setup Instructions

Clone the repository:
bashgit clone https://github.com/ejigsonpeter/adult-income-prediction.git
cd adult-income-prediction

Create and activate the Conda environment:
bashconda create -n azureml_py310_sdkv2 python=3.10
conda activate azureml_py310_sdkv2

Install dependencies:
bashpip install -r requirements.txt
Or using Conda:
bashconda install --file requirements.txt

Install Jupyter kernel:
bashpython -m ipykernel install --user --name azureml_py310_sdkv2 --display-name "azureml_py310_sdkv2"

Configure Azure ML:

Download config.json from your AML workspace (azurelabsbyme09) and place it in the project directory, or use direct authentication with your subscription ID, resource group, and workspace name.



Usage

Launch Jupyter Notebook:
bashjupyter notebook

Open adult_income_model.ipynb and select the azureml_py310_sdkv2 kernel.
Run the notebook cells sequentially to:

Load and preprocess the UCI Adult Income dataset.
Train the Random Forest Classifier.
Evaluate the model with validation metrics.
Deploy the model as a web service.


Check the output for validation metrics, visualizations, and deployment status.

Evaluation Results
The model was evaluated on a validation set, yielding the following metrics:

Accuracy: 0.8508 (85.08%)
Precision: 0.7354 (73.54%)
Recall: 0.6431 (64.31%)
F1 Score: 0.6862 (68.62%)
ROC-AUC: 0.9038 (90.38%)

Analysis

The model achieves a strong overall accuracy of 85.08% and an excellent ROC-AUC of 90.38%, indicating robust classification performance.
Precision (73.54%) and recall (64.31%) suggest a moderate trade-off, with room to improve recall for better identification of >50K cases.
The F1 score (68.62%) balances these metrics, but optimization could enhance positive class performance.

Visualizations

Confusion Matrix: Plotted with matplotlib to show true positives, false positives, etc.
ROC Curve: Displays the model's discriminative power with an AUC of 0.9038.

Deployment

Model Registration: Registered as adult_income_model in the AML workspace.
Service Deployment: Deployed as adult-income-service using ACI configuration.
Scoring Script: score.py handles inference, tested with sample input data.
Endpoint: Accessible via the scoring URI for real-time predictions.

Recommendations

Hyperparameter Tuning: Adjust n_estimators, max_depth, or class_weight to improve recall and F1 score.
Class Balancing: Use oversampling or class_weight='balanced' to address potential imbalance.
Feature Engineering: Explore additional features to enhance model performance.
Further Evaluation: Perform cross-validation or test on a holdout set for stability.

Screenshots

AML Workspace Overview
Model Registration (Models section)
Deployment Settings (Endpoints section)

(Include these as images in the repository, e.g., workspace_overview.png, model_registration.png, deployment_settings.png.)
Download README
To download this README file:

Visit the raw file URL: https://raw.githubusercontent.com/ejigsonpeter/adult-income-prediction/main/README.md
Right-click the link and select "Save As" to download the .md file.
Alternatively, clone the entire repository using the command above to get all files, including this README.

References

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324
Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.). O'Reilly Media.
Microsoft. (2025). Azure Machine Learning documentation. https://docs.microsoft.com/en-us/azure/machine-learning/
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf

Contributing
Feel free to fork this repository, submit issues, or create pull requests for improvements. Ensure you follow the setup instructions and test changes in the Jupyter notebook environment.
License
[\MIT License. ]
Contact
For questions or collaboration, contact ejigsonpeter@gmail.com.