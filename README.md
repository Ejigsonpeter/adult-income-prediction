# Adult Income Prediction Model Using Azure

This repository contains the implementation of a **Random Forest Classifier** model to predict income levels (`<=50K` or `>50K`) using the **UCI Adult Income dataset**. The project is developed and deployed with **Azure Machine Learning (AML)**.  

It leverages **Python 3.10** in the `azureml_py310_sdkv2` environment and includes:
- Data preprocessing
- Model training & evaluation
- Deployment as a web service  

📅 *Last updated: 02:22 AM EDT on Monday, September 22, 2025.*

---

## 📌 Overview
- **Dataset**: UCI Adult Income dataset with 14 features (e.g., age, education, occupation) and a binary income target.  
- **Model**: Random Forest Classifier with 100 estimators, trained on 80% of the data and validated on 20%.  
- **Environment**: Azure ML workspace (`azurelabsbyme09`), Jupyter notebook, dependencies managed via `requirements.txt`.  
- **Visualizations**: Confusion matrix and ROC curve generated with `matplotlib` (no `seaborn`).  
- **Deployment**: Model registered as `adult_income_model` and deployed as `adult-income-service` using ACI.  

---

## ⚙️ Installation

### Prerequisites
- Python 3.10  
- Conda or virtual environment manager  
- Azure ML SDK and related libraries  
- Git for version control  

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/ejigsonpeter/adult-income-prediction.git
cd adult-income-prediction
```

```bash
# Create and activate the Conda environment
conda create -n azureml_py310_sdkv2 python=3.10
conda activate azureml_py310_sdkv2
```

```bash
# Install dependencies
pip install -r requirements.txt
# Or using Conda
conda install --file requirements.txt
```

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name azureml_py310_sdkv2 --display-name "azureml_py310_sdkv2"
```

**Configure Azure ML**  
- Download `config.json` from your AML workspace (`azurelabsbyme09`) and place it in the project directory  
- Or authenticate directly with subscription ID, resource group, and workspace name  

---

## 🚀 Usage
```bash
# Launch Jupyter Notebook
jupyter notebook
```

1. Open `adult_income_model.ipynb`  
2. Select the `azureml_py310_sdkv2` kernel  
3. Run cells sequentially to:  
   - Load & preprocess dataset  
   - Train Random Forest Classifier  
   - Evaluate with validation metrics  
   - Deploy model as a web service  

✅ Check outputs for validation metrics, plots, and deployment status.  

---

## 📊 Evaluation Results
**Validation metrics:**
- Accuracy: **85.08%**
- Precision: **73.54%**
- Recall: **64.31%**
- F1 Score: **68.62%**
- ROC-AUC: **90.38%**

### Analysis
- Strong overall accuracy (**85.08%**) and excellent ROC-AUC (**90.38%**)  
- Precision (73.54%) and recall (64.31%) indicate moderate trade-off  
- F1 score (68.62%) shows balance, but recall could be improved for better detection of `>50K` cases  

---

## 📈 Visualizations
- **Confusion Matrix**: Shows true vs. false predictions  
- **ROC Curve**: AUC = 0.9038 (strong discriminative ability)  

---

## ☁️ Deployment
- **Model Registration**: Registered as `adult_income_model` in AML  
- **Service Deployment**: Deployed as `adult-income-service` with ACI  
- **Scoring Script**: `score.py` handles inference, tested with sample input data  
- **Endpoint**: Accessible via scoring URI for real-time predictions  

---

## 🔧 Recommendations
- **Hyperparameter Tuning**: Adjust `n_estimators`, `max_depth`, `class_weight`  
- **Class Balancing**: Use oversampling or `class_weight='balanced'`  
- **Feature Engineering**: Explore additional derived features  
- **Further Evaluation**: Cross-validation or test on holdout set  

---

## 🖼️ Screenshots
*(Add these images to the repository and reference them here)*  
- AML Workspace Overview (`workspace_overview.png`)  
- Model Registration (`model_registration.png`)  
- Deployment Settings (`deployment_settings.png`)  

---

## 📥 Download README
- View raw file: [README.md](https://raw.githubusercontent.com/ejigsonpeter/adult-income-prediction/main/README.md)  
- Or clone the repository to access all files  

---

## 📚 References
- Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5-32. [DOI](https://doi.org/10.1023/A:1010933404324)  
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly Media.  
- Microsoft. (2025). *Azure Machine Learning documentation.* [Docs](https://docs.microsoft.com/en-us/azure/machine-learning/)  
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830. [PDF](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)  

---

## 🤝 Contributing
Feel free to fork this repository, submit issues, or create pull requests for improvements.  
Ensure you follow the setup instructions and test changes in the notebook environment.  

---

## 📜 License
[MIT License](LICENSE)  

---

## 📧 Contact
For questions or collaboration: **ejigsonpeter@gmail.com**
