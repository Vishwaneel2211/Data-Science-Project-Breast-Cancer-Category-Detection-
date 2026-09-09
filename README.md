🩺 Breast Cancer Diagnosis — Ensemble Learning

Classifies breast tumors as Malignant or Benign using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, with an ensemble of Extra Trees + Random Forest + XGBoost (soft voting), compared against a Logistic Regression baseline.

**Features Used**

10 real-valued nucleus measurements (mean values): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

**Pipeline**

Data preprocessing → EDA → Baseline (Logistic Regression) → Ensemble (Extra Trees + Random Forest + XGBoost, soft voting) → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC) → Feature Importance → Interactive prediction widget (Colab).

**Results**
Model	                            Accuracy	             Precision	        Recall	         F1	         ROC-AUC
Logistic Regression (Baseline)	  0.9298	                0.8864	         0.9286	         0.9070	       0.9841
Extra Trees                     	0.9298	                0.9048	         0.9048          0.9048      	 0.9907
Random Forest	                    0.9386	                0.9268	         0.9048 	       0.9157	       0.9866
XGBoost	                          0.9649	                0.9524	         0.9524	         0.9524 	     0.9785
Voting Ensemble (ET+RF+XGB)	      0.9561	                0.9512	         0.9286	         0.9398	       0.9901

Disclaimer

Educational/portfolio project only — not a certified medical device. Not a substitute for professional diagnosis.
