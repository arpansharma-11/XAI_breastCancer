# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 23:54:58 2022

@author: Arpan
"""


# Utilizing our same xgb_mod model object created above
# Import pacakages
import lime
import numpy as np
import lime.lime_tabular
from SHAP import explainer
import xgboost 
from Modelling import xgb_mod, X_test, X

############## create explainer ###########
# we use the dataframes splits created above for SHAP
explainer = lime.lime_tabular.LimeTabularExplainer(X_test.to_numpy(), feature_names=X_test.columns, class_names=['0','1'], verbose=True)

############## visualizations #############
X_np = X_test.to_numpy()
exp = explainer.explain_instance(X_np[79], xgb_mod.predict_proba, num_features=20)
exp.show_in_notebook(show_table=True)
#print(exp.as_list())