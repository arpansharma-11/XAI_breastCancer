# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:31:38 2022

@author: Arpan
"""

import shap
from Modelling import X, xgb_mod, X_train
import matplotlib as plt

# Generate the Tree explainer and SHAP values
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

print(shap_values)
print()
print("Expected Values" , expected_value)



############## visualizations #############
# Generate summary dot plot
shap.summary_plot(shap_values, X,title="SHAP summary plot") 

# Generate summary bar plot 
shap.summary_plot(shap_values, X,plot_type="bar") 

# Generate waterfall plot  
shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=X.loc[79,:], feature_names=X.columns, max_display=15, show=True)

# Generate dependence plot
shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")


# Generate multiple dependence plots
for name in X_train.columns:
     shap.dependence_plot(name, shap_values, X)



shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")


# Generate force plot - Multiple rows 
shap.force_plot(explainer.expected_value, shap_values[:100,:], X.iloc[:100,:]) 
plt.savefig('tmp.svg')
plt.close()

#Generate force plot - Single
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# Generate Decision plot 
shap.decision_plot(expected_value, shap_values[20],link='logit' ,features=X.loc[20,:], feature_names=(X.columns.tolist()),show=True,title="Decision Plot")

