
"""************************Import necessary packages********************"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


"""****************Function to train the model*********************************"""
def train_and_tune_model(X_train, y_train, model, param_grid, cv, scoring = "f1", verbose=1):
    """
    Train and fine-tune a binary classification model using GridSearchCV.

    Parameters
    ----------
    X_train : DataFrame
        Training data features.
    y_train : Series
        Training data target.
    model : Estimator object
        The binary classification model to be trained.
    param_grid : dict
        The hyperparameter grid to use for fine-tuning the model.
    cv : Cross-validation strategy
        The cross-validation splitting strategy.
    scoring : str or list of str, optional
        The scoring metric(s) to use for evaluation. Default is 'f1-score'.
    verbose : int, optional
        The verbosity level.

    Returns
    -------
    best_model : Estimator object
        The best model found during the GridSearchCV process.
    """
    # Setting up the GridSearchCV
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               cv=cv, 
                               scoring=scoring,  
                               n_jobs=-1,  # Use parallel processing
                               verbose=verbose)  # amount of messaging (information) output 

    # Fitting the model
    grid_search.fit(X_train, y_train)

    # Retrieving the best model
    best_model = grid_search.best_estimator_

    return best_model


"""************************ Display the results*******************************"""
def display_results(dict_models, X_train, y_train, X_test, y_test, cv, disp_col):
    """
    Display the F1 scores of different models in a DataFrame for comparison purposes.

    Parameters
    ----------
    dict_models : dict
        Contains the models that we want to compare along with their parameter grids.
    X_train : DataFrame
        Training data features.
    y_train : Series
        Training data target
    X_test : DataFrame
        Test data features
    y_test : Series
        Test data target
    cv : StratifiedKFold
        Cross-validation strategy
    disp_col : str
        Name of the column to be displayed

    Returns
    -------
    df_results : DataFrame
        DataFrame with the F1 scores.
    """
    
    df_results = pd.DataFrame(columns=["Model Name",disp_col])

    for model_name, model_details in tqdm(dict_models.items(), desc="Going through each model defined in the dictionnary..."):
        #extract the details related to every model from the dict
        model_params = model_details["param_grid"]
        model = model_details["model"]
        best_model = train_and_tune_model(X_train, y_train, model, model_params, cv)
        score = test_model(X_test, y_test, best_model) #evaluate f1 score on test data
        rounded_score = np.round(score*100,2)
        new_row = {"Model Name": model_name, disp_col: rounded_score}
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

        
        conf_matrix = confusion_matrix(y_test, best_model.predict(X_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Print and analyze additional evaluation metrics
        y_pred = best_model.predict(X_test)
        print(f'Model: {model_name}')
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        print(f'Precision: {precision_score(y_test, y_pred)}')
        print(f'Recall: {recall_score(y_test, y_pred)}')
        print(f'ROC-AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])}')
        print('\n')

        # Plot learning curves
        train_sizes, train_scores, valid_scores = learning_curve(best_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training F1 Score')
        plt.plot(train_sizes, np.mean(valid_scores, axis=1), label='Validation F1 Score')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title(f'Learning Curves - {model_name}')
        plt.show()
    # Apply styling after creating the DataFrame
    df_results = df_results.style.highlight_max(subset=[disp_col], color='salmon') #highlight the model with the higher f1 score
    return df_results


""" ******************************* function to test my model*********************************** """
def test_model(X_test, y_test, model):
    """
    Evaluate the F1 score of a trained model on the test set.

    Parameters
    ----------
    X_test : DataFrame
        Test data features.
    y_test : Series
        Test data target.
    model : trained model
        The model to be evaluated.

    Returns
    -------
    score : float
        The F1 score of the model on the test set.
    """
    
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    return score


"""**********************Function to chose which type standardization we want to use**************************"""
def standardization(data):

    
        scaler1 = MinMaxScaler()
        minmax = scaler1.fit_transform(data[['R2', 'R14', 'R17','R32']])
        minmax = pd.DataFrame(minmax, columns =['R2','R14','R17','R32'])
    
        scaler2 = StandardScaler()
        standard = scaler2.fit_transform(data[['R2', 'R14', 'R17','R32']])
        standard = pd.DataFrame(standard, columns =['R2','R14','R17','R32'])
        
        scaler3 = preprocessing.RobustScaler()
        robust = scaler3.fit_transform(data[['R2', 'R14', 'R17','R32']])
        robust = pd.DataFrame(robust, columns =['R2','R14','R17','R32'])
        #display(robust.head())
        return minmax, standard, robust
    
"""********************Function to plot the densities of columns***************** """
def plot_densities(data):

    minmax, standard, robust = standardization(data)

    plt.figure(figsize=(15,4))
    plt.subplot(1,4,1)
    sns.kdeplot(data['R2'], color ='red')
    sns.kdeplot(data['R14'],  color ='green')
    sns.kdeplot(data['R17'],  color ='blue')
    sns.kdeplot(data['R32'],  color ='brown')
    plt.title('Without Scaling')

    plt.subplot(1,4,4) 
    sns.kdeplot(minmax['R2'],  color ='red')
    sns.kdeplot(minmax['R14'],  color ='green')
    sns.kdeplot(minmax['R17'],  color ='blue')
    sns.kdeplot(minmax['R32'],  color ='brown')
    plt.title('After Min-Max Scaling')

    plt.subplot(1,4,3)
    sns.kdeplot(standard['R2'], color ='red')
    sns.kdeplot(standard['R14'],  color ='green')
    sns.kdeplot(standard['R17'],  color ='blue')
    sns.kdeplot(standard['R32'],  color ='brown')
    plt.title('After Standard Scaling')

    plt.subplot(1,4,2)
    sns.kdeplot(robust['R2'],  color ='red')
    sns.kdeplot(robust['R14'],  color ='green')
    sns.kdeplot(robust['R17'],  color ='blue')
    sns.kdeplot(robust['R32'],  color ='brown')
    plt.title('After Robust Scaling')


    plt.tight_layout()
    plt.show()