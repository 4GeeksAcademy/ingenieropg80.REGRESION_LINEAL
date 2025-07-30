# librerías usadas en útils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math

# ----------------------------------------------------------------------------------------------
# 
def plot_numerical_data(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        fig, axis = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [6, 1]})
        # Calculate mean, median, and standard deviation
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])
        # Create a multiple subplots with histograms and box plots
        sns.histplot(ax=axis[0], data=dataframe, kde=True, x=column).set(xlabel=None)
        axis[0].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[0].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[0].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1, label='Standard Deviation')
        axis[0].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)
        sns.boxplot(ax=axis[1], data=dataframe, x=column, width=0.6).set(xlabel=None)
        axis[1].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[1].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[1].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1)
        axis[1].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)
        axis[0].legend()
        fig.suptitle(column)
        # Adjust the layout
        plt.tight_layout()
        # Show the plot
        plt.show()
#---------------------------------------------------------------------------------------------------

def plot_categorical_data(dataframe):
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Conteo de cada categoría (barras)
        sns.countplot(x=column, data=dataframe, ax=axs[0], palette='pastel')
        axs[0].set_title(f'Conteo de categorías en {column}')
        axs[0].set_xlabel('')
        axs[0].set_ylabel('Frecuencia')
        axs[0].tick_params(axis='x', rotation=45)
        
        # Gráfico de pastel (proporciones)
        dataframe[column].value_counts().plot.pie(autopct='%1.1f%%', ax=axs[1], colors=sns.color_palette('pastel'))
        axs[1].set_ylabel('')
        axs[1].set_title(f'Proporción de categorías en {column}')
        
        plt.tight_layout()
        plt.show()
