from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from pandas import read_csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
chedf = pd.read_csv('C:\\Users\\shwet\\Desktop\\che293\\che.csv')
chedf['switch'] = chedf[['HAS_H', 'HAS_O', 'HAS_S', 'HAS_U']].sum(axis=1)
chedf['switch'] = chedf['switch'].apply(lambda x: 1 if x > 1 else 0)
chedf.rename(columns={'Unnamed: 0': 'Serial_No'}, inplace=True)
print(chedf.columns)
print(chedf.shape)
print(chedf.head())
print(chedf.describe())
print(chedf.dtypes)
chedf = chedf.drop(['E6.1', 'E20.1', 'E22', 'Serial_No', 'Residue'], axis=1)
print(chedf.columns)
percent_switches = (chedf['switch'].sum())/(len(chedf['switch']))
print(f"Percentage of proteins in dataset that are switches is {percent_switches:.4%}")
def normalize_column(quant_column):
    magnitude = np.sqrt((np.power(quant_column,2)).sum())
    return quant_column/magnitude

def standardize_column(quant_column): # assumes Gaussian distribution
    U = np.mean(quant_column)
    o = np.std(quant_column)
    return (quant_column - U)/o

normalized_data = chedf
for x in ['isUnstruct', 'Vkbat', 'E6', 'E20']:
    normalized_data[x] = normalize_column(chedf[x])

print(normalized_data.describe())

print(normalized_data['chou_fasman'].unique())

for i in normalized_data.columns:
  nan = normalized_data[i].isnull().sum()
  print(f"The column {i} has {nan} null values")

normalized_data.dropna(inplace=True)

for i in normalized_data.columns:
  nan = normalized_data[i].isnull().sum()
  print(f"The column {i} has {nan} null values")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
drop_col = ['Protein', 'No.', 'Res','chou_fasman', 'sspro_5', 'gor4', 'dsc', 'jnet', 'psipred',
       '# homologues', 'HAS_H', 'HAS_S', 'HAS_O', 'HAS_U', 'ProteinID','switch']

x = normalized_data.drop(columns=drop_col,axis=1)

y = normalized_data['switch']

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=37)
rf_class = RandomForestClassifier()
rf_class.fit(x_tr, y_tr)
probabilities = rf_class.predict_proba(x_te)[:, 1].
fitted = rf_class.predict(x_te)


print("Classification Report:")
print(classification_report(y_te, fitted))

fpr, tpr, _ = roc_curve(y_te, probabilities)
roc_auc = auc(fpr, tpr)

feat = pd.DataFrame(rf_class.feature_importances_, index = x.columns)
print(feat)

c_mat = confusion_matrix(y_te, fitted)
sns.heatmap(c_mat, annot=True, fmt='.2f', cmap='Oranges')
plt.xlabel('Actual Observations')
plt.ylabel('Predictions')
plt.title('Confusion Matrix')
plt.show()

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
