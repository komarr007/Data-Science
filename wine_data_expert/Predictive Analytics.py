# %% [markdown]
# # Importing Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
%matplotlib inline

import json

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %% [markdown]
# # Describing the dataset

# %%
df = pd.read_csv('wine_quality.csv')
df.info()

# %%
df.describe()

# %% [markdown]
# # Data Understanding
# 
# ## Explatory Data Analysis
# 
# ### Univariate Analysis

# %% [markdown]
# Pada data ini klasifikasi hanya dimiliki oleh data target dan data fitur memiliki nilai kontinu.

# %%
count = df['quality'].value_counts()
percent = 100*df['quality'].value_counts(normalize=True)
df_sample_percentage = pd.DataFrame({'sample count':count, 'percentage':percent.round(1)})
print(df_sample_percentage)
sns.histplot(df['quality'])

# %% [markdown]
# Pada gambar di atas maka dapat diketahui bahwa data berisikan data dengan dominasi pada qualitas 5 dan 6. Pada dataset juga tidak memiliki data dengan qualitas 2, 9 dan 10.

# %% [markdown]
# ## Multivariate Analisis

# %%
plt.figure(figsize=(18,14))
sns.pairplot(df)

# %% [markdown]
# Pada data pairplot dapat dilihat bahwa: <br>
# <ul>
#     <li>Qualitas cenderung memiliki relasi positif dengan alcohol dan berelasi negatif dengan volatile acid</li>
#     <li>Alcohol cenderung berelasi negatif dengan densitas</li>
#     <li>pH memiliki relasi negatif dengan fixed acidity</li>
#     <li>Total sulfur dioxide beralasi positif dengan free sulfur dioxide memiliki relasi negatif dengan fixed acidity</li>
#     <li>Citric Acid berelasi positif dengan fixed acidity dan beralasi negatif dengan pH</li>
# </ul>
# 

# %% [markdown]
# dari data di atas akan dilakukan relasi proving menggunakan heatmap

# %%
plt.figure(figsize=(18,14))
sns.heatmap(df.corr(), xticklabels = df.columns, yticklabels = df.columns, annot=True,
            linewidths=0.5, cmap = "YlGnBu")

# %% [markdown]
# Pada heatmap di atas dapat dilihat untuk persebaran relasi antar fitur sebagai berikut:
# <ul>
#     <li>relasi positif quality dan alcohol</li>
#     <li>relasi positif cenderung quality dan citric acid</li>
#     <li>sulphates relasi positif dengan chlorides dan citric acid</li>
#     <li>pH relasi negatif dengan citric acid dan dengan density</li>
#     <li>relasi negatif dengan density dengan alcohol dan dengan pH dan relasi positif dengan fixed acidity</li>
# </ul>

# %% [markdown]
# # Data Preparation

# %% [markdown]
# pada tahap ini akan dilakukan penyeragaman satuan data dengan code di bawah.

# %%
df['density'] = df['density']/1000
df['free sulfur dioxide'] = df['free sulfur dioxide']/0.001
df['total sulfur dioxide'] = df['total sulfur dioxide']/0.001

# %% [markdown]
# pada describe di bawah menunjukkan penyeragaman data pada fitur densitas, free sulfur dioxide dan total sulfur dioxide

# %%
df.describe()

# %% [markdown]
# ## Outlier

# %% [markdown]
# pada code di bawah akan dilakukan filtrasi pada data dari outlier, proses penanganan outlier diperlukan untuk mengurangi bias pada data dan meningkatkan performa model prediksi.

# %%
# checking for outlier
def outlier(vals):
    dict_holder = {}
    for column in vals.columns:
        data = vals[column].to_numpy()
        mean = np.mean(data)
        std = np.std(data)
        threshold_right = 3
        threshold_left = -3
        outlier = []
        for i in data:
            z = (i-mean)/std
            if z > threshold_right or z < threshold_left:
                if i not in outlier:
                    outlier.append(i)
        dict_holder[column] = outlier
    return dict_holder

# %% [markdown]
# code di bawah menunjukkan kolom dengan outlier, pada penanganan saya tidak menggunakan visualisasi boxplot untuk menganalisa outlier dikarenakan kurangnya akurasi visualisasi pada gambar untuk menemukan outlier

# %%
wine_outlier_json = outlier(df.drop('quality',axis=1))
print("Kolom dengan outlier")
for key in wine_outlier_json:
    print(key)

# %% [markdown]
# code di bawah melakukan masking terhadap data outlier dan melakukan pengecekekan apakah masih terdapat outlier setelah data di masking.

# %%
mask_outlier = (df['fixed acidity'].isin(wine_outlier_json['fixed acidity']) == False)\
            & (df['volatile acidity'].isin(wine_outlier_json['volatile acidity']) == False)\
            & (df['citric acid'].isin(wine_outlier_json['citric acid']) == False)\
            & (df['residual sugar'].isin(wine_outlier_json['residual sugar']) == False)\
            & (df['chlorides'].isin(wine_outlier_json['chlorides']) == False)\
            & (df['free sulfur dioxide'].isin(wine_outlier_json['free sulfur dioxide']) == False)\
            & (df['total sulfur dioxide'].isin(wine_outlier_json['total sulfur dioxide']) == False)\
            & (df['density'].isin(wine_outlier_json['density']) == False)\
            & (df['pH'].isin(wine_outlier_json['pH']) == False)\
            & (df['sulphates'].isin(wine_outlier_json['sulphates']) == False)\
            & (df['alcohol'].isin(wine_outlier_json['alcohol']) == False)

df_wine_clean_outlier = df[mask_outlier]

outlier_checker = outlier(df_wine_clean_outlier.drop('quality',axis=1))

for key in outlier_checker:
    print("{} memiliki outlier sebanyak {} data".format(key, len(outlier_checker[key])))

# %% [markdown]
# code di bawah mengiterasi masking outlier sehingga tidak ditemukan lagi outlier pada data

# %%
iterate = True

while iterate:
    holder = []

    wine_outlier_json = outlier(df_wine_clean_outlier.drop('quality',axis=1))

    mask_outlier = (df_wine_clean_outlier['fixed acidity'].isin(wine_outlier_json['fixed acidity']) == False)\
                & (df_wine_clean_outlier['volatile acidity'].isin(wine_outlier_json['volatile acidity']) == False)\
                & (df_wine_clean_outlier['citric acid'].isin(wine_outlier_json['citric acid']) == False)\
                & (df_wine_clean_outlier['residual sugar'].isin(wine_outlier_json['residual sugar']) == False)\
                & (df_wine_clean_outlier['chlorides'].isin(wine_outlier_json['chlorides']) == False)\
                & (df_wine_clean_outlier['free sulfur dioxide'].isin(wine_outlier_json['free sulfur dioxide']) == False)\
                & (df_wine_clean_outlier['total sulfur dioxide'].isin(wine_outlier_json['total sulfur dioxide']) == False)\
                & (df_wine_clean_outlier['density'].isin(wine_outlier_json['density']) == False)\
                & (df_wine_clean_outlier['pH'].isin(wine_outlier_json['pH']) == False)\
                & (df_wine_clean_outlier['sulphates'].isin(wine_outlier_json['sulphates']) == False)\
                & (df_wine_clean_outlier['alcohol'].isin(wine_outlier_json['alcohol']) == False)

    df_wine_clean_outlier = df_wine_clean_outlier[mask_outlier]

    wine_outlier_json = outlier(df_wine_clean_outlier.drop('quality',axis=1))

    for key in wine_outlier_json:
        result = any(item in wine_outlier_json[key] for item in wine_outlier_json[key])
        if result == True:
            holder.append(True)

    result = any(item in holder for item in holder)

    if result == False:
        break

outlier(df_wine_clean_outlier.drop('quality', axis=1))

# %% [markdown]
# ## Label Encoding

# %% [markdown]
# pada code di bawah akan dilakukan perubahan pada target data dengan mengubah data dengan nilai >= 6 akan dikategorikan sebagai kualitas baik dan data dengan nilai < 6 akan dikategorikan sebagai kualitas buruk. Baik == 1 dan Buruk == 0.

# %%
threshold = 6

df_wine_clean_outlier['quality'] = np.where(df_wine_clean_outlier['quality'] >= threshold, 1, 0)

# %% [markdown]
# code di bawah menunjukkan histplot dari encoding target data dan dapat dilihat bahwa data memiliki lebih banyak data dengan kualitas wine baik yang mana seusai dengan tujuan awal pembuatan model untuk industri F&B sehingga dibutuhkan wine dengan kualitas baik, dengan persebaran data ini diharapkan memodel memiliki keketatan dalam memprediksi kualitas wine.

# %%
count = df_wine_clean_outlier['quality'].value_counts()
percent = 100*df_wine_clean_outlier['quality'].value_counts(normalize=True)
df_sample_percentage = pd.DataFrame({'sample count':count, 'percentage':percent.round(1)})
print(df_sample_percentage)
sns.histplot(df_wine_clean_outlier['quality'])

# %% [markdown]
# ## Scaling data dengan Min Max Scaler

# %% [markdown]
# Pada code di bawah akan dilakukan pengaturan skala data menggunakan min max scaler tujuannya untuk mempermudah model mengolah data dengan nilai yang telah di reduksi skalanya menggunakan z score.

# %%
scaler = MinMaxScaler()
scaler = scaler.fit(df_wine_clean_outlier.drop(["quality"],axis=1))
df_wine_scaled = scaler.transform(df_wine_clean_outlier.drop(["quality"],axis=1))
df_wine_scaled = pd.DataFrame(df_wine_scaled, columns=df_wine_clean_outlier.drop(["quality"],axis=1).columns)
df_wine_scaled.describe()

# %% [markdown]
# ## Splitting data

# %% [markdown]
# Pada code di bawah akan dilakukan pembagian train data dan test data agar dapat dilakukan evaluasi terhadap model yang telah dibuat.

# %%
X = df_wine_scaled
y = df_wine_clean_outlier["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # Modeling

# %% [markdown]
# pada code di bawah akan dilakukan modeling klasifikasi menggunakan algoritme logistic regression, support vector machine dan random forest

# %%
logreg = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier()


logreg.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)


logreg_pred = logreg.predict(X_test)
svc_pred = svc.predict(X_test)
rf_pred = rf.predict(X_test)


# %% [markdown]
# # Evaluation

# %% [markdown]
# Pada model ini akan dilakukan evaluasi matrics menggunakan accuracy, precision, recall dan f1 score. Tujuan utama dari pembuatan model ini adalah dengan mengambil model dengan nilai recall lebih rendah. Dikarenakan jika nilai recall rendah maka nilai false negative akan lebih besar yang artinya akan lebih ketat dalam model memprediksi suatu kualitas wine dikatakan baik. Sehingga perusahaan atau industri yang menggunakan model ini akan lebih mungkin mendapatkan kualitas wine baik lebih besar. Tetapi untuk pemilihan model akan diukur juga menggunakan f1 score, nilai f1 score = 1 memiliki arti bahwa data memiliki balance antara precision dan recall. 

# %%
logreg_acc = accuracy_score(y_test, logreg_pred)
logreg_precision = precision_score(y_test, logreg_pred)
logreg_recall = recall_score(y_test, logreg_pred)
logreg_f1 = f1_score(y_test, logreg_pred)

svc_acc = accuracy_score(y_test, svc_pred)
svc_precision = precision_score(y_test, svc_pred)
svc_recall = recall_score(y_test, svc_pred)
svc_f1 = f1_score(y_test, svc_pred)

rf_acc = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)


results = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest'],
    'Accuracy': [logreg_acc, svc_acc, rf_acc],
    'Precision': [logreg_precision, svc_precision, rf_precision],
    'Recall': [logreg_recall, svc_recall, rf_recall],
    'F1 Score': [logreg_f1, svc_f1, rf_f1]
}


df_model_eval = pd.DataFrame(results)

df_model_eval

# %% [markdown]
# Dari hasil model di atas maka model yang paling optimum digunakan adalah **Random Forest**. Random Forest memiliki nilai f1 score yang paling mendekati 1 dan nilai recall lebih rendah dibandingkan dengan precision, maka dengan itu model random forest memiliki performa lebih baik untuk penyelesaikan model bisnis yang akan dilakukan.


