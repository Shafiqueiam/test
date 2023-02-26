import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# loading data
df = pd.read_csv("E:\wasim\Project\Consumer_Complaints.csv")
df.shape
df.head(2)


# Create a new dataframe 
df1 = df[['Product', 'Consumer complaint narrative']].copy()

# Remove missing values 
df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]

# Renaming column 
df1.columns = ['Product', 'Consumer_complaint'] 

df1.shape


# Percentage of complaints with text
total = df1['Consumer_complaint'].notnull().sum()
round((total/len(df)*100),1)

pd.DataFrame(df.Product.unique()).values

# Because the computation is time consuming (in terms of CPU), the data was sampled
df2 = df1.sample(10000, random_state=1).copy()

# Renaming categories
df2.replace({'Product': 
             {'Credit reporting, credit repair services, or other personal consumer reports': 
              'Credit reporting, repair, or other', 
              'Credit reporting': 'Credit reporting, repair, or other',
             'Credit card': 'Credit card or prepaid card',
             'Prepaid card': 'Credit card or prepaid card',
             'Payday loan': 'Payday loan, title loan, or personal loan',
             'Money transfer': 'Money transfer, virtual currency, or money service',
             'Virtual currency': 'Money transfer, virtual currency, or money service'}}, 
            inplace= True)

pd.DataFrame(df2.Product.unique())

# Create a new column 'category_id' with encoded categories 
df2['category_id'] = df2['Product'].factorize()[0]
category_id_df = df2[['Product', 'category_id']].drop_duplicates()


# Dictionaries for use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)

# New dataframe
df2.head()

fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue']

df2.groupby('Product').Consumer_complaint.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF COMPLAINTS IN EACH PRODUCT CATEGORY\n')
plt.xlabel('Number of ocurrences', fontsize = 10);


x = df2['Consumer_complaint'] # Collection of documents
y = df2['Product'] # Target or the labels we want to predict (i.e., the 13 different complaints of products

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# using vectorization

vectorization = TfidfVectorizer()
vector = vectorization.fit(x_train)
xv_train = vector.transform(x_train)
xv_test = vectorization.transform(x_test)

xv_train

xv_test

LR = LogisticRegression()

LR = LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)


print(classification_report(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)

print(classification_report(y_test, pred_dt))

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

print(classification_report(y_test, pred_rfc))

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
  plt.ylabel('Truth')
  plt.xlabel('Predicted')
  
cm = confusion_matrix(y_test, pred_lr)
df_cm = pd.DataFrame(cm)
show_confusion_matrix(df_cm)

def label_text(text, model, vect):
  text = vect.transform([text])
  return model.predict(text).flatten()[0]

text = 'I want a new credit card'
label_text(text, LR, vectorization)

import pickle
with open('final_model.pickle', 'wb')as f:
    pickle.dump(LR,f)
    
with open('vect.pickle', 'wb')as f:
    pickle.dump(vector,f)
    
# using flask for deployment    
    
from flask import Flask, request, render_template
import pickle


LR = pickle.load(open('final_model.pickle', 'rb'))
vect = pickle.load(open('vect.pickle', 'rb'))


import pickle


LR = pickle.load(open('model.pickle', 'rb'))
vect = pickle.load(open('vector.pickle', 'rb'))

app = Flask(__name__)

@app.route('/')
def issue():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        text = request.form['nm']
        result_pred = LR.predict(vect.transform([text]))
        return render_template("result.html", result = result_pred.flatten()[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 