import pandas as pd 
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
penguin_df = pd.read_csv('/home/kareem/hacking/Github_projects/streamlit_apps/streamlit_apps/penguin_ml/penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)#! maps the labels to numbers
x_train,x_test,y_train,y_test = train_test_split(
    features,output,test_size=0.8
)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values,y_train)
y_pred =  rfc.predict(x_test.values)
score = accuracy_score(y_pred,y_test)
print("Our accuracy score for the model is {}".format(score))
rf_pickle = open("Random_forces_penguins.pickle","wb")
pickle.dump(rfc,rf_pickle)
rf_pickle.close()
output_pickle = open("output_penguins.pickle","wb")
pickle.dump(uniques,output_pickle)
