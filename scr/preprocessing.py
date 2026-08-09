import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df=pd.read_csv("data/insurance.csv")
    # print(df.shape)     # return tuple of rows and col
    
    df.drop("id",axis=1,inplace=True) # id is not useful for prediction so eleminate it
    # print(df.columns)

    # print(df.isnull().sum())  # checking for null values

    # print(df.duplicated().sum()) # checking for duplicate values in dataframe

    df.drop_duplicates(inplace=True) # dropping duplicates

    # print(df.shape)
    # print(df["Gender"].unique())        #checking for unique values
    # print(df["Gender"].value_counts())

    # feature and target sepration
    X=df.drop("Response",axis=1)
    y=df["Response"]

         #    ENCODING
    X["Gender"]=X["Gender"].map({"Male":1,"Female":0})
    X["Vehicle_Damage"]=X["Vehicle_Damage"].map({"Yes":1,"No":0})
    X["Vehicle_Age"]=X["Vehicle_Age"].map({"< 1 Year":0,"1-2 Year":1,"> 2 Years":2})
    X["Vehicle_Age"]=X["Vehicle_Age"].astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y) # random_state is used to ensure that the split is reproducible, meaning that if you run the code multiple times with the same random_state, you'll get the same train-test split each time. This is important for consistency in model evaluation and comparison.

            # Scaling the features (for logistic regression, SVM, neural networksand KNN)
    # scaler=StandardScaler()
    # X_train_scaled=scaler.fit_transform(X_train)
    # X_test_scaled=scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
