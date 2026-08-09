import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("data/insurance.csv")
# print(df["Response"].value_counts())

# sns.countplot(x="Response",data=df)
# plt.title("Target Variable Distribution")
# plt.show()

# print(df[["Age","Annual_Premium","Vintage"]].describe()) # checking for statistical summary of numerical features

# print(df["Gender"].value_counts())

# distribution (histogram)
# plt.figure(figsize=(8,5))
# sns.histplot(df["Vintage"],bins=30,kde=True,color="red")
# plt.title("Vintage Distribution")
# plt.xlabel("Vintage")
# plt.ylabel("count")
# plt.show()

# # boxplot
# plt.figure(figsize=(8,5))
# sns.boxplot(x=df["Vintage"],color="green")
# plt.title("Vintage boxplot..")
# plt.show()
# print(df["Vintage"].describe()) # checking for outliers in Vintage column

# print(df["Annual_Premium"].skew())
# print(df["Annual_Premium"].value_counts().head(10))

# print(df["Vintage"].skew())

# plt.figure(figsize=(8,5))
# sns.countplot(x="Vehicle_Damage",hue="Response",data=df)
# plt.title("Vehicle Damage Distribution")
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(x="Vehicle_Age",hue="Response",data=df)
# plt.title("Vehicle Age Distribution")
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(x="Previously_Insured",hue="Response",data=df)
# plt.title("Previous Insurance Distribution")
# plt.show()

print(pd.crosstab(df["Vehicle_Damage"], df["Response"]))
print(pd.crosstab(df["Vehicle_Age"], df["Response"]))
print(pd.crosstab(df["Previously_Insured"], df["Response"]))

print(pd.crosstab(df["Gender"], df["Response"]))
print(pd.crosstab(df["Driving_License"], df["Response"]))
print(pd.crosstab(df["Region_Code"], df["Response"]))
print(pd.crosstab(df["Policy_Sales_Channel"], df["Response"]))
