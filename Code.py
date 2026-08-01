import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\anush\Downloads\student_performance_dataset.csv")
print(df.head())
df.describe()
df.info()
df.isnull().sum()
df = df.drop("student_id", axis=1)
print(df.head())
sns.countplot(x="gender", data=df)  
plt.show()


print(df["attendance_percent"].groupby(df["final_grade"]).describe())
sns.boxplot(y=df["attendance_percent"], x=df["final_grade"])
plt.show()


print(df["sleep_hours"].groupby(df["final_grade"]).describe())
sns.boxplot(y=df["sleep_hours"], x=df["final_grade"])
print(df["study_time_hours"].groupby(df["final_grade"]).describe())
sns.boxplot(y=df["study_time_hours"], x=df["final_grade"])
print(df["previous_grade"].groupby(df["final_grade"]).describe())
sns.boxplot(y=df["previous_grade"], x=df["final_grade"])
plt.show()


print(df["final_exam_score"].groupby(df["final_grade"]).describe())
sns.boxplot(y=df["final_exam_score"], x=df["final_grade"])
plt.show()


num_df=df[["study_time_hours","attendance_percent","sleep_hours","previous_grade","final_exam_score"]]

matrix = num_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Class Performance Dataset Correlation Heatmap")
plt.show()
sns.countplot(df, x="final_grade", hue="gender")
plt.show()

ct_internet = pd.crosstab(df['final_grade'], df["internet_access"])
sns.heatmap(ct_internet, cmap='coolwarm',annot=True,fmt='g')
plt.show()

sns.countplot(df, x="final_grade", hue="parental_education")
plt.show()

ct_parent = pd.crosstab(df['final_grade'], df["parental_education"])
sns.heatmap(ct_parent, cmap='coolwarm',annot=True,fmt='.3g')
plt.show()

