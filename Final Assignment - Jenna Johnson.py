#!/usr/bin/env python
# coding: utf-8

# In[1]:


def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading data from {file_path}: {str(e)}")
        return None
def analyze_hospital_data(hospital_name, df):
    if df is None:
        return


# In[4]:


# Task 1: Number of patients who were readmitted
readmitted_patients = df[df['Readmission'] == 1]
num_readmitted = len(readmitted_patients)
print(f"\nNumber of patients readmitted in {hospital_name}: {num_readmitted}")


# In[5]:


# Task 2: Average satisfaction score for each category
    satisfaction_columns = ['StaffSatisfaction', 'CleanlinessSatisfaction', 'FoodSatisfaction', 'ComfortSatisfaction', 'CommunicationSatisfaction']
    avg_satisfaction = df[satisfaction_columns].mean()
    print(f"\nAverage satisfaction scores for {hospital_name}:")
    print(avg_satisfaction)


# In[6]:


# Task 3: Logistic Regression
    X = df[satisfaction_columns]
    y = df['Readmission']


# In[7]:


# Create a logistic regression model
    model = LogisticRegression()
    model.fit(X, y)


# In[8]:


# Task 4: Display logistic regression results
   correlation = np.corrcoef(X.sum(axis=1), y)[0, 1]
   print("\nLogistic Regression Results:")
   if correlation > 0.5:
       print("Correlation between Overall Satisfaction Scores and Readmission: Strong correlation")
   elif correlation > 0.3:
       print("Correlation between Overall Satisfaction Scores and Readmission: Moderate correlation")
   elif correlation > 0.1:
       print("Correlation between Overall Satisfaction Scores and Readmission: Weak correlation")
   else:
       print("Correlation between Overall Satisfaction Scores and Readmission: No correlation")


# In[9]:


# Task 5: Plot the data points along with the logistic regression curve
   plt.figure(figsize=(8, 6))
   plt.scatter(X.sum(axis=1), y, color='black', label='Actual Data Points')
   plt.xlabel('Overall Satisfaction Scores')
   plt.ylabel('Readmission')
   plt.title(f'Logistic Regression - {hospital_name}')
   plt.legend()
   plt.show()

   return num_readmitted, avg_satisfaction


# In[10]:


# Read data from files
hospital1_data = read_data('hospital1_data.txt')
hospital2_data = read_data('hospital2_data.txt')


# In[11]:


# Perform data analysis for each hospital
if hospital1_data is not None and hospital2_data is not None:
    print("\nHospital Comparison:")
    print("--------------------")

    print("\nHospital 1 Data Analysis:")
    num_readmitted_1, avg_satisfaction_1 = analyze_hospital_data('Hospital 1', hospital1_data)

    print("\nHospital 2 Data Analysis:")
    num_readmitted_2, avg_satisfaction_2 = analyze_hospital_data('Hospital 2', hospital2_data)


# In[12]:


# Task 6: Compare the logistic regression results of both hospitals
    print("\nConclusion:")
    if num_readmitted_1 < num_readmitted_2:
        print("Hospital 1 has fewer readmissions.")
    elif num_readmitted_1 > num_readmitted_2:
        print("Hospital 2 has fewer readmissions.")
    else:
        print("Both hospitals have the same number of readmissions.")

    if avg_satisfaction_1.mean() > avg_satisfaction_2.mean():
        print("Hospital 1 has higher average satisfaction scores.")
    elif avg_satisfaction_1.mean() < avg_satisfaction_2.mean():
        print("Hospital 2 has higher average satisfaction scores.")
    else:
        print("Both hospitals have the same average satisfaction scores.")


# In[ ]:




