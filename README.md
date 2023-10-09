# algo-8-project
Analysing the Wine Quality Dataset with Python
Wine Quality Dataset Report

1.Introduction: The dataset is related to the red and white variants of the wine. It contains various physicochemical properties of the wines and a quality rating.

2. Data Overview: The dataset contains the following attributes:
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (score between 0 and 10)


a.	In the above reference, two datasets were created, using red and white wine samples.
b.	The inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). 
c.	Also, we plot the relative importance of the input variables vs quality.
3. Data Exploration:
from the above dataset using the [. describe ()] from  pandas library we get basic statistics for each attribute, like mean, standard deviation, minimum, and maximum values. 
the finding we get here is:
•	Most wines have an alcohol percentage around 10%.
•	The average quality rating is around 5.6.
•	
    4. Data Visualization:
Using libraries like Matplotlib and Seaborn, you could visualize the data:
Histograms: To see the distribution of each attribute.
Correlation heatmap: To check the relationship between the attributes.
From visualizations, you might find:
•	Most wines have a quality rating of 5 or 6.
•	Attributes like alcohol, sulphates, and citric acid might have a positive correlation with wine quality.

The key findings of the dataset are as follows for red wine :

•	There is a positive correlation between alcohol content and wine quality.
•	There is a negative correlation between volatile acidity and wine quality.
•	There is a negative correlation between residual sugar and wine quality.
•	There is a negative correlation between pH and wine quality.
•	There is a positive correlation between density and wine quality.
•	There is a positive correlation between Sulphates and wine quality.
 
 

The key findings of the dataset are as follows for white wine :
* There is a positive correlation between alcohol and wine quality.
* There is a negative correlation between volatile acidity and wine quality
* There is a positive correlation between residual sugar and wine quality.
* There is a positive correlation between pH and wine quality.
* There is a negative correlation between density and wine quality.
* There is a positive correlation between sulphates and wine quality.
* There is a positive correlation between fixed acidity and wine quality.

 

 
