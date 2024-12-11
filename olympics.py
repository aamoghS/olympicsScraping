import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get today's date
today = date.today().strftime("%m%d%Y")  # Format the date as MMDDYYYY

# URL of the webpage to scrape for medal counts
url = "https://www.bbc.com/sport/olympics/paris-2024/medals/"

# Make the request to get the data
html = requests.get(url).content

# Parse the HTML content using BeautifulSoup and the html5lib parser
soup = BeautifulSoup(html, 'html5lib')  # run 'pip install html5lib' if this errors out

# Look through the html results to see how we can identify the portions with the medal counts for each country
country_sections = soup.find_all('div', {'class': 'ssrcss-7kfmgb-BadgeContainer ezmsq4q1'})

# Create dictionary to store our medal results
country_medals = {}

# Iterate through all the country sections (there's a section for each country)
for country_section in country_sections:
    # Get the country name
    country_name = country_section.find_next('span', {'class': 'ssrcss-ymac56-CountryName ew4ldjd0'}).get_text(strip=True)
    
    # Create an empty list to store medal counts for the current country
    medal_counts = []
    
    # Find the first 'td' element next to the current country section
    current = country_section.find_next('td')

    # Iterate over the sibling 'td' elements to get all the medal counts
    while current:
        value = current.get_text(strip=True)
        if value.isdigit():
            medal_counts.append(int(value))
        current = current.find_next_sibling('td')
    
    # Add the medal counts to the dictionary with the country name as the key
    country_medals[country_name] = medal_counts

# Convert to a dataframe and store the data
df = pd.DataFrame.from_dict(country_medals)

# Rotate df so the medals are the columns
df = df.T

# Rename columns
new_column_names = ['Gold', 'Silver', 'Bronze', 'Total']
df.columns = new_column_names

# Save as dataframe to keep each day's data
file_name = "./Medals" + today + ".csv"
df.to_csv(file_name)

# Plot the data
plot_df = df[df["Total"] > 0]
plot_df = plot_df.reset_index().rename(columns={'index': 'Country'})
plot_df = plot_df.sort_values('Total', ascending=True)

# Create the bar plot
Country = plot_df.Country
Gold = plot_df.Gold
Silver = plot_df.Silver
Bronze = plot_df.Bronze
Total = plot_df.Total

# Define width of stacked chart
w = 0.8

# Define figure size
plt.figure(figsize=(12, 8))

# Plot stacked bar chart
plt.barh(Country, Gold, w, color='#E2C053', label='Gold')
plt.barh(Country, Silver, w, left=Gold, color='#A5ACB4', label='Silver')
plt.barh(Country, Bronze, w, left=Gold+Silver, color='#B6775B', label='Bronze')

# Add total number of medals to the end of each bar
for i, (country, total) in enumerate(zip(Country, Total)):
    plt.text(total + 0.1, i, str(total), va='center')

plt.title(f"Olympic Medal Counts on {today}", size=18)  # Add title to the plot

# Add legend and set its location to the lower right corner
plt.legend(loc='lower right', fontsize=20)

# Modify axes
plt.xlabel("Medal Count", size=14)
plt.tick_params(axis='y', labelsize=8)
plt.tick_params(axis='x', labelsize=14)

# Set x-axis tick values
plt.xticks(np.arange(0, max(Total)+1, 5))

# Save plot
plot_name = "./Medals" + today + ".png"
plt.savefig(plot_name)  # Save as PNG file

# Display
plt.show()

# Allow user to choose a specific country
chosen_country = input("Enter the name of the country you want to see details for: ")

if chosen_country in df.index:
    country_details = df.loc[chosen_country]
    print(f"\nDetails for {chosen_country}:")
    print(country_details)
else:
    print("Country not found. Please check the name and try again.")

# Prepare data for model
# Using Gold, Silver, and Bronze medals as features
X = df[['Gold', 'Silver', 'Bronze']]
y = df['Total']  # Predicting the Total medal count

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict which country might win the most medals based on current data
# Using the same features from the scraped data
current_data = df[['Gold', 'Silver', 'Bronze']]
predicted_totals = model.predict(current_data)
df['Predicted_Total'] = predicted_totals
df = df.sort_values('Predicted_Total', ascending=False)

# Display predictions for the chosen country
if chosen_country in df.index:
    chosen_country_prediction = df.loc[chosen_country]
    print(f"\nPredicted Total for {chosen_country}:")
    print(chosen_country_prediction[['Gold', 'Silver', 'Bronze', 'Total', 'Predicted_Total']])
else:
    print("Prediction not available for the specified country.")
