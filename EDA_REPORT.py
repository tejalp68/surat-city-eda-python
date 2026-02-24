import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#IGNORE WARNINGS
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("surat.csv")     #Reading CSV file
df

# -----------------------------

df.info()

# -----------------------------

df.isnull().sum()

# -----------------------------

df.nunique()

# -----------------------------

df.columns

# -----------------------------

df["price"] = df["price"].str.lstrip("₹")
df["price"] = df["price"].str.rstrip(" Lac")
df["price"] = df["price"].str.rstrip(" Cr")
df['square_feet'] = df['square_feet'].str.rstrip(" sqft")
df['price_per_sqft'] = df['price_per_sqft'].str.rstrip(" per sqft")
df['price_per_sqft'] = df['price_per_sqft'].str.lstrip("₹")
df.head()


# -----------------------------

df[['current floor','total floors']] = df['floor'].str.split('out of' , n=1 , expand= True)
df.head(5)

# -----------------------------

df['price_per_sqft'] = df['price_per_sqft'].str.replace(',', '', regex=True)
df['price_per_sqft'] = pd.to_numeric(df['price_per_sqft'], errors='coerce')


# -----------------------------

df['transaction'] = df['transaction'].fillna(df['transaction'].mode()[0])
df['price_per_sqft'] = df['price_per_sqft'].fillna(df['price_per_sqft'].median())
df['status'] = df['status'].fillna(df['status'].mode()[0])
df['furnishing'] = df['furnishing'].fillna('Unknown')
df['facing'] = df['facing'].fillna('Unknown')
df.drop(columns=['description'], inplace=True) 
df

# -----------------------------

df['current floor'] = df['current floor'].astype(str).str.replace('[^0-9]' , "" , regex=True)
df['current floor'] = pd.to_numeric(df['current floor'], errors='coerce')
df['current floor'] = df['current floor'].fillna(df['current floor'].median())
df['total floors'] = df['total floors'].astype(str).str.replace('[^0-9]' , "" , regex=True)
df['total floors'] = pd.to_numeric(df['total floors'], errors='coerce')
df['total floors'] = df['total floors'].fillna(df['total floors'].median())
df['furnishing'] = df['furnishing'].astype(str).str.replace('[^A-Za-z]', " ", regex=True)
df['furnishing'] = df['furnishing'].replace('', np.nan)
df['furnishing'] = df['furnishing'].fillna(df['furnishing'].mode()[0])
df

# -----------------------------

df = df.drop('floor', axis=1)

# -----------------------------

df

# -----------------------------

plt.hist(df['price'].head(50), color = "orange" ,edgecolor = 'black' )
plt.title("Distribution of property prices")
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()


# -----------------------------

df['bhk'] = df['property_name'].str.extract(r'(\d+)\s*BHK').astype(float)
df

# -----------------------------

df['location'] = df['property_name'].str.extract(r'in ([a-zA-Z\s]+) Surat', expand=False).str.strip()
avg_price_per_sqft = df.groupby('location')['price_per_sqft'].mean().sort_values(ascending=False)
print(avg_price_per_sqft.head(10))  # Show top 10 locations

# -----------------------------

df.describe()

# -----------------------------

a = df['location'].value_counts()
print(a)
b = df['bhk'].value_counts()
print(b)

# -----------------------------

df['price'] = df['price'].astype(str).str.replace(',', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])          # Remove rows with missing price

df['square_feet'] = df['square_feet'].astype(str).str.replace('[^0-9]', '', regex=True)
df['square_feet'] = pd.to_numeric(df['square_feet'], errors='coerce')

# Drop rows with missing values in either column
df = df.dropna(subset=['square_feet', 'price'])


# -----------------------------

avg_price_bhk = df.groupby('bhk')['price'].mean()
print(avg_price_bhk)
avg_price_location = df.groupby('location')['price'].mean()
print(avg_price_location)
correlation = df['square_feet'].corr(df['price'])
print("Correlation between square_feet and price:", correlation)


# -----------------------------

plt.hist(df['price'], color = "orange" ,edgecolor = 'black' )
plt.title("Distribution of property prices")
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

# -----------------------------

plt.hist(df['square_feet'].head(50), color = "orange" ,edgecolor = 'black' )
plt.title("Distribution of property sizes")
plt.xlabel('Size about Square feet')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

# -----------------------------

plt.hist(df['price_per_sqft'].head(50), color = "orange" ,edgecolor = 'black' )
plt.title("Distribution of property prices per sqft")
plt.xlabel('Price per sqft')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

# -----------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(df['price'])
plt.title("Outlier Detection in Property Prices")
plt.ylabel("Price (in ₹)")
plt.grid(True)
plt.show()

# -----------------------------

a = df['location'].value_counts()
plt.bar(a, color = "blue"  ,height = 10 )
plt.title("number of listings per area")
plt.xlabel('area')
plt.ylabel('Frequency')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()


# -----------------------------

plt.scatter(df['square_feet'], df['price'], color='skyblue', edgecolors='black')

plt.title("Price vs Square Feet", fontsize=16)
plt.xlabel("Square Feet", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()