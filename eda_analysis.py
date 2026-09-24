
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CODEALPHA TASK 2 - EDA PROJECT
# E-Commerce Sales Exploratory Data Analysis
# ==========================================

# 1. Create project folders
os.makedirs("Dataset", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

# 2. Dataset path
dataset_path = "Dataset/ecommerce_sales.csv"

# 3. Generate sample dataset if CSV does not exist
if not os.path.exists(dataset_path):

    random.seed(42)
    np.random.seed(42)

    products = [
        "Laptop",
        "Mobile Phone",
        "Headphones",
        "Keyboard",
        "Mouse",
        "Monitor",
        "Tablet",
        "Smart Watch"
    ]

    categories = {
        "Laptop": "Electronics",
        "Mobile Phone": "Electronics",
        "Headphones": "Accessories",
        "Keyboard": "Accessories",
        "Mouse": "Accessories",
        "Monitor": "Electronics",
        "Tablet": "Electronics",
        "Smart Watch": "Accessories"
    }

    cities = [
        "Chennai",
        "Coimbatore",
        "Tirupur",
        "Erode",
        "Bangalore",
        "Madurai",
        "Salem",
        "Trichy"
    ]

    rows = []

    for order_id in range(1, 501):

        product = random.choice(products)
        category = categories[product]
        city = random.choice(cities)

        order_date = pd.Timestamp("2025-01-01") + pd.Timedelta(
            days=random.randint(0, 364)
        )

        quantity = random.randint(1, 5)

        price = {
            "Laptop": 55000,
            "Mobile Phone": 25000,
            "Headphones": 2500,
            "Keyboard": 1500,
            "Mouse": 800,
            "Monitor": 12000,
            "Tablet": 18000,
            "Smart Watch": 5000
        }[product]

        unit_price = price + random.randint(-500, 1000)

        sales = quantity * unit_price

        rows.append({
            "Order_ID": order_id,
            "Order_Date": order_date,
            "Product": product,
            "Category": category,
            "City": city,
            "Quantity": quantity,
            "Unit_Price": unit_price,
            "Sales": sales
        })

    df = pd.DataFrame(rows)

    df.to_csv(dataset_path, index=False)

    print("New dataset created successfully!")

else:
    print("Existing dataset found. Loading dataset...")

# 4. Load dataset
df = pd.read_csv(dataset_path)

print("\n========== DATASET LOADED ==========")
print(df.head())

# 5. Dataset information
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 6. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 7. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# 8. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 9. Data cleaning
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df = df.drop_duplicates()

df = df.dropna(
    subset=[
        "Order_Date",
        "Product",
        "Category",
        "Quantity",
        "Unit_Price",
        "Sales"
    ]
)

# 10. Create useful date columns
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%b")

# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

# 11. Total sales
total_sales = df["Sales"].sum()

print("\n========== BUSINESS INSIGHTS ==========")
print("Total Sales:", round(total_sales, 2))

# 12. Total quantity sold
total_quantity = df["Quantity"].sum()

print("Total Quantity Sold:", total_quantity)

# 13. Best-selling products by sales
product_sales = (
    df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY PRODUCT ==========")
print(product_sales)

# 14. Sales by category
category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY CATEGORY ==========")
print(category_sales)

# 15. Sales by city
city_sales = (
    df.groupby("City")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY CITY ==========")
print(city_sales)

# 16. Monthly sales
monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
)

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)

# ==========================================
# DATA VISUALIZATION
# ==========================================

sns.set_theme(style="whitegrid")

# Chart 1: Sales by Product
plt.figure(figsize=(10, 6))

product_sales.plot(kind="bar")

plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/charts/sales_by_product.png")
plt.show()
plt.close()

# Chart 2: Sales by Category
plt.figure(figsize=(8, 5))

category_sales.plot(kind="bar")

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/charts/sales_by_category.png")
plt.show()
plt.close()

# Chart 3: Sales by City
plt.figure(figsize=(10, 6))

city_sales.plot(kind="bar")

plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/charts/sales_by_city.png")
plt.show()
plt.close()

# Chart 4: Monthly Sales Trend
plt.figure(figsize=(10, 6))

monthly_sales.plot(
    kind="line",
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(range(1, 13))
plt.tight_layout()

plt.savefig("outputs/charts/monthly_sales_trend.png")
plt.show()
plt.close()

# Chart 5: Quantity Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Quantity"],
    bins=5,
    kde=True
)

plt.title("Quantity Distribution")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/charts/quantity_distribution.png")
plt.show()
plt.close()

# Chart 6: Category Sales Pie Chart
plt.figure(figsize=(8, 8))

category_sales.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Sales Distribution by Category")
plt.ylabel("")
plt.tight_layout()

plt.savefig("outputs/charts/category_sales_pie.png")
plt.show()
plt.close()

# ==========================================
# SAVE CLEANED DATA
# ==========================================

cleaned_path = "outputs/cleaned_ecommerce_sales.csv"

df.to_csv(cleaned_path, index=False)

# ==========================================
# SAVE EDA REPORT
# ==========================================

report_path = "outputs/eda_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("CODEALPHA TASK 2 - EDA REPORT\n")
    report.write("=" * 40 + "\n\n")

    report.write(f"Total Records: {len(df)}\n")
    report.write(f"Total Sales: {total_sales:.2f}\n")
    report.write(f"Total Quantity Sold: {total_quantity}\n\n")

    report.write("Sales by Product:\n")
    report.write(str(product_sales))
    report.write("\n\n")

    report.write("Sales by Category:\n")
    report.write(str(category_sales))
    report.write("\n\n")

    report.write("Sales by City:\n")
    report.write(str(city_sales))
    report.write("\n\n")

    report.write("Monthly Sales:\n")
    report.write(str(monthly_sales))
    report.write("\n")

print("\n========================================")
print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nGenerated Files:")
print("1. Dataset/ecommerce_sales.csv")
print("2. outputs/cleaned_ecommerce_sales.csv")
print("3. outputs/eda_report.txt")
print("4. outputs/charts/")

# ==========================================
# STEP 3 - BUSINESS QUESTIONS AND INSIGHTS
# ==========================================

print("\n========================================")
print("BUSINESS QUESTIONS AND INSIGHTS")
print("========================================")

# Question 1: Which product generates the highest sales?
top_product = product_sales.idxmax()
top_product_sales = product_sales.max()

print("\nQ1. Which product generates the highest sales?")
print(f"Answer: {top_product}")
print(f"Sales: {top_product_sales:,.2f}")


# Question 2: Which city has the highest sales?
top_city = city_sales.idxmax()
top_city_sales = city_sales.max()

print("\nQ2. Which city has the highest sales?")
print(f"Answer: {top_city}")
print(f"Sales: {top_city_sales:,.2f}")


# Question 3: Which month has the highest sales?
top_month = monthly_sales.idxmax()
top_month_sales = monthly_sales.max()

print("\nQ3. Which month has the highest sales?")
print(f"Answer: Month {top_month}")
print(f"Sales: {top_month_sales:,.2f}")


# Question 4: Which category contributes the most sales?
top_category = category_sales.idxmax()
top_category_sales = category_sales.max()

print("\nQ4. Which category contributes the most sales?")
print(f"Answer: {top_category}")
print(f"Sales: {top_category_sales:,.2f}")


# Question 5: What is the average order value?
average_order_value = df["Sales"].mean()

print("\nQ5. What is the average order value?")
print(f"Average Order Value: {average_order_value:,.2f}")


# Question 6: Which product has the lowest sales?
lowest_product = product_sales.idxmin()
lowest_product_sales = product_sales.min()

print("\nQ6. Which product has the lowest sales?")
print(f"Answer: {lowest_product}")
print(f"Sales: {lowest_product_sales:,.2f}")


# Question 7: What percentage of sales comes from Electronics?
electronics_sales = category_sales.get("Electronics", 0)

electronics_percentage = (
    electronics_sales / total_sales
) * 100

print("\nQ7. What percentage of sales comes from Electronics?")
print(f"Electronics Sales Percentage: {electronics_percentage:.2f}%")


# ==========================================
# SAVE BUSINESS INSIGHTS REPORT
# ==========================================

insights_path = "outputs/business_insights.txt"

with open(insights_path, "w", encoding="utf-8") as insights:

    insights.write("CODEALPHA TASK 2 - BUSINESS INSIGHTS\n")
    insights.write("=" * 45 + "\n\n")

    insights.write("Q1. Highest Sales Product\n")
    insights.write(f"Product: {top_product}\n")
    insights.write(f"Sales: {top_product_sales:,.2f}\n\n")

    insights.write("Q2. Highest Sales City\n")
    insights.write(f"City: {top_city}\n")
    insights.write(f"Sales: {top_city_sales:,.2f}\n\n")

    insights.write("Q3. Highest Sales Month\n")
    insights.write(f"Month: {top_month}\n")
    insights.write(f"Sales: {top_month_sales:,.2f}\n\n")

    insights.write("Q4. Highest Sales Category\n")
    insights.write(f"Category: {top_category}\n")
    insights.write(f"Sales: {top_category_sales:,.2f}\n\n")

    insights.write("Q5. Average Order Value\n")
    insights.write(f"Average: {average_order_value:,.2f}\n\n")

    insights.write("Q6. Lowest Sales Product\n")
    insights.write(f"Product: {lowest_product}\n")
    insights.write(f"Sales: {lowest_product_sales:,.2f}\n\n")

    insights.write("Q7. Electronics Sales Percentage\n")
    insights.write(f"Percentage: {electronics_percentage:.2f}%\n\n")

print("\nBusiness insights report saved successfully!")
print("File: outputs/business_insights.txt")

# ==========================================
# STEP 4 - FINAL EDA SUMMARY REPORT
# ==========================================

print("\n========================================")
print("FINAL EDA SUMMARY")
print("========================================")

# Data quality checks
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print("\nData Quality:")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Missing Values: {missing_values}")
print(f"Duplicate Rows: {duplicate_rows}")

# Generate final summary report
final_report_path = "outputs/final_eda_summary.txt"

with open(final_report_path, "w", encoding="utf-8") as report:

    report.write("CODEALPHA INTERNSHIP - TASK 2\n")
    report.write("EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write("1. DATASET OVERVIEW\n")
    report.write(f"Total Rows: {len(df)}\n")
    report.write(f"Total Columns: {len(df.columns)}\n")
    report.write(f"Total Sales: {total_sales:,.2f}\n")
    report.write(f"Total Quantity Sold: {total_quantity}\n\n")

    report.write("2. DATA QUALITY\n")
    report.write(f"Missing Values: {missing_values}\n")
    report.write(f"Duplicate Rows: {duplicate_rows}\n\n")

    report.write("3. KEY BUSINESS INSIGHTS\n")
    report.write(f"Highest Sales Product: {top_product}\n")
    report.write(f"Highest Sales City: {top_city}\n")
    report.write(f"Highest Sales Month: {top_month}\n")
    report.write(f"Highest Sales Category: {top_category}\n")
    report.write(f"Average Order Value: {average_order_value:,.2f}\n")
    report.write(f"Lowest Sales Product: {lowest_product}\n")
    report.write(
        f"Electronics Sales Percentage: "
        f"{electronics_percentage:.2f}%\n\n"
    )

    report.write("4. CONCLUSION\n")
    report.write(
        "The exploratory analysis identified sales patterns "
        "across products, cities, categories, and months.\n"
    )
    report.write(
        "The generated charts support the interpretation "
        "of the dataset and business insights.\n"
    )

print("\nFinal EDA summary report created successfully!")
print(f"File: {final_report_path}")