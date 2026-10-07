import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to CSV
#change per party
file_path = "output/manifesto_summary/AIADMK/gemini_manifesto_percentages.csv"

# Read CSV
df = pd.read_csv(file_path)

# Get party name
party = df["party"][0]

# Get category names
categories = list(df.columns[2:])

# Get percentage values
percentages = list(df.iloc[0, 2:])

# Create output folder if it does not exist
output_folder = "output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create pie chart
plt.figure(figsize=(10, 10))

plt.pie(
    percentages,
    labels=categories,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(party + " Manifesto - Category-wise Percentage")

plt.tight_layout()

# Save inside output folder
#change for models
output_path = output_folder + "/" + party + "_manifesto_piechart_qwen.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Pie chart saved at:", output_path)