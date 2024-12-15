
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel", 
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai  # Make sure you install this library: pip install openai

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

# Core Functions for Data Analysis and Reporting
def analyze_data(df):
    """
    Perform basic data analysis: summary statistics, missing values, and correlation matrix.
    """
    if df.empty:
        raise ValueError("The dataset is empty. Please provide a valid CSV file.")

    summary_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    return summary_stats, missing_values, corr_matrix

def detect_outliers(df):
    """
    Identify outliers using the IQR method.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        return pd.Series(dtype=int, name="Outliers")

    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    return outliers

def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate visualizations: correlation heatmap, outliers plot, and distribution plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Matrix Heatmap
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

    # Outliers Plot
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(os.path.join(output_dir, 'outliers.png'))
        plt.close()

    # Distribution Plot (First Numeric Column)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue', bins=30)
        plt.title(f'Distribution: {numeric_columns[0]}')
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """
    Create a README.md file summarizing the analysis.
    """
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")
        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")
        f.write("## Outliers\n")
        f.write(outliers.to_markdown() + "\n\n")
        if not corr_matrix.empty:
            f.write("## Correlation Matrix\n")
            f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        f.write("## Visualizations\n")
        if not outliers.empty and outliers.sum() > 0:
            f.write("- Outliers: ![Outliers](outliers.png)\n")
        f.write("- Distribution: ![Distribution](distribution.png)\n")
    return readme_path

def generate_story(prompt, context):
    """
    Generate a story using an AI assistant (proxy-based).
    """
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    token = os.getenv("AIPROXY_TOKEN")

    if not token:
        return "Error: AIPROXY_TOKEN not set. Please configure your environment variables."

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n{prompt}"}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "")
    except requests.exceptions.RequestException as e:
        return f"Failed to generate story. Error: {e}"

def main(csv_file):
    """
    Main function to execute the data analysis workflow.
    """
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    try:
        summary_stats, missing_values, corr_matrix = analyze_data(df)
        outliers = detect_outliers(df)

        output_dir = "output"
        visualize_data(corr_matrix, outliers, df, output_dir)
        readme_path = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)

        # Generate a narrative story
        context = f"Dataset Analysis:\nSummary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}"
        story = generate_story("Generate a story from the data analysis.", context)

        # Append story to README
        with open(readme_path, 'a') as f:
            f.write("\n## Story\n")
            f.write(story)

        print(f"Analysis complete! Results saved to {output_dir}/")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Data Analysis Tool")
    parser.add_argument("csv_file", help="Path to the CSV file for analysis")
    args = parser.parse_args()

    main(args.csv_file)
