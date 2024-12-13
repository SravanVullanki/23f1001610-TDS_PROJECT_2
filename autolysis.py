# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "chardet",
#   "scipy",
#   "scikit-learn",
#   "pyyaml"
# ]
# ///

import os
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
CONFIG_PATH = "config.yaml"

# Retrieve token from environment variable
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load configuration
def load_config():
    """Load configuration from a YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_PATH}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

config = load_config()

# Function Definitions
def create_or_use_directory(dir_name):
    """Create a directory if it doesn't exist, or use the existing one."""
    os.makedirs(dir_name, exist_ok=True)
    print(f"Using directory: {dir_name}")
    return dir_name

def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    dtype_map = config.get('dtypes', {})
    return pd.read_csv(file_path, encoding=encoding, dtype=dtype_map)

def analyze_data(df):
    """Perform generic data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)

    # Summary statistics for all columns
    summary = df.describe(include='all').to_dict()

    # Counting missing values
    missing_values = df.isnull().sum().to_dict()

    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr().to_dict()

    # Outlier detection using Z-score for numeric columns
    z_scores = numeric_df.apply(stats.zscore, nan_policy='omit')
    outliers = {col: list(numeric_df.index[abs(z_scores[col]) > 3]) for col in numeric_df.columns}

    # Clustering (e.g., KMeans) for numeric columns
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df.fillna(0))  # Replace NaNs with 0 or another strategy
    kmeans = KMeans(n_clusters=config['clustering']['n_clusters'], random_state=42).fit(numeric_scaled)
    
    # Assign clusters to the original DataFrame
    df['cluster'] = pd.Series(kmeans.labels_, index=numeric_df.index)

    # Consolidating analysis results
    analysis = {
        'summary': summary,
        'missing_values': missing_values,
        'correlation': correlation,
        'outliers': outliers,
        'clustering': df[['cluster']].head().to_dict()
    }
    
    print("Data analysis complete.")
    return analysis

def visualize_data(df, output_dir):
    """Generate and save visualizations for all numeric features."""
    sns.set(style="whitegrid")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Correlation heatmap for numeric features
    if not numeric_df.empty:
        plt.figure(figsize=(14, 10))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        print(f"Saved correlation heatmap: {heatmap_path}")
        plt.close()

    # Pairplot for numeric features with clustering (if applicable)
    if 'cluster' in df.columns:
        pairplot = sns.pairplot(df, vars=numeric_df.columns, hue='cluster', palette='viridis', markers=["o", "s"])
        pairplot_path = os.path.join(output_dir, 'pairplot.png')
        pairplot.savefig(pairplot_path)
        print(f"Saved pairplot: {pairplot_path}")
        plt.close()


def generate_dynamic_prompt(analysis, file_path, df):
    """Generate a dynamic prompt for the LLM based on data insights."""
    prompt = f"Analyze the dataset {file_path}. Here is the summary:\n"

    if analysis['missing_values']:
        prompt += "There are missing values. Suggest imputation strategies.\n"
    if analysis['outliers']:
        prompt += "Outliers detected. Suggest methods to handle them.\n"
    if 'clustering' in analysis:
        prompt += f"Clustering applied with {config['clustering']['n_clusters']} clusters.\n"
    
    return prompt

def generate_narrative(prompt):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during narrative generation: {e}")
        return "Narrative generation failed."

def main(file_path):
    print("Starting autolysis process...")

    output_dir = create_or_use_directory(os.path.splitext(os.path.basename(file_path))[0])

    print("Loading dataset...")
    df = load_data(file_path)

    print("Analyzing data...")
    analysis = analyze_data(df)

    print("Generating visualizations...")
    visualize_data(df, output_dir)

    print("Generating narrative...")
    prompt = generate_dynamic_prompt(analysis, file_path, df)
    narrative = generate_narrative(prompt)

    if narrative:
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(narrative)
        print(f"Narrative written to {readme_path}.")

    print("Autolysis process completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
