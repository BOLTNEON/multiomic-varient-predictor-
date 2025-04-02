import pandas as pd

def load_variant_data(path):
    df = pd.read_csv(path)
    df = df[df['ClinicalSignificance'].isin(['Benign', 'Pathogenic'])]
    df = df[df['Type'] == 'single nucleotide variant']
    return df

def load_expression_data(path):
    return pd.read_csv(path, index_col=0)

def load_protein_data(path):
    return pd.read_csv(path, index_col=0)
