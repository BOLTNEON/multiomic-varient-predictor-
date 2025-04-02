from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_labels(df):
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['ClinicalSignificance'])
    return df

def merge_data(variant_df, expr_df, prot_df):
    common_genes = list(set(variant_df['GeneSymbol']) & set(expr_df.columns) & set(prot_df.columns))
    variant_df = variant_df[variant_df['GeneSymbol'].isin(common_genes)]
    expr_df = expr_df[common_genes]
    prot_df = prot_df[common_genes]
    return variant_df, expr_df, prot_df

def scale_features(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)
