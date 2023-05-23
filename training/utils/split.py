from sklearn.model_selection import train_test_split

def split_data(df, col_name, test_size=0.2, val_size=0.2, random_state=42):
    # Extract unique texts
    unique_texts = df[col_name].unique()

    # Split unique texts into train and test
    train_texts, test_texts = train_test_split(unique_texts, 
                                               test_size=test_size, 
                                               random_state=random_state)

    # Split the train texts further into train and validation
    train_texts, val_texts = train_test_split(train_texts, 
                                              test_size=val_size, 
                                              random_state=random_state)

    # Filter the original DataFrame based on the splits
    train_df = df[df[col_name].isin(train_texts)]
    val_df = df[df[col_name].isin(val_texts)]
    test_df = df[df[col_name].isin(test_texts)]

    return train_df, val_df, test_df