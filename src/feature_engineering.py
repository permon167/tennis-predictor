def create_features(data):
    # Example feature: Win/Loss ratio
    data['win_loss_ratio'] = data['wins'] / (data['losses'] + 1)  # Adding 1 to avoid division by zero

    # Example feature: Average aces per match
    data['average_aces'] = data['aces'] / data['matches_played']

    # Example feature: Average double faults per match
    data['average_double_faults'] = data['double_faults'] / data['matches_played']

    # Example feature: Recent performance (last 5 matches)
    data['recent_performance'] = data['last_5_wins'] / (data['last_5_matches'] + 1)

    return data

def feature_selection(data):
    # Select relevant features for the model
    features = data[['win_loss_ratio', 'average_aces', 'average_double_faults', 'recent_performance']]
    return features

def encode_categorical_features(data):
    # Example encoding for categorical features
    data = pd.get_dummies(data, columns=['surface', 'player_hand'], drop_first=True)
    return data

def preprocess_features(data):
    data = create_features(data)
    data = encode_categorical_features(data)
    return feature_selection(data)