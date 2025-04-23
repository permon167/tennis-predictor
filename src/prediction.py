import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
DATA_PATH = 'data/processed/us_open_data.csv'
df = pd.read_csv(DATA_PATH)

# Construcción de estadísticas acumuladas
player_stats = {}
head_to_head = {}
features = []

for _, row in df.iterrows():
    p1, p2 = row['Player_1'], row['Player_2']
    winner = row['Winner']

    for p in [p1, p2]:
        if p not in player_stats:
            player_stats[p] = {'wins': 0, 'matches': 0}

    p1_wins, p1_matches = player_stats[p1]['wins'], player_stats[p1]['matches']
    p2_wins, p2_matches = player_stats[p2]['wins'], player_stats[p2]['matches']

    p1_winrate = p1_wins / p1_matches if p1_matches > 0 else 0
    p2_winrate = p2_wins / p2_matches if p2_matches > 0 else 0

    # Head-to-head stats
    pair = tuple(sorted([p1, p2]))
    if pair not in head_to_head:
        head_to_head[pair] = {p1: 0, p2: 0}
    h2h_p1 = head_to_head[pair].get(p1, 0)
    h2h_p2 = head_to_head[pair].get(p2, 0)

    label = 1 if winner == p1 else 0

    features.append({
        'p1': p1, 'p2': p2,
        'p1_winrate': p1_winrate, 'p2_winrate': p2_winrate,
        'p1_matches': p1_matches, 'p2_matches': p2_matches,
        'h2h_p1': h2h_p1, 'h2h_p2': h2h_p2,
        'label': label
    })

    player_stats[p1]['matches'] += 1
    player_stats[p2]['matches'] += 1
    player_stats[winner]['wins'] += 1
    head_to_head[pair][winner] += 1

# Crear dataset
training_df = pd.DataFrame(features)

# Entrenar el modelo
X = training_df[['p1_winrate', 'p2_winrate', 'p1_matches', 'p2_matches', 'h2h_p1', 'h2h_p2']]
y = training_df['label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_winner(player1, player2):
    def get_stats(p):
        if p in player_stats:
            s = player_stats[p]
            wr = s['wins'] / s['matches'] if s['matches'] > 0 else 0
            return wr, s['matches']
        return 0, 0  # Jugador nuevo

    p1_wr, p1_matches = get_stats(player1)
    p2_wr, p2_matches = get_stats(player2)

    pair = tuple(sorted([player1, player2]))
    h2h_p1 = head_to_head.get(pair, {}).get(player1, 0)
    h2h_p2 = head_to_head.get(pair, {}).get(player2, 0)

    features = pd.DataFrame([{
        'p1_winrate': p1_wr,
        'p2_winrate': p2_wr,
        'p1_matches': p1_matches,
        'p2_matches': p2_matches,
        'h2h_p1': h2h_p1,
        'h2h_p2': h2h_p2
    }])
    prediction = model.predict(features)[0]
    return player1 if prediction == 1 else player2

# Ejemplo de uso
if __name__ == "__main__":
    print("Predicción:", predict_winner("Auger-Aliassime F.", "Alcaraz C."))
    
    print("Predicción:", predict_winner("Cuevas C.", "Zverev A."))