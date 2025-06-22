import pandas as pd


def build_team_encoder():
    all_teams = [
        "BRE", "LIV", "ARS", "CHE", "MCI", "WOL", "BHA", "NEW", "FUL", "SOU", "BOU", "AST", "EVE", "WHU", "MUN", "LEI", "IPS", "NOT", "TOT", "CRY",
        "GIR", "VLL", "RMA", "BET", "MAL", "BAR", "LEG", "VAL", "LAS", "CEL", "ATH", "RSO", "VIL", "SEV", "ESP", "RAY", "ATM", "ALA", "GET", "OSA",
        "GEN", "TOR", "ATA", "INT", "ROM", "JUV", "COM", "LAZ", "NAP", "FIO", "PAR", "BOL", "LEC", "MIL", "UDI", "CAG", "VER", "MON", "EMP", "VEN",
        "RBL", "KIE", "STU", "DOR", "FRE", "FCB", "STP", "HOF", "AUG", "LEV", "MAI", "WER", "EIN", "WOL", "HEI", "UNI", "BOC", "MON",
        "PSG", "ASM", "REI", "REN", "STE", "LEN", "LIL", "STR", "TOU", "BRE", "MPR", "HAV", "ANG", "NAN", "MAR", "NIC", "LYO", "AUX",
        "TWE", "PSV", "FEY", "PEC", "GRO", "UTR", "HER", "AJA", "GAE", "AZA", "ALM", "NAC", "FOR", "HEE", "RKC", "SPA", "WIL", "NEC",
        "BRA", "SPO", "FAR", "EST", "BEN", "POR", "BOA", "RIO", "ARO", "ETA", "FAM", "AVS", "CAS", "MOR", "GUI", "GIL", "NAC", "SAN"
    ]
    team_to_id = {team: idx for idx, team in enumerate(sorted(set(all_teams)))}
    return team_to_id



def match_batch_generator(filepath, batch_size=34):
    """
    Generator to read a CSV file in chunks and process each row into a structured format.
    """

    # Build the team encoder only once
    team_to_id = build_team_encoder()


    for chunk in pd.read_csv(filepath, chunksize=batch_size):
        processed_batch = []

        # Shuffle the chunk to ensure randomness
        # chunk = chunk.sample(frac=1).reset_index(drop=True)

        for _, row in chunk.iterrows():
            try:

                # Parse "ARS - AST" into "ARS", "AST"
                teams = row['Teams'].split(' - ')
                home_team = teams[0].strip()
                away_team = teams[1].strip()


                processed_batch.append({
                    "home_odds": float(row['Home']),
                    "draw_odds": float(row['Draw']),
                    "away_odds": float(row['Away']),
                    "btts_yes_odds": float(row['Yes']),
                    "btts_no_odds": float(row['No']),
                    "home_team_id": team_to_id[home_team] / len(team_to_id),  # normalized
                    "away_team_id": team_to_id[away_team] / len(team_to_id),
                    "gameweek": int(row['Week']),
                    "btts_result": int(row['BTTS_Winner']),
                    "match_result": int(row['Team_Winner'])
                })

            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

        if processed_batch:
            yield processed_batch
