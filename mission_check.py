# mission_check.py
import pandas as pd

for f in ["gen_benapole.csv", "gen_sylhet.csv", "gen_coxs.csv", "gen_chittagong.csv"]:
    df = pd.read_csv(f)
    print("\n==", f, "| rows:", len(df), "| seeds:", df.seed.min(), "-", df.seed.max())
    print(df.groupby(["terrain", "wind"]).agg(
        n=("seed", "size"),
        LARS_cov=("LARS_coverage_pct", "mean"),
        C_cov=("C_coverage_pct", "mean"),
        D_cov=("D_coverage_pct", "mean"),
        LARS_missionOK=("LARS_mission_completed", "mean"),
        C_missionOK=("C_mission_completed", "mean"),
        D_missionOK=("D_mission_completed", "mean"),
        LARS_strand=("LARS_stranded_events", "mean"),
        LARS_success=("LARS_relay_success_rate", "mean"),
    ).round(3).to_string())