"""Keep secondary output and multiplicity labels separate from primary results."""
import pandas as pd
import summarize_study as s
from study import OUT
if __name__=='__main__':
    s.SCORES=OUT/'scores_session_control'
    s.OUT=OUT/'session_control_summary';s.OUT.mkdir(exist_ok=True)
    s.main()
    path=s.OUT/'inference.csv'
    d=pd.read_csv(path).rename(columns={'p_holm_primary':'p_holm_sensitivity'})
    d.to_csv(path,index=False)
