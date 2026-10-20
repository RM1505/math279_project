import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from event_ingest import aggregate,event_contribution

class EventIngest(unittest.TestCase):
    def test_minute_sums_chunk_carry_and_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            messages=pd.DataFrame([[t,1,1,100,100,1] for t in [34200,34230,34260,34261,34320]])
            rows=[[p+1,100,p-1,100] for p in [100,101,102,101,100]]
            messages.to_csv(td/"m.csv",header=False,index=False)
            pd.DataFrame(rows).to_csv(td/"b.csv",header=False,index=False)
            a=aggregate(td/"m.csv",td/"b.csv",levels=1,chunksize=2)
            b=aggregate(td/"m.csv",td/"b.csv",levels=1,chunksize=10)
            pd.testing.assert_frame_equal(a,b)
            np.testing.assert_allclose(a.ofi_1[:2],[4,-4])
            np.testing.assert_allclose(a.mid_end[:2],[102,100])
            self.assertTrue(np.isnan(a.ofi_1.iloc[2]))
    def test_asymmetric_queues(self):
        # Rising ask consumes prior ask queue; falling bid removes prior bid.
        np.testing.assert_allclose(event_contribution([101,300,99,50],[102,20,100,100]),[400])
    def test_requires_initial_state(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            pd.DataFrame([[34201,1,1,100,100,1]]).to_csv(td/"m",header=False,index=False)
            pd.DataFrame([[101,100,99,100]]).to_csv(td/"b",header=False,index=False)
            with self.assertRaisesRegex(ValueError,"opening state"):
                aggregate(td/"m",td/"b",levels=1)

if __name__=="__main__": unittest.main()
