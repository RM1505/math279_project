"""Lossless aggregation of OBSERVED quote-transition increments, not hidden events.

Adjacent chunks must share exactly one boundary state; their increments do not
then overlap. This API checks timestamps and boundary quotes and refuses gaps.
"""
from dataclasses import dataclass
import numpy as np
from resolution_study import side_increment,FlowSummary

@dataclass
class QuoteChunk:
    start: int
    end: int
    first: tuple
    last: tuple
    summary: FlowSummary
    def merge(self,other):
        if self.end!=other.start or self.last!=other.first:raise ValueError('Chunks must share the same boundary observation')
        return QuoteChunk(self.start,other.end,self.first,other.last,self.summary.merge(other.summary))

def summarize_quotes(times,quotes,spacing=60):
    """quotes columns: bid price, bid size, ask price, ask size. Times in seconds."""
    t=np.asarray(times);q=np.asarray(quotes,dtype=float)
    if len(t)<2 or q.shape!=(len(t),4):raise ValueError('At least two aligned observations required')
    if not np.isfinite(t).all() or not np.all(t==np.rint(t)) or spacing<=0:raise ValueError('Use integer seconds and a positive spacing')
    if not np.all(np.diff(t)==spacing):raise ValueError('Missing or duplicate observations')
    if not np.isfinite(q).all() or not np.all((q[:,0]>0)&(q[:,2]>q[:,0])&(q[:,1]>0)&(q[:,3]>0)):raise ValueError('Invalid quote state')
    b=side_increment(q[:-1,0],q[:-1,1],q[1:,0],q[1:,1]).sum()
    a=side_increment(-q[:-1,2],q[:-1,3],-q[1:,2],q[1:,3]).sum()
    s=FlowSummary(float(b),float(a),float(((q[1:,1]+q[1:,3])/2).sum()),len(t)-1)
    return QuoteChunk(int(t[0]),int(t[-1]),tuple(q[0]),tuple(q[-1]),s)
