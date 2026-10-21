# What endpoint snapshots identify about observed OFI

These are elementary consequences of the standard quote-transition operator, offered as a transparent diagnostic rather than a claim of novel mathematics. An observed minute transition is not an individually observed market event.

## 1. Exact aggregation error

For bid price p and displayed size q, write

B(p₀,q₀;p₁,q₁) = q₁ 1[p₁≥p₀] − q₀ 1[p₁≤p₀].

Checking up, down, and unchanged prices gives

B = (q₁−q₀) + q₀ 1[p₁>p₀] − q₁ 1[p₁<p₀].

For an observed path j=0,…,n, the difference between summing transitions and applying B only to endpoints is exactly

Eᴮ = Σⱼ {qⱼ₋₁ 1[pⱼ>pⱼ₋₁] − qⱼ 1[pⱼ<pⱼ₋₁]} − q₀ 1[pₙ>p₀] + qₙ 1[pₙ<p₀].

All size differences telescope. Apply the same expression to negative ask prices and ask sizes to obtain Eᴬ. Net OFI aggregation error is Eᴮ−Eᴬ. Dividing every term by the same strictly positive block-start mean depth preserves this identity. Independently normalizing each minute before summing defines a different statistic and generally does not preserve this identity.

Flat price paths have zero error even when sizes vary. Monotone prices need not have zero error: multiple upward bid transitions count several queues, whereas the endpoint operator counts only the last queue. The problem is not restricted to price reversals.

## 2. Endpoint non-identification

Keep displayed bid size equal to q>0. Paths p, p+δ, p+2δ, p and p, p−δ, p−2δ, p have the same endpoint price and size, but summed bid increments +q and −q. Keep the ask unchanged above every bid and its size constant; these are valid noncrossed quote paths, and their net-OFI sums likewise differ. The endpoint operator is zero in both cases.

For any endpoint-only estimate a, at least one of |a−q| and |a+q| is at least q. Repeating either closed path k times gives an error lower bound kq across this unrestricted path class. Thus no endpoint-only transformation can recover the summed observed statistic universally. This is a deterministic ambiguity result, not a statement about the conditional expected error under a specified stochastic model. It does not exclude useful model-based estimators, deeper-book information, or bounds under additional path restrictions.

## 3. A price-input null with large contemporaneous fit

Freeze the bid and ask sizes at the block start. The normalized endpoint operator becomes

F = {qᴮ₀ sign(Δbid) + qᴬ₀ sign(Δask)} / D₀.

With equal fixed sizes and a constant proportional spread, F=2 sign(R), where R is the midpoint log return. For a symmetric continuous zero-mean return distribution with finite nonzero second moment, population OLS with an intercept yields

R² = (E|R|)² / E[R²].

For Gaussian R this is 2/π ≈ 63.66%. An independent following return has zero population correlation with F. A seeded simulation checks both statements.

This null removes within-block queue-size changes but retains price changes by construction. It is not an equilibrium model with no order activity, a causal decomposition of observed price impact, or a proof that real OFI adds no information. Strong contemporaneous fit alone cannot distinguish independent flow information from a statistic that already incorporates the price response.

## 4. Practical preservation

At the finest available observation grid, store (sum of bid increments, sum of reversed-ask increments, sum of endpoint depths, number of increments), with boundary timestamps and quote states. Adjacent summaries merge by addition; the common boundary observation contributes no duplicate increment. The depth sum counts the ENDPOINT of each increment once. Normalize after merging if a mean-depth denominator is desired.

`resolution_tools.py` implements this representation, checks shared boundaries, rejects missing observations, and is tested against direct aggregation. This is ordinary additive aggregation applied correctly, not a new compression algorithm. It prevents *additional* loss from replacing an observed path with its endpoints. It cannot reconstruct events missed before the finest snapshots were recorded. The empirical comparisons instead use a common START depth; that distinct normalization is deliberate.

## 5. Equal explanatory fit can coexist with sign disagreement

An analytical extension of the fixed-size null makes the problem sharper. Let minute midpoint log returns Xⱼ be independent N(0,σ²), with constant positive proportional spread and equal constant bid/ask sizes. One valid construction is a positive midpoint following a geometric random walk, with bid and ask fixed fractions below and above it. Over n minutes the normalized fine measure is 2Σsign(Xⱼ), while the endpoint measure is 2sign(ΣXⱼ). Both have population contemporaneous OLS R²=2/π for every n: for the fine measure, Cov(ΣXⱼ,Σsign(Xⱼ))=n E|X|, Var(ΣXⱼ)=nσ² and Var(Σsign(Xⱼ))=n.

Nevertheless, as n grows, the bivariate central limit theorem for (ΣXⱼ,Σsign(Xⱼ)) yields limiting correlation √(2/π). The standard zero-mean bivariate-normal sign-disagreement identity gives

P[sign(fine) ≠ sign(endpoint)] → arccos(√(2/π))/π ≈ 20.6%.

At finite even n the fine measure has a positive probability of being exactly zero; simulations explicitly report that probability and condition sign disagreement on a nonzero fine measure. It vanishes asymptotically. `resolution_null.py` verifies the finite-scale behavior and large-n limit using 100,000 paths at each stated scale.

Thus equal, high explanatory R² does not certify that two measured imbalances even have the same sign. This strengthens the diagnostic interpretation without establishing novelty of the underlying Gaussian/sign identities, or claiming that this null is an empirical model of genuine event flow.
