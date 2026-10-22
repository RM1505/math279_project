# LaTeX manuscript and submission plan

Prepared October 22, 2026. This supersedes the four-page Word draft as the main manuscript.

## Files and build

- `SNAPSHOT_OFI_PAPER.pdf`: nine-page research manuscript, including references.
- `SNAPSHOT_OFI_PAPER.tex`: standalone editable LaTeX source. All tables, plot coordinates and references are embedded; no external figures or bibliography files are required.
- `SNAPSHOT_OFI_LATEX.zip`: source, compiled PDF and this guide, suitable for uploading to Overleaf.

Compile with `tectonic SNAPSHOT_OFI_PAPER.tex`, or compile twice with a standard LaTeX installation. On Overleaf, select `SNAPSHOT_OFI_PAPER.tex` as the main document. Standard packages include pgfplots, natbib and amsthm. The local exported PDF was built with Tectonic.

The manuscript is a research draft pending coauthor review, not an assertion of acceptance or established novelty. The existing empirical archive was already explored before this paper, and the bounded transformations and additional robustness checks are explicitly described as post-result analyses.

## Recommended first submission: FINEC 2027, Lancaster

The [official call](https://wp.lancs.ac.uk/finec2027/) invites financial econometrics papers, explicitly including market microstructure, high-frequency econometrics and forecasting. This is a better topical fit than presenting the result as a new AI forecasting method.

- Conference: SoFiE-sponsored joint Financial Econometrics Conference, Lancaster University, April 21–23, 2027.
- Submission deadline: December 1, 2026.
- Submission fee: £30, paid before completing the submission form; the form requires the payment reference.
- Format: PDF paper; the public call does not state a page limit or an anonymity requirement. Check any additional instructions in the live form before uploading.
- All coauthor contact information is required.
- Attendance: entirely in person, with no online sessions. Registration and travel are separate from the submission fee and have not been budgeted here.
- [Submission form](https://lancasteruni.eu.qualtrics.com/jfe/form/SV_09BPvd5c6ru7c2y); follow the payment link from the official call.

This is a competitive conference, not a guaranteed outlet. The main reviewer objection is likely to be that non-additivity under endpoint substitution is familiar and the empirical study does not compare generalized OFI on common data. The strongest response is the paired audit across 483 stocks, explicit distinction between measurement and fit, and honest reporting of the bounded-transform qualification. The current data cannot provide an event-level validation. Conference acceptance would provide feedback and presentation, not itself a refereed journal publication.

## Submission fields prepared

**Title:** What snapshot sampling does to order flow imbalance

**Authors:** Brendan Connelly; Rigel Mummers. Listed authorship follows the existing joint project; both authors must approve the final submission version.

**Abstract:** We study a measurement problem that arose while trying to reproduce order flow imbalance results with minute snapshots. Applying the usual quote-transition formula once to a block’s endpoints generally differs from summing the transitions observed inside it. Across 483 US stocks, the two measures have opposite signs in 17.03% of materially nonzero five-minute comparisons in 2019 (95% interval: 16.82–17.24%), rising to 21.61% at thirty minutes. We derive the aggregation error and give a simple model in which both measures have the same high contemporaneous fit despite persistent sign disagreement. A frozen-queue control explains 54.32% of five-minute return variation; bounding ordinary endpoint OFI raises its fit from 24.25% to 49.88%, an important qualification. Neither observed measure improves our controlled five-minute forward forecast. The evidence supports a measurement check alongside price-fit statistics, rather than a new trading claim. It does not identify the error relative to event-level OFI.

**Keywords:** order flow imbalance; limit order book; temporal aggregation; market microstructure; measurement.

**JEL:** C43; C58; G14.

**Known project email addresses:** Brendan — brendanconnelly@g.ucla.edu; Rigel — rigelmummers@ucla.edu. Current affiliations and corresponding/presenting author need author confirmation; no affiliation has been invented in the paper.

Before the actual paid submission, resolve: approval of the £30 fee, agreement on an in-person presenter if accepted, final coauthor approval, and current affiliations/contact details. The paper, abstract and source package are ready for that review. No payment or conference submission has been made, and no claim of coauthor approval has been entered into a form.

## Other routes

- **Quantitative Finance** is a plausible eventual journal home for a short measurement paper. Its published contents include Research Letters, but the current publisher author-instructions page could not be retrieved (HTTP 403). No current word limit, fee, template or article-type eligibility is asserted here. Check the [official journal instructions](https://www.tandfonline.com/action/authorSubmission?journalCode=rquf20&show=instructions) before adapting or submitting. A direct generalized-OFI comparison or independent event-data validation would materially improve the journal case.
- **CFMR 2027**, May 6–7 in Washington, DC, has a November 18, 2026 deadline and a $50 submission fee according to the [official Darden call](https://wwwprod3.darden.virginia.edu/mayo-center/events/cfmr). Its market-regulation focus is less suitable for the present paper, which does not establish a regulatory effect.
- **ICAIF 2026** is not an available route: the main-paper deadline was August 9 and the workshop-paper deadline was extended to October 12, both past as of this review. See the [main call](https://icaif2026.org/call-for-papers.html) and [workshop call](https://icaif2026.org/call-for-workshop-proposals.html). We have not verified a 2027 call and do not assume its dates.

## Verification added for this manuscript

`paper_robustness.py` reads existing per-stock results, checks matching counts and benchmark losses, and creates `full_study/resolution/paper_robustness.json`. It adds descriptive bootstrap-block sensitivity, the paired frozen-versus-tanh-endpoint loss comparison, and stock-rate quantiles. It does not refit models or search for a new trading strategy. The result labels preserve its post-result status.

The paper keeps the restrictions of the data prominent: minute sums are not event-level ground truth; source-depth groups both use level one; high same-block fit is not forecasting; bounding substantially reduces the original raw fit gap; 2019 is chronologically separated but was not untouched by prior research. Raw data are not redistributed.
