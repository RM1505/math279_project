# Optional future event-data validation

**Not a prerequisite:** the completed snapshot study uses the existing data. This document preserves an optional route to a different, event-level replication if access becomes available. No purchase or download is requested now.

Purpose: validate the OFI construction on true event data before purchasing/downloading a large panel.

First request **AAPL, 2017-01-03, ten levels, every order-book event**, with both matching `message` and `orderbook` CSV files, plus any README/export metadata. Preserve a valid opening book state (or pre-open data if available). Do not choose sixty-second or other periodic sampling. If this date is unavailable under the entitlement, select a mutually available full trading day and record the substitution before modeling.

After one-day validation, request **AAPL, MSFT, JPM, JNJ, XOM on the same twenty trading days, 2017-01-03 through 2017-01-31**, ten levels, every event. Include timestamps, event types, and corresponding book states. Download one stock-day first to estimate size. Keep archives compressed and process sequentially. Obtain the cost/storage estimate before expanding or placing a paid order.

This is a methodological pilot, not the full paper universe. After validating it, choose the replication universe and dates based on access and the target paper, then reserve an evaluation period not used in prior research. The proposed January 2017 pilot is already explored and is not a holdout.

What Brendan can provide: the LOBSTER portal URL, whether access is individual or institutional, and where downloaded files will live. Do not send account passwords in chat. If authentication must be performed manually, sign in through the provider's normal interface.

Official reference: [LOBSTER data structure](https://data.lobsterdata.com/info/DataStructure.php). Standard output pairs message rows with order-book rows and retains event timestamps. [Sample files](https://php.lobsterdata.com/info/DataSamples.php) can also validate the loader format, but a single sample stock/day cannot establish multi-asset findings.

Existing local data location: `/Users/brendanconnelly/Documents/raw`. All archive filenames currently inventoried indicate sixty-second sampling; the initial 100-file pilot confirmed this, and the full 2017–2019 reconstruction subsequently checked each selected session against the minute grid. If there is another unsampled export, provide its path instead of downloading duplicates.
