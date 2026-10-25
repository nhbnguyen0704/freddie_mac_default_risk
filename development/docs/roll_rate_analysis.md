# Roll-Rate Analysis

_Generated 2026-09-24 12:46 by `src/jobs/roll_rate_main.py`._

The worst delinquency a loan reaches in the FIRST half of the performance window, beside the worst it reaches in the SECOND half -- over the 53-month window (26 observed + 27 predicted), anchored in 2016-01-2025-12. The two halves an anchor opens may run past the end of that period; the panel is what bounds them. The window comes from seed vintage analysis, observation basis (data/curated/vintage_seed/suggested_window.json): the age by which 75% of a cohort's OWN bads have arrived, as the median over the 70 cohorts at least that deep -- a self-consistent fixed point, not a share of a pooled total (horizon 122 months); this module splits it and reads the bad definition off the result. Terminations (Zero Balance Code) are states in their own right and absorbing, so a loan that pays off or goes to REO is an outcome rather than a disappearance.

## The bad definition

**No rung on this book is a point of no return.**

INCONCLUSIVE -- no rung reached 75% deterioration among its movers, so on this book every measured rung still cures more often than it rolls across the window. Quoted is the deepest readable rung (90+ DPD, 54.7%), which is a floor on the threshold rather than a recommendation. A book observed over a benign window will do this.

Each row is a rung reached in the **observation half**; the columns are where the loan got to in the **outcome half**. `deteriorated` folds in credit events and `cured` folds in prepayment, because a loan that leaves the book owing nothing has cured and one that leaves through a loss has not. `roll share` is the decisive column: of the loans that resolved in a direction between the halves, the share that went the wrong way. Loans whose outcome half reached the same rung are normally excluded from it -- the window told you nothing new about direction -- and `roll share counts` says where that exclusion applied.

**`Current` is the only rung where staying put is held out.** There it is the good outcome and says nothing about direction. At every delinquent rung, a loan that sits at the same rung through the whole outcome half has NOT cured -- it has been delinquent for the entire window -- so it counts as the failure to recover that it is. This matters most at the deepest rung, which has nowhere worse to go: its only route into `deteriorated` is a COMPLETED credit event, and foreclosure runs 18-36 months from the delinquency, longer than most outcome halves. Held out, that rung would be scored on the one outcome the servicing calendar forbids and could never reach the threshold on any book. Every rung is eligible however few transitions sit behind it, so read `pairs` before trusting a roll share.

| state   |   dpd_threshold |      pairs |   cured (%) |   stayed (%) |   deteriorated (%) |   roll share (%) | roll share counts   |   point_of_no_return |
|:--------|----------------:|-----------:|------------:|-------------:|-------------------:|-----------------:|:--------------------|---------------------:|
| 30 DPD  |               1 | 25,391,509 |       73.99 |        17.96 |               5.66 |            24.21 | moved + stayed      |                False |
| 60 DPD  |               2 |  3,748,608 |       70.31 |         8.92 |              11.49 |            22.49 | moved + stayed      |                False |
| 90+ DPD |               3 | 10,215,173 |       36.66 |        27.28 |              16.91 |            54.66 | moved + stayed      |                False |

Rule applied: the shallowest rung at which at least 75% of the loans FAILED TO RECOVER across the two halves of the performance window -- ending at a worse rung, at a credit event, or still at the same delinquent rung -- rather than recovering to a better rung or prepaying. Staying put counts against the loan at every delinquent rung, because a loan that sat at 60 DPD for the whole window did not cure; only at `Current`, where staying is the good outcome and carries no direction, is it held out.

Read over the **53-month window (26 observed + 27 predicted)**. The ladder stops at `90+ DPD`: deeper rungs measure a servicer's foreclosure timeline rather than a borrower's credit, and splitting them spreads the deepest rows over several thin buckets.

This threshold is half of the label. The other half -- how long to watch -- is what the vintage analysis measures, and it is measured twice: once at 90+ DPD to produce the window split here, and again at this threshold to produce the window that goes into the label.

Computed over 0 performance rows, None to None, anchored in the analysis period 2016-01-2025-12, release (unstamped).

## The analysis period

| loan-months read         |          rows |
|:-------------------------|--------------:|
| inside 2016-01-2025-12   | 1,390,947,554 |
| after the period         |             0 |
| total opened by this run | 1,431,383,840 |

Months before the period were **never opened**. The floor is applied as partition paths named at read time, so 204 of 327 monthly partitions were left out of the read entirely.

An anchor must be inside the period; the two halves it opens need not be. An anchor in the last month of the period is a real observation whose outcome half falls after it, and dropping it would throw away the newest transitions in the matrix.

*Censored* in the exclusions below means an anchor whose window would close after that.

## The window and its two halves

The window is **not decided here**. It comes from the seed vintage analysis, which measured the shortest window holding most of the bads visible anywhere in its panel, using the conventional 90+ DPD as a SEED It is a floor: the panel's total is the bads visible by its horizon, not the bads that eventually arrive.

Source: seed vintage analysis, observation basis (data/curated/vintage_seed/suggested_window.json): the age by which 75% of a cohort's OWN bads have arrived, as the median over the 70 cohorts at least that deep -- a self-consistent fixed point, not a share of a pooled total (horizon 122 months).

| part               |   months | ages covered   | what it is                                     |
|:-------------------|---------:|:---------------|:-----------------------------------------------|
| performance window |       53 | 1 .. 53        | the span the label's `window_months` covers    |
| observation half   |       26 | 1 .. 26        | the FROM state is the worst rung reached in it |
| outcome half       |       27 | 27 .. 53       | the TO state is the worst rung reached in it   |

The odd month, when the window is odd, goes to the **outcome** half: the thing being predicted should never have less time to happen in than the observation had, or the matrix would flatter itself.

Both halves are **calendar offsets from the anchor month**, not row offsets, so a loan that skips three months of reporting contributes the months it actually reported rather than quietly shifting its halves forward. An anchor with nothing observed in a half is dropped and counted under *What was excluded*.

Age 0 -- the anchor month itself -- is in neither half.

## Does the threshold hold at other window lengths?

Every window length picks the same rung, so the threshold does not depend on how long step 1 said to watch.

`12m` -> 90+ DPD, `24m` -> 90+ DPD, `36m` -> 90+ DPD, `53m` -> 90+ DPD

| state   |   12m (%) |        n(12m) |   24m (%) |      n(24m) |   36m (%) |      n(36m) |   53m (%) |      n(53m) |
|:--------|----------:|--------------:|----------:|------------:|----------:|------------:|----------:|------------:|
| Current |      8.78 | 1,157,747,633 |      7.04 | 996,247,633 |      5.78 | 842,692,458 |      4.25 | 637,335,182 |
| 30 DPD  |      37.5 |    19,395,279 |     32.02 |  25,661,000 |     28.22 |  27,668,703 |     24.21 |  25,391,509 |
| 60 DPD  |     46.17 |     3,003,699 |     37.09 |   3,695,066 |     30.19 |   3,916,206 |     22.49 |   3,748,608 |
| 90+ DPD |     77.76 |     9,642,110 |     68.17 |  10,462,827 |     61.61 |  10,604,816 |     54.66 |  10,215,173 |

Read a row left to right: of the loans whose outcome half reached a different rung from that one, the share that reached a worse one. `n` is the transitions behind each figure -- they shrink as the window lengthens, because a longer window needs more months of data after the anchor before it has closed.

## Sensitivity: anchors whose outcome half was empty

`rung_second` is measured over the OUTCOME half, but the terminal flags (`ce`, `pp`, `oe`) are measured over the whole window. So a loan that reached an absorbing state inside the OBSERVATION half has no rows in the outcome half at all -- `rung_second` is NULL, the DPD branch cannot fire -- and still receives a definite `to_state` from first-half evidence. Its `from_state` is read off a truncated half too: a mean of 5.9 months of a nominal 12 at the 24-month window, and 9.5 of 27 at the 54-month one.

These pairs are **kept**, classified by the worst rung reached before the payoff, because the outcome itself is real -- 97.6% of voluntary payoffs are non-delinquent within six months of exit, so they are genuine cures. What is wrong is the WINDOW: a transition that resolved in ten months is being pooled with ones that took twenty-four. So the matrices are read a second time without them, and if the threshold is the same in both, the early-resolving pairs are not driving it.

**The threshold HOLDS with them removed.**

| window   |   pairs (all) |   pairs (both halves) |   dropped (%) | state (all)   | state (both halves)   |   roll share (all) % |   roll share (both halves) % |
|:---------|--------------:|----------------------:|--------------:|:--------------|:----------------------|---------------------:|-----------------------------:|
| 12m      | 1,189,788,721 |         1,109,091,439 |          6.78 | 90+ DPD       | 90+ DPD               |                 0.78 |                         0.81 |
| 24m      | 1,036,066,526 |           891,848,855 |         13.92 | 90+ DPD       | 90+ DPD               |                 0.68 |                         0.74 |
| 36m      |   884,882,183 |           693,351,334 |         21.64 | 90+ DPD       | 90+ DPD               |                 0.62 |                         0.69 |
| 53m      |   676,690,472 |           447,464,141 |         33.87 | 90+ DPD       | 90+ DPD               |                 0.55 |                         0.63 |

The concentration is what makes this worth a section: the affected share of each row climbs with the rung, because a deeply delinquent loan leaves the book far more often than a current one. At the 24-month window it is 9.2% of `Current` and 32.2% of `90+ DPD`; at 54 months, 26.6% and 51.3%. The deepest rung is the one the threshold is read off.

`movement_both_halves.csv` carries the full folded table for the restricted population.

## The analysed population

944,764 of 28,595,976 loans are held out of this analysis. 27,651,212 remain. 1,407,774 were ever in a workout (modification, payment deferral, forbearance or repayment plan, or principal made non-interest-bearing) and 944,764 were ever delinquent due to a declared disaster. These loans were removed only during the bad definition analysis. These loans were included in the model training with tags of special mentioned events.

## Transition matrices, row %

Read one row at a time: *given* the worst rung a loan reached in the observation half, where did it get to in the outcome half? Rows sum to 100%.

A terminal state is absorbing, so a loan that prepaid in the first half is `Prepaid` in the outcome half rather than being dropped as unobserved.

**12-month window (6 observed + 6 predicted)**

| from_state   |   Current |   30 DPD |   60 DPD |   90+ DPD |   Prepaid |   Credit event |   Other exit |
|:-------------|----------:|---------:|---------:|----------:|----------:|---------------:|-------------:|
| Current      |     85.91 |     1.06 |     0.08 |      0.09 |     12.77 |              0 |         0.09 |
| 30 DPD       |     53.74 |    27.31 |     5.74 |      4.15 |      8.31 |           0.01 |         0.74 |
| 60 DPD       |     20.04 |    18.84 |    20.01 |     24.82 |     13.59 |           0.16 |         2.53 |
| 90+ DPD      |      8.66 |     3.42 |      2.6 |     65.96 |       6.6 |           8.44 |         4.31 |

**24-month window (12 observed + 12 predicted)**

| from_state   |   Current |   30 DPD |   60 DPD |   90+ DPD |   Prepaid |   Credit event |   Other exit |
|:-------------|----------:|---------:|---------:|----------:|----------:|---------------:|-------------:|
| Current      |     73.42 |     1.55 |     0.13 |      0.18 |     24.57 |              0 |         0.15 |
| 30 DPD       |     50.62 |    23.18 |      4.4 |      3.99 |     16.52 |           0.04 |         1.24 |
| 60 DPD       |     20.19 |    17.74 |    15.05 |     20.02 |     22.14 |           0.35 |          4.5 |
| 90+ DPD      |      9.51 |     4.67 |     3.14 |     49.79 |     11.82 |           12.6 |         8.48 |

**36-month window (18 observed + 18 predicted)**

| from_state   |   Current |   30 DPD |   60 DPD |   90+ DPD |   Prepaid |   Credit event |   Other exit |
|:-------------|----------:|---------:|---------:|----------:|----------:|---------------:|-------------:|
| Current      |     61.44 |     1.84 |     0.15 |      0.23 |     36.14 |              0 |          0.2 |
| 30 DPD       |     45.82 |    20.54 |     3.63 |       3.5 |     24.73 |           0.07 |         1.71 |
| 60 DPD       |     18.92 |     16.5 |    11.93 |     15.82 |     29.87 |           0.48 |         6.48 |
| 90+ DPD      |      8.48 |     5.24 |     3.32 |     38.57 |     16.35 |          15.01 |        13.03 |

**53-month window (26 observed + 27 predicted)**

| from_state   |   Current |   30 DPD |   60 DPD |   90+ DPD |   Prepaid |   Credit event |   Other exit |
|:-------------|----------:|---------:|---------:|----------:|----------:|---------------:|-------------:|
| Current      |     43.89 |     1.97 |     0.16 |      0.23 |     53.47 |           0.01 |         0.26 |
| 30 DPD       |     36.62 |    17.96 |     2.86 |      2.72 |     37.37 |           0.09 |         2.39 |
| 60 DPD       |     15.69 |     14.8 |     8.92 |     10.99 |     39.82 |            0.5 |         9.28 |
| 90+ DPD      |      6.71 |     5.38 |     3.09 |     27.28 |     21.49 |          16.91 |        19.14 |

## Transition matrices, counts

**12-month window**

| from_state   |     Current |     30 DPD |    60 DPD |   90+ DPD |     Prepaid |   Credit event |   Other exit |
|:-------------|------------:|-----------:|----------:|----------:|------------:|---------------:|-------------:|
| Current      | 994,667,651 | 12,252,462 |   920,349 | 1,045,156 | 147,835,683 |          7,465 |    1,018,867 |
| 30 DPD       |  10,422,971 |  5,296,337 | 1,114,121 |   805,662 |   1,610,795 |          2,779 |      142,614 |
| 60 DPD       |     601,825 |    565,959 |   601,186 |   745,659 |     408,244 |          4,720 |       76,106 |
| 90+ DPD      |     834,727 |    330,019 |   250,693 | 6,360,104 |     636,640 |        814,209 |      415,718 |

**24-month window**

| from_state   |     Current |     30 DPD |    60 DPD |   90+ DPD |     Prepaid |   Credit event |   Other exit |
|:-------------|------------:|-----------:|----------:|----------:|------------:|---------------:|-------------:|
| Current      | 731,467,691 | 15,466,350 | 1,262,216 | 1,771,060 | 244,755,564 |         22,074 |    1,502,678 |
| 30 DPD       |  12,989,103 |  5,948,234 | 1,130,068 | 1,024,913 |   4,240,366 |         11,017 |      317,299 |
| 60 DPD       |     746,079 |    655,457 |   556,090 |   739,752 |     818,231 |         13,057 |      166,400 |
| 90+ DPD      |     994,580 |    488,244 |   328,826 | 5,209,816 |   1,236,650 |      1,317,959 |      886,752 |

**36-month window**

| from_state   |     Current |     30 DPD |    60 DPD |   90+ DPD |     Prepaid |   Credit event |   Other exit |
|:-------------|------------:|-----------:|----------:|----------:|------------:|---------------:|-------------:|
| Current      | 517,715,740 | 15,477,489 | 1,278,321 | 1,901,157 | 304,590,789 |         41,712 |    1,687,250 |
| 30 DPD       |  12,676,935 |  5,682,886 | 1,003,449 |   969,405 |   6,843,689 |         19,212 |      473,127 |
| 60 DPD       |     740,823 |    646,118 |   467,394 |   619,570 |   1,169,660 |         18,765 |      253,876 |
| 90+ DPD      |     899,374 |    555,586 |   352,320 | 4,090,321 |   1,733,756 |      1,591,874 |    1,381,585 |

**53-month window**

| from_state   |     Current |     30 DPD |    60 DPD |   90+ DPD |     Prepaid |   Credit event |   Other exit |
|:-------------|------------:|-----------:|----------:|----------:|------------:|---------------:|-------------:|
| Current      | 279,716,074 | 12,555,980 | 1,016,485 | 1,492,089 | 340,812,688 |         53,098 |    1,688,768 |
| 30 DPD       |   9,298,599 |  4,561,374 |   725,815 |   690,302 |   9,487,634 |         21,996 |      605,789 |
| 60 DPD       |     587,972 |    554,845 |   334,221 |   411,949 |   1,492,859 |         18,772 |      347,990 |
| 90+ DPD      |     684,957 |    549,254 |   315,512 | 2,787,074 |   2,195,637 |      1,727,691 |    1,955,048 |

## Cure, inertia, deterioration

The same matrices folded into the directions a loan can move. An `Other exit` -- Freddie repurchasing or selling the loan -- is counted as neither cure nor deterioration, because it is a decision of Freddie's rather than an outcome the borrower produced.

**12-month window**

| state   |         pairs |   cured (%) |   stayed (%) |   deteriorated (%) |   other exit (%) |   roll share of movers (%) | roll share counts   |
|:--------|--------------:|------------:|-------------:|-------------------:|-----------------:|---------------------------:|:--------------------|
| Current | 1,157,747,633 |       12.77 |        85.91 |               1.23 |             0.09 |                       8.78 | moved               |
| 30 DPD  |    19,395,279 |       62.04 |        27.31 |               9.91 |             0.74 |                       37.5 | moved + stayed      |
| 60 DPD  |     3,003,699 |       52.47 |        20.01 |              24.98 |             2.53 |                      46.17 | moved + stayed      |
| 90+ DPD |     9,642,110 |       21.28 |        65.96 |               8.44 |             4.31 |                      77.76 | moved + stayed      |

**24-month window**

| state   |       pairs |   cured (%) |   stayed (%) |   deteriorated (%) |   other exit (%) |   roll share of movers (%) | roll share counts   |
|:--------|------------:|------------:|-------------:|-------------------:|-----------------:|---------------------------:|:--------------------|
| Current | 996,247,633 |       24.57 |        73.42 |               1.86 |             0.15 |                       7.04 | moved               |
| 30 DPD  |  25,661,000 |       67.14 |        23.18 |               8.44 |             1.24 |                      32.02 | moved + stayed      |
| 60 DPD  |   3,695,066 |       60.07 |        15.05 |              20.37 |              4.5 |                      37.09 | moved + stayed      |
| 90+ DPD |  10,462,827 |       29.13 |        49.79 |               12.6 |             8.48 |                      68.17 | moved + stayed      |

**36-month window**

| state   |       pairs |   cured (%) |   stayed (%) |   deteriorated (%) |   other exit (%) |   roll share of movers (%) | roll share counts   |
|:--------|------------:|------------:|-------------:|-------------------:|-----------------:|---------------------------:|:--------------------|
| Current | 842,692,458 |       36.14 |        61.44 |               2.22 |              0.2 |                       5.78 | moved               |
| 30 DPD  |  27,668,703 |       70.55 |        20.54 |                7.2 |             1.71 |                      28.22 | moved + stayed      |
| 60 DPD  |   3,916,206 |       65.28 |        11.93 |               16.3 |             6.48 |                      30.19 | moved + stayed      |
| 90+ DPD |  10,604,816 |       33.39 |        38.57 |              15.01 |            13.03 |                      61.61 | moved + stayed      |

**53-month window**

| state   |       pairs |   cured (%) |   stayed (%) |   deteriorated (%) |   other exit (%) |   roll share of movers (%) | roll share counts   |
|:--------|------------:|------------:|-------------:|-------------------:|-----------------:|---------------------------:|:--------------------|
| Current | 637,335,182 |       53.47 |        43.89 |               2.37 |             0.26 |                       4.25 | moved               |
| 30 DPD  |  25,391,509 |       73.99 |        17.96 |               5.66 |             2.39 |                      24.21 | moved + stayed      |
| 60 DPD  |   3,748,608 |       70.31 |         8.92 |              11.49 |             9.28 |                      22.49 | moved + stayed      |
| 90+ DPD |  10,215,173 |       36.66 |        27.28 |              16.91 |            19.14 |                      54.66 | moved + stayed      |

## Headline transitions by year

One matrix pooled over the whole panel is an average of several credit environments. This is where they separate. Years are the ANCHOR month's year, over the 53-month window (26 observed + 27 predicted).

| year   |   Current -> 30 DPD |   n(Current) |   30 DPD -> 60 DPD |   n(30 DPD) |   60 DPD -> 90+ DPD |   n(60 DPD) |
|:-------|--------------------:|-------------:|-------------------:|------------:|--------------------:|------------:|
| 2016   |                2.26 |  101,747,245 |               4.52 |   3,816,896 |               12.88 |     853,283 |
| 2017   |                2.07 |  103,391,845 |               2.82 |   4,261,198 |                8.96 |     837,451 |
| 2018   |                1.77 |  104,906,068 |               1.99 |   4,588,757 |                6.47 |     725,516 |
| 2019   |                1.61 |  108,360,836 |               2.24 |   4,340,656 |                9.12 |     517,972 |
| 2020   |                1.81 |  113,468,766 |               2.78 |   4,096,656 |               14.67 |     405,127 |
| 2021   |                2.33 |  105,460,422 |               3.04 |   4,287,346 |               17.92 |     409,259 |

## What was excluded

Per window, because the longer the window the more of the panel runs out before it closes.

| window   | halves   |          kept |   censored (window had not closed) |   nothing observed in a half |   kept (%) |
|:---------|:---------|--------------:|-----------------------------------:|-----------------------------:|-----------:|
| 12m      | 6+6      | 1,189,788,721 |                        115,981,496 |                          556 |      91.12 |
| 24m      | 12+12    | 1,036,066,526 |                        269,703,879 |                          368 |      79.35 |
| 36m      | 18+18    |   884,882,183 |                        420,888,357 |                          233 |      67.77 |
| 53m      | 26+27    |   676,690,472 |                        629,080,195 |                          106 |      51.82 |

Rows with an unparseable delinquency status (`XX` or blank) and no Zero Balance Code: **468,723**. They take part in nothing

**Censored** anchors are the price of a finite as-of date, not an error. An anchor is kept only when every month of its window exists in the panel: a loan that reached an absorbing state early is NOT exempt, because prepayment resolves in a month and default only after foreclosure, so exempting terminations would select the matrix on its own outcome.

**Nothing observed in a half** is different.
