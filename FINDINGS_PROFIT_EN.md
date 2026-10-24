# Which Success Criterion? Long Life versus Cumulative Profit

> **English edition.** The Chinese original is
> [FINDINGS_盈利口径.md](FINDINGS_盈利口径.md); both cover the same analysis and numbers.

**English summary.** The main analysis used survival (`is_open`) as the success criterion. This one
asks whether that criterion is even valid, since survival is not profit. Using an explicitly
parameterised profit model — the dataset contains no revenue, rent or ticket price — it shows that
the density band with the *lowest* closure rate can carry *negative* cumulative profit while the
band with the *highest* closure rate carries the highest. **Survival and profitability can point in
opposite directions.**

---

## 1. The problem

The main analysis treats `is_open` as success. That criterion has a flaw:

- **Survival ≠ profit.** A store can lose money for a decade and stay open (long lease, family
  capital, an owner unwilling to quit).
- Traffic and survival are negatively correlated, but the comparison conflates two different
  quantities: **high traffic over a short life** versus **low traffic over a long life**. Their
  cumulative output cannot be compared directly.
- Doing so requires **rent, average ticket and gross margin** — none of which exist in this dataset.

So what follows is not a profit estimate. It is: **under explicit assumptions, compute the critical
conditions for each strategy, and state which assumption the conclusion depends on.** That is
defensible, and it is the only honest option.

## 2. First, a correction to the previous section

The traffic analysis stated "dense locations have 0.77× the lifespan". That was a **right-censoring
artefact**.

Recomputed on **closed stores only** (complete lifetimes), by density quintile:

| Density quintile | Stores | Mean lifespan | p25 | Median | p75 |
|---|---:|---:|---:|---:|---:|
| lowest | 3,553 | 43.0 mo | 15.1 | 35.5 | 65.2 |
| 2nd | 3,260 | 44.3 | 16.8 | 36.3 | 66.7 |
| 3rd | 3,350 | 44.3 | 16.5 | 36.0 | 66.7 |
| 4th | 3,276 | 43.3 | 15.7 | 34.3 | 64.9 |
| highest | 3,348 | 44.6 | 15.7 | 36.1 | 67.1 |

**Lifespan is essentially identical** (mean 43.0–44.6 months, median 34.3–36.3).

The reason: when medians are computed over *all* stores, dense districts contain more **recently
opened, not-yet-closed** stores, which drags the median down. A textbook mixture-distribution trap.

**Corrected conclusion**:
> Dense districts are not "faster to fail" — they are **high-churn markets** (more entries and more
> exits). The survival duration of an individual store is essentially independent of density.

This matters because it **simplifies** the comparison: if lifespan is the same, the dense-versus-
sparse difference lies **entirely in traffic and cost**, not in how long you last.

## 3. What the data provides

Closed stores only (complete lifetimes), by 1 km density decile:

| Decile | Competitors (median) | Cumulative check-ins (median) | Lifespan (months) | Intensity = cum ÷ life | Price tier |
|---|---:|---:|---:|---:|---:|
| D1 sparsest | 6 | 22 | 39.1 | 0.56 | 1.61 |
| D5 | 36 | 39 | 41.5 | 0.94 | 1.65 |
| D10 densest | 612 | **87** | 40.3 | **2.16** | 1.78 |

**D10 / D1: cumulative 3.95× · lifespan 1.03× · intensity 3.86× · price tier 1.10×**

> **Dense districts deliver about 4× the cumulative traffic with essentially the same lifespan.**
> The advantage comes from being busier per unit time, not from lasting longer.

## 4. The assumption-based model

### 4.1 Formula and the role of each variable

$$\pi_i = \underbrace{V_i \cdot c \cdot t_i \cdot B \cdot m}_{\text{cumulative gross profit}} - \underbrace{M_i \cdot (R_i + F)}_{\text{cumulative fixed cost}}$$

**This model contains no decision variable.** It is not an optimisation model — no objective to
maximise, no constraints, no solver. It is a **group comparison**: the observed cumulative quantities
of ten density deciles are pushed through the same profit-and-loss identity, and the results compared.

| Symbol | Meaning | Role | Value / range |
|---|---|---|---|
| $V_i$ | cumulative check-ins (traffic × lifespan) | **observed** | D1 22 → D10 87 |
| $M_i$ | months operated | **observed** (censored, see §2) | D1 39.1 → D10 40.3 |
| $t_i$ | mean price tier 1–4 | **observed** | D1 1.61 → D10 1.78 |
| $\text{dense}_i$ | density decile index 0→1 | **given by data** (not a choice) | 0, 0.111, …, 1 |
| $t_i$ conversion | ticket multiplier $=1+0.5(t_i-2)$ | **derived** | D1 0.805× → D10 0.890× |
| $c$ | real customer visits per check-in | **assumption** (the one critical unknown) | swept 200–12,000 (11 steps) |
| $B$ | base ticket at price tier 2 | **assumption** | fixed \$15 (not swept) |
| $m$ | gross margin before rent | **assumption** | swept 0.50–0.85 (8 steps) |
| $F$ | other monthly fixed cost | **assumption** | swept \$6,000–\$25,000 (6 steps) |
| $R_{\text{sparse}}$ | monthly rent at D1 | **assumption** | fixed \$8,000 (not swept) |
| $R_i$ | rent $=R_{\text{sparse}}\cdot k^{\text{dense}_i}$ | **assumption** (derived from density, not free) | determined by $k$ and decile |
| $k$ | rent premium of dense over sparse | **scenario parameter** (supplied and swept) | swept 1.5×–6.0× (6 steps) |
| $\pi_i$ | cumulative net profit | **model output** | D1 −276k → D10 +1,075k |
| $k^*$ | break-even rent multiple | **model output** | ≈ 6.3× at the calibration point |

**Sweep size**: 11 × 8 × 6 = **528 combinations**, × 6 values of $k$ = **3,168 evaluations**.

### 4.2 Rent is not a decision variable — and what that implies

Rent enters as an **exogenous parameter, pinned to density**:

$$R_i = R_{\text{sparse}} \cdot k^{\text{dense}_i}$$

Once $k$ and the decile are fixed, rent is fully determined; nothing in the model "chooses" rent.

| The model **can** answer | The model **cannot** answer |
|---|---|
| Is the dense district's rent premium worth the traffic it brings? | If I negotiated 20% off rent in a dense district, would it pay? |
| How high must the rent multiple be before dense stops paying? | How much should I pay, and where? |

Because **rent and traffic are driven by the same variable (density), they are structurally
collinear**: $\text{rent}=f(\text{density})$, $\text{traffic}=g(\text{density})$. To first order
that matches economic intuition — you cannot buy cheap footfall in a prime location — but it means a
single decision ("where to locate") has been split into two quantities that look independent and can
be misread as two separately operable levers.

**Making rent a genuine decision variable would require three things**: real rent data (completely
absent here); rent decoupled from density (or measured separately rather than derived); and an
optimisation framework maximising expected cumulative profit over the joint rent–traffic
distribution. That is **a different model**, not a parameter change.

### 4.3 An endogeneity issue left unaddressed

If **better operators systematically choose dense districts**, the "dense → high traffic" link
contains a **selection effect**, not a pure location effect; rent likewise reflects unobserved
location quality. This model therefore makes **no causal claim** — only conditional comparison.

### 4.4 The one conversion factor that must be supplied from outside

$c$ is the only truly critical unknown. It decides whether check-ins can stand in for customers. So
rather than a single result, the model reports a whole curve:

| Check-in represents | Implied monthly revenue, median store | **Break-even rent multiple (D10÷D1)** |
|---:|---:|---:|
| 200 visits | \$2,482 | **1.39×** |
| 500 | \$6,205 | 1.98× |
| 1,000 | \$12,411 | 2.97× |
| 1,500 | \$18,616 | 3.95× |
| 2,000 | \$24,821 | 4.93× |
| 3,000 | \$37,232 | 6.90× |
| 5,000 | \$62,053 | 10.84× |

### 4.5 Calibration choice

In a competitive industry the **median firm should earn roughly zero economic profit**. Inverting
that condition:

$$c \approx 2{,}716 \quad\Rightarrow\quad \text{implied monthly revenue of the median store} \approx \$33{,}900$$

(reasonable for a small restaurant). At that point the **break-even rent multiple is ≈ 6.3×**.

## 5. Results

![Figure 26](charts/fig26_profit_breakeven.png)

**Read the axes carefully**: the vertical axis is the *actual* rent ratio (D10 ÷ D1) and the curve is
the *break-even* ratio. Dense wins when the actual ratio falls **below** the curve.

- The calibrated break-even value is **6.3×**
- Realistic dense-versus-sparse rent ratios are **2–4×**
- Therefore the actual value sits below the threshold and **the high-traffic strategy wins on
  cumulative profit**

The curve is steep: if $c$ is really only 500 (check-ins being a sparse but high-signal indicator),
the threshold drops to **1.98×**, which lands inside the realistic range and **makes the conclusion
genuinely uncertain**.

### Distribution of cumulative profit

![Figure 27](charts/fig27_survival_vs_profit.png)

| Density decile | Cumulative net profit | Closure rate |
|---|---:|---:|
| D1 sparsest | **−276k** | **24%** |
| D2 | −204k | 27% |
| D3 | −98k | 28% |
| D4 | −29k | 29% |
| D5 | −32k | 31% |
| D6 | +27k | 32% |
| D7 | +338k | 33% |
| D8 | +462k | 37% |
| D9 | +542k | 42% |
| D10 densest | **+1,075k** | **45%** |

**This is the heart of the question:**

> The *lowest*-closure decile (D1) carries **negative** cumulative profit; the *highest*-closure
> decile (D10) carries the **highest**. **Survival and profitability point in opposite directions.**

Why: D1 stores last long (39 months) but earn only 0.56 units of traffic per month, so after rent and
fixed costs they lose money each month — **"staying alive" means slowly bleeding**. D10 stores close
more often, but while alive they earn enough that the cumulative total is clearly positive.

## 6. Answering "long life versus cumulative profit"

**Cumulative profit is the correct criterion; survival is a biased proxy.** Three reasons:

1. **Survival counts "loss-making continuance" as success.** D1–D4 are cumulatively negative under
   reasonable assumptions, yet close at only 24–29% — they "successfully" survive inefficiently.
2. **Survival misses "short but profitable".** A high-traffic store may earn back its entire
   investment in two years and then close; under a survival lens that is a failure, under a profit
   lens it is a success.
3. **Survival comparisons are contaminated by censoring** — the error corrected in §2 is an example.

**None of which means "a long life has no value".** The accurate statement is:

> Survival measures **risk** (probability of failure); cumulative profit measures **return** (value
> created). They are different things, and a decision needs both ends of the picture. Compressing
> them into a single `is_open` boolean loses the key information.

## 7. On the "four-stage life cycle"

The observation is correct, and it exposes a deeper problem with `is_open`: **of the four stages
(introduction, growth, maturity, decline), survival only answers "is it still standing" and says
nothing about which stage it is in.** A store in decline can survive for years — passing as a
"success" until the final day, when it abruptly becomes a failure.

That also explains the profit chart above: a meaningful share of the "low-closure, low-traffic" D1
stores are probably in **decline / zombie** state rather than being stable.

**What the data could do but this round did not**: classify stages from the check-in series by
(1) counting annual check-ins per store, (2) **dividing by that year's platform-wide per-store
mean** to remove the platform trend, then (3) classifying growth / flat / decline from the
normalised slope.

Not done because step (2) amplifies small-sample noise badly in 2020–2021 (the pandemic years) and
introduces spurious signals correlated with the global trend. Doing it robustly needs peer
normalisation stratified by city, which was beyond this round. **It is a concrete next step, not
something the data cannot support.**

## 8. Joint sensitivity: sweeping the three assumptions together

### 8.1 A provable simplification: $F$ cancels

The cumulative-profit difference is

$$\Delta\Pi = c\,B\,m\,\Delta V - \big[M_{10}(R_{10}+F) - M_1(R_1+F)\big]$$

Because lifespan is essentially equal ($M_{10} \approx M_1 \approx M$), the fixed-cost term becomes
$F(M_{10}-M_1)\approx 0$. **So $F$ only shifts the absolute level on both sides; it does not change
who wins.**

Numeric check ($c$ and $m$ fixed, $F$ swept from \$6,000 to \$25,000):

| $c$ | $m$ | $k^*(F{=}6k)$ | $k^*(F{=}12k)$ | $k^*(F{=}25k)$ |
|---:|---:|---:|---:|---:|
| 500 | 0.60 | 1.73 | 1.72 | 1.69 |
| 2,000 | 0.60 | 4.27 | 4.25 | 4.20 |
| 8,000 | 0.75 | 17.56 | 17.53 | 17.49 |

**Maximum movement in $k^*$ is 0.072 (0.4% relative).** In other words:

> The three-dimensional sensitivity degenerates to two dimensions — **only the product $c \times m$
> and the rent ratio $k$ decide the winner**, and $F$ only decides whether the business is
> profitable in absolute terms.

### 8.2 The actual two-dimensional surface

![Figure 28](charts/fig28_sensitivity_surface.png)

$c$ and $m$ enter multiplicatively, so the surface is a family of hyperbolas: **a higher margin
requires a lower $c$.**

Translating "dense wins" into a threshold on $c \times m$:

| Rent ratio $k$ | Minimum $c \times m$ required | Reading |
|---:|---:|---|
| 1.5× | 208 | very easy |
| 2.0× | 389 | easy |
| 3.0× | **750** | needs $c \approx 1{,}000$, $m = 0.75$ |
| 4.0× | 1,111 | needs $c \approx 1{,}500$, $m = 0.75$ |
| 6.0× | 1,834 | needs $c \approx 2{,}400$, $m = 0.75$ |

The calibration point ($c = 2{,}716$, $m = 0.70$) gives $c \times m = 1{,}901$, above the 1,111
threshold for a 4× rent ratio — so under this calibration dense wins **even if rent is 4× higher**.

### 8.3 Who actually wins across the parameter space

Share of the 528 ($c$, $m$, $F$) combinations in which dense wins, by rent ratio:

| Rent ratio $k$ | Share where dense wins | Median profit gap (D10 − D1) |
|---:|---:|---:|
| 1.5× | 88.1% | +692k |
| 2.0× | 73.7% | +530k |
| 2.5× | 63.8% | +369k |
| 3.0× | 56.8% | +208k |
| 4.0× | **46.4%** | −114k |
| 6.0× | 33.0% | −759k |

**This table is the most honest conclusion of this section**:

> Across the realistic 2–4× rent range, the probability that high traffic wins falls from **74% to
> 46%**. The claim is therefore not "high traffic always earns more" but "it depends on how much
> more expensive the dense district actually is" — and at roughly 4× the two strategies genuinely
> tie.

### 8.4 So what is the correct statement

- If you believe check-ins are a sparse signal ($c$ large, e.g. 2,000+) and that dense rent is at
  most 3–4× → **high traffic earns more cumulatively**
- If you believe check-ins are widely used ($c$ small, e.g. 200–500) or that dense rent exceeds 4×
  → **low traffic with a long life earns more cumulatively**
- **Neither can be falsified with this dataset**, because $c$ is not identified

That is not evasion; it states exactly **what information would be required to settle the question**:
real customer counts or real revenue.

## 9. Limitations

1. **Every monetary parameter is an assumption.** $B$, $m$, $R$, $F$ come from plausible industry
   ranges, not from this dataset; only $V$, $M$, $t$ are observed. **The output is a critical
   condition, not a profit.**
2. **$c$ is not identifiable.** There is no real traffic or revenue in the data, so $c$ can only be
   calibrated via "median store breaks even" — itself an assumption (competitive equilibrium). The
   break-even curve makes that uncertainty explicit.
3. **Check-in penetration varies by venue.** Nightlife, younger and urban clienteles check in more,
   so part of the dense district's higher traffic **may be measurement bias**. This is the biggest
   threat: if the bias is large, the 4× cumulative advantage is overstated.
4. **Closed stores only** for the lifespan comparison, which discards "still open and long-lived"
   observations; the censoring argument is given in §2, but it remains a choice.
5. **Within-decile variance is ignored.** The tables report medians/means. D10 contains both stores
   that lose everything in a year and stores that thrive for a decade; decisions cannot rest on
   decile averages alone.
6. **Geometric rent escalation is a setup, not an observation.** Real rent depends on floor area,
   frontage, floor level and lease length, and correlates only weakly with "competitors within 1 km".
7. **No full scan over every parameter.** $c$ was swept one-dimensionally, and $m$/$F$/$t_i$ were
   covered by the joint grid, but the complete surface over all four has not been enumerated.

## 10. Outputs

| File | Contents |
|---|---|
| `analysis/13_profit_scenarios.py` | profit model, calibration, break-even scan |
| `analysis/14_profit_charts.py` | figures 26–27 |
| `analysis/15_sensitivity.py` | joint sensitivity plus the analytical check on $F$ |
| `analysis/16_sensitivity_charts.py` | figures 28–29 |
| `07_profit_by_density.csv` | V/M/t, gross profit, cost and net profit by decile |
| `07_profit_breakeven.csv` | $c$ versus break-even rent multiple |
| `07_sensitivity_*.csv` | $k^*$ surface, 528-combination grid, $F$ level effects |
