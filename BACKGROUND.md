# Background: statistical disclosure control for health data

This primer explains the problem `protect` addresses, the legal and regulatory context, and the method categories the package exposes.

## The problem: re-identification

Even after removing names and direct identifiers, a record described by `(birth_date, sex, ZIP-5)` is uniquely identifiable for 87 % of the US population (Sweeney, 1997). Health data is especially vulnerable because:

- Patients average 50+ distinct ICD codes; rare codes act as near-unique identifiers (Loukides et al., 2010).
- Rare-disease mentions on public forums can dramatically reduce k when linked to anonymized hospital data.
- Dates of service at day-resolution are stronger identifiers than ZIP3 when combined with public records (obituaries, press releases).

## The two functions: data vs result protection

`protect` distinguishes:

**Data protection (input-side):** transforms applied to microdata before analysis or release.
Most verbs in this package operate here — `noise`, `bin`, `shorten`, `collapse`, `pseudonymize`, `insert`, `eliminate`, `swap`, the date verbs.

**Result protection (output-side):** transforms applied to *outputs* — tables, regression results, plots. The single verb `suppress` covers this domain. It dispatches on input type:

- For tables: primary suppression (`min_n`), dominance rule, p%-rule, rounding, fuzzy counts
- For regression results: intercept redaction, confidence-interval widening
- For plot data: hex-bin scatter with sparse-hex suppression, jittered points, suppressed histograms

Both protection layers are usually needed. microdata.no implements ~50% input-side, ~50% output-side controls.

## Legal / regulatory context

### HIPAA (US health data)

Two paths to permitted use of identifiable health information for research:

1. **Safe Harbor** (45 CFR §164.514(b)(2)) — remove the 18 listed identifiers (names, geographic units smaller than state, **all date elements except year**, phone/fax/email/SSN/MRN/plan/account/certificate/vehicle/device IDs, URLs/IPs/biometrics, full-face photos, and "any other unique identifying number, characteristic, or code"). Age > 89 must aggregate to "90+". ZIP3 allowed only if population ≥ 20,000.

2. **Expert Determination** (§164.514(b)(1)) — a qualified statistician certifies that re-identification risk is "very small." Lets you keep month-of-service, finer geography, etc. — but requires defensible methodology and documentation.

The `profile('safe_harbor')` verb implements path 1. For path 2, use `risk()` to produce metrics and `TransformLog` to document the methodology.

### GDPR (EU)

GDPR Article 4(5) and Recital 26 distinguish:

- **Pseudonymization** — still personal data, inside GDPR. Replaces identifiers with pseudonyms but keeps the data re-linkable.
- **Anonymization** — irreversible, no reasonable means of re-identification. Falls outside GDPR.

The EDPB January 2025 guidance reinforces that a hash-only "anonymization" is pseudonymization, not anonymization. `protect.pseudonymize()` produces pseudonymized data; the `TransformLog` records this fact for compliance documentation.

### Norwegian `helseregisterloven`

Governs central health registers (NPR, KPR, Reseptregisteret, etc.). Direct identifiers may be processed without consent only for statutorily named registers. Anonymized / pseudonymized health data can be transferred to non-EEA recipients if links are kept inside Norway. FHI and SIKT are the standard gatekeepers.

Layered with `personopplysningsloven` (GDPR transposition) and `helseforskningsloven` (research-specific).

### microdata.no

The microdata.no service implements 10 numbered "Tiltak" (measures):

- Input-side (data protection): Tiltak 1 (min population ≥ 1,000), Tiltak 6 (min change ≥ 10 units), Tiltak 7 (min population for descriptives)
- Output-side (result protection): Tiltak 2 (winsorization), Tiltak 3 (noise on counts), Tiltak 4 (hexbin), Tiltak 5 (sparse table suppression), Tiltak 8 (3-digit precision), Tiltak 9 (regression intercept suppression), Tiltak 10 (microaggregation + smoothing)

The `profile('microdata_no')` verb implements the input-side guards; the `suppress()` verb handles the output-side rules.

## Methods catalog

| Method | Input or output | `protect` verb |
|---|---|---|
| k-anonymity (measure) | Both | `risk` |
| k-anonymization (enforce) | Input | `profile('k_anonymize')` |
| l-diversity | Input (measure on outputs) | `risk` |
| Local suppression | Input | `eliminate(columns=...)` |
| Cell suppression (table) | Output | `suppress(min_n=...)` |
| Generalization / recoding | Input | `collapse(mapping=...)` |
| Top/bottom coding | Both | `winsorize(method='value')` |
| Numeric or date resolution coarsening | Input | `coarsen` |
| Winsorization | Output (and sometimes input) | `winsorize` |
| Noise addition (continuous) | Both | `noise`, `jitter` |
| Microaggregation | Input | `noise(method='group_mean')` |
| Rank swapping | Input | `swap(method='rank', level='row')` |
| Record swapping | Input | `swap(method='random', level='unit')` |
| PRAM | Input | `swap(method='pram')` |
| Truncation (codes) | Input | `shorten` |
| Pseudonymization | Input | `pseudonymize` |
| Decoy injection | Input | `insert` |
| Subsampling | Input | `eliminate(share=...)` |
| Cell rounding | Output | `suppress(round=...)` |
| Fuzzy counts | Output | `suppress(ranges=...)` |
| Dominance rule (n,k) | Output | `suppress(dominance=...)` |
| p%-rule | Output | `suppress(p_percent=...)` |
| Regression CI widening | Output | `suppress(widen_alpha=...)` |
| Regression intercept redaction | Output | `suppress(redact_intercept=...)` |
| Hexbin / sparse-hex suppression | Output | `suppress(hexbin=True)` |
| Histogram with sparse-bin suppression | Output | `suppress(bin_histogram=True)` |

## The unit-of-protection problem

When an individual has multiple rows (longitudinal / panel data), every perturbation must be unit-consistent: a person's birth-year noise should be the same on every visit, not redrawn per row. The `unit_id=` argument on every data-side verb in `protect` enforces this.

For the record-level verbs (`insert`, `eliminate`, `swap`), `level='unit'` enforces that whole patients are added / dropped / swapped together, never half-patients.

This is the distinguishing feature of `protect` versus single-row tools like ARX. R's `sdcMicro` has comparable support via its `hhId` mechanism.

## Re-identification attacks worth knowing

1. **Sweeney's 87 %** — {ZIP5, DOB, sex} alone uniquely identifies most of the US population. Driven by ZIP5 + DOB being widely available in public records.
2. **ICD code linkage** — Loukides et al. (2010) showed that patient ICD code lists are themselves quasi-identifiers; rare codes (orphan diseases) act as near-unique IDs.
3. **Forum mention attack** — when patients discuss rare conditions in public forums (rare-disease groups), linking those mentions to hospital data dramatically reduces k.
4. **Date-of-service attack** — day-resolution dates combined with obituaries / press releases / sports results identify people.

These attacks motivate the package defaults: year-only dates, rare-code collapsing, ZIP3 with population threshold, unit-consistent noise.

## Sources

- Microdata.no manual: https://microdata.no/manual/konfidensialitet
- sdcMicro: https://cran.r-project.org/web/packages/sdcMicro/sdcMicro.pdf
- HIPAA Safe Harbor: https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/
- EDPB pseudonymization guidance (2025): https://www.edpb.europa.eu/system/files/2025-01/edpb_guidelines_202501_pseudonymisation_en.pdf
- Sweeney L. (2000). "Simple Demographics Often Identify People Uniquely." Carnegie Mellon Data Privacy Working Paper 3.
- Loukides G., Gkoulalas-Divanis A., Malin B. (2010). "Anonymization of electronic medical records for validating genome-wide association studies." PNAS.
