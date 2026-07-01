# Epidemiology and Risk Factors for Diabetic Retinopathy
Source: Synthesized from International Diabetes Federation estimates, WESDR
(Wisconsin Epidemiologic Study of Diabetic Retinopathy), UKPDS, ACCORD Eye
Study, and pooled global-prevalence meta-analyses (Yau et al., Diabetes Care
2012; Ting et al., Clin Exp Ophthalmol 2016). Paraphrased summary for
internal knowledge-base use.

## Global burden
Diabetic retinopathy is a leading cause of vision loss in working-age
adults (roughly ages 20-74) worldwide. Pooled population-based studies
report any-DR prevalence of approximately 30% among people with diabetes,
with proliferative DR (PDR) present in roughly 1% and some degree of
diabetic macular edema in a further 1-7%, though clinic-based (as opposed
to population-based) surveys tend to report notably higher prevalence
since they capture a population already seeking care, sometimes with
symptoms.

## Non-modifiable risk factors
  - **Duration of diabetes**: the single most consistently reported risk
    factor -- longer duration is associated with steadily increasing DR
    prevalence and severity, largely independent of diabetes type.
  - **Age / age at diagnosis**: interacts with duration; younger age at
    type 1 diagnosis is associated with different long-term trajectories
    than later-onset type 2 diabetes.
  - **Genetic predisposition and ethnicity**: prevalence and progression
    rates vary meaningfully across populations (e.g. some Asian and
    Hispanic cohorts show faster progression patterns than others),
    though the specific genetic loci involved remain an active research
    area.
  - **Pregnancy**: independently associated with accelerated retinopathy
    progression, warranting the shortened pregnancy-specific follow-up
    intervals noted in the referral guidance document.
  - **Baseline retinopathy severity itself**: more severe retinopathy at
    any given exam is the strongest single predictor of future progression
    to vision-threatening disease (see icdr_severity_scale.md for
    quantitative progression-risk figures).

## Modifiable risk factors
  - **Chronic hyperglycemia (elevated HbA1c)**: after diabetes duration,
    the most consistently associated modifiable risk factor across
    observational studies and randomized trials; the Diabetes Control and
    Complications Trial (DCCT) attributed a meaningful fraction of DR risk
    to glycemic exposure specifically, though a majority of variance is
    linked to other/interacting factors.
  - **Hypertension**: randomized evidence (e.g. UKPDS, which randomized
    over 1,000 hypertensive type 2 diabetic patients to tight vs.
    conventional blood-pressure targets) demonstrated that tighter blood
    pressure control reduces DR incidence and progression, even though
    observational studies alone have shown a less consistent association.
  - **Dyslipidemia**: elevated total cholesterol, triglycerides, and LDL,
    together with reduced HDL, are increasingly recognized as independent
    contributors to DR severity, with some cohorts reporting dyslipidemia
    prevalence as high as 70-97% among people with diabetes.
  - **Nephropathy**: diabetic kidney disease and diabetic retinopathy are
    correlated microvascular complications -- presence of one should raise
    clinical suspicion for the other, and both share duration/hyperglycemia/
    hypertension as common upstream drivers.
  - **Smoking and obesity**: reported as risk factors in some but not all
    cohort studies -- evidence is less consistent than for hyperglycemia,
    hypertension, and dyslipidemia.

## Practical implication for a decision-support system
When patient metadata is available (age, diabetes type/duration, HbA1c,
blood pressure, lipid panel), it should be used to contextualize -- not
override -- the image-derived severity grade. A patient with long diabetes
duration, poor glycemic control, and hypertension who presents with
borderline moderate NPDR findings represents a meaningfully higher
progression-risk profile than a recently-diagnosed, well-controlled patient
with the same image-level findings, and this context is appropriate to
surface in the clinical justification and follow-up recommendation, while
still keeping the core severity grade grounded primarily in the retinal
image evidence.