# Retinal Lesion Definitions (for grounding Lesion Analysis Agent outputs)
Source: Standard ophthalmology terminology, consistent with ETDRS and ICDR
literature. Paraphrased for internal knowledge-base use.

## Microaneurysms (MA)
Small, round, focal dilations of retinal capillaries -- typically the
earliest visible sign of diabetic retinopathy. Appear as tiny red dots on
fundus photography.

## Dot and Blot Hemorrhages (HE)
Intraretinal hemorrhages confined to the compact middle retinal layers,
appearing as round ("dot") or larger irregular ("blot") red lesions.
Their quadrant-wise count and distribution is central to the 4-2-1 severe
NPDR rule.

## Hard Exudates (EX)
Yellow-white lipid deposits resulting from leakage of plasma lipoproteins
through damaged capillaries. Clinically important when located within one
disc diameter of the fovea, since this is a defining feature of diabetic
macular edema.

## Cotton Wool Spots (CWS)
Fluffy white patches representing localized nerve fiber layer infarcts from
retinal capillary non-perfusion (microinfarction), rather than a hemorrhage
or lipid deposit. Their presence signals microvascular ischemia.

## Venous Beading (VB)
Focal narrowing and dilation ("sausage-like" segments) of retinal veins,
reflecting significant retinal ischemia. Definite beading in two or more
quadrants is one of the three 4-2-1 criteria for severe NPDR.

## Intraretinal Microvascular Abnormalities (IRMA)
Abnormal, dilated capillary/shunt vessels within the retina that develop in
areas of capillary non-perfusion. Prominent IRMA in one or more quadrants is
one of the three 4-2-1 criteria for severe NPDR, and IRMA is considered a
precursor lesion to frank neovascularization.

## Neovascularization (NV)
New, fragile vessel growth arising from retinal or optic disc vasculature in
response to ischemia-driven VEGF signaling. Defines proliferative diabetic
retinopathy (PDR). Subclassified as NVD (on/near the optic disc) or NVE
(elsewhere in the retina); NVD is associated with higher risk of vision-
threatening complications.

## Vessel Tortuosity (VT)
Abnormal winding/curvature of retinal vessels, often used as a secondary
severity indicator and studied as an early biomarker of microvascular
stress, though it is not one of the formal ICDR grading criteria.

## Distinguishing IRMA from early neovascularization
This distinction is clinically important and a common source of grading
disagreement between human graders (and a useful check for an automated
system): IRMA vessels are located WITHIN the retina and do not cross the
internal limiting membrane, do not leak significantly on fluorescein
angiography, and do not extend into the vitreous. True neovascularization
grows ON the retinal surface or disc, breaches the internal limiting
membrane, leaks profusely on angiography, and can extend into the
vitreous cavity -- which is why NV (not IRMA) defines the transition from
severe NPDR to PDR. An automated system should treat IRMA as a severe-NPDR
marker and reserve the PDR call specifically for confirmed NV.

## Center-involving vs. non-center-involving hard exudates
Hard exudates are graded differently depending on distance from the
fovea: exudates far from the macula mainly inform NPDR severity staging,
while exudates within one disc diameter of the foveal center are a
defining feature of diabetic macular edema (DME) and should be flagged
separately from the NPDR/PDR severity grade, since DME can co-occur with
any severity level and drives its own referral/treatment pathway (see
aao_followup_referral.md and treatment_protocols.md).