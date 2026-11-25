# Motor cortical gamma organisation following preterm birth — analysis code

Analysis code for the article

> Yang J\*, Sathishkumar S\*, Mitani Y, Hasegawa C, Ikeda T, Tanaka S, Yeom H, Chan KTJ, Waitt A, Yaoi K, Iwasaki S, Saito DN, Hirosawa T, Kikuchi M, Yoshimura Y, An K. **Motor cortical gamma organisation following preterm birth relates to early motor development.** *Brain Communications* (under review). \*Equal contribution.

Nineteen children born full-term and 18 children born preterm (5–7 years) performed a child-friendly right index-finger button-press task during child-customised MEG. Movement-related gamma activity (70–90 Hz) was quantified in the left (contralateral) and right (ipsilateral) precentral gyrus, and its hemispheric lateralisation was related to early motor milestones and motor performance.

This repository contains the Python code for all statistical analyses and data figures reported in the article and in the response to the reviewers. It contains **no participant data**.

## Data availability

The participant data are not publicly available because of privacy and ethical restrictions on paediatric neuroimaging and developmental data. They are available from the corresponding authors upon reasonable request, subject to the applicable ethical and institutional approvals. The input files expected by the code are described below.

## MEG preprocessing (Brainstorm)

Preprocessing, source estimation and time–frequency decomposition were carried out in [Brainstorm](https://neuroimage.usc.edu/brainstorm/) (Tadel et al., 2011), with the following settings:

| Step | Settings |
|---|---|
| Recording | 151-channel child-customised MEG (PQ1151R, Yokogawa/KIT), 2000 Hz, 200 Hz low-pass |
| Preprocessing | down-sampled to 500 Hz; band-pass 0.5–100 Hz; 60 Hz notch (1 Hz bandwidth); bad channels marked by visual inspection |
| Artefacts | ICA (Infomax, 20 components) on the continuous data; ocular and cardiac components removed after visual inspection |
| Epochs | −3 to 3 s around button-press onset; correct responses only (button press 200–2000 ms after target onset); epochs with high-amplitude muscle or movement artefacts within −2 to 2 s excluded by visual inspection |
| Source model | individual MRI, overlapping-spheres head model, weighted minimum-norm estimation, noise covariance from −2000 to −1500 ms, Tikhonov regularisation λ = 0.1 |
| Regions | left and right "precentral" parcels of the Desikan–Killiany atlas; first principal component of each parcel, computed for each trial before the time–frequency transform |
| Time–frequency | Morlet wavelets (seven cycles; central frequency 1 Hz, FWHM 3 s), 1–99 Hz in 2 Hz steps, evoked response removed (induced power), averaged over trials, normalised to percentage change from the mean power at −2000 to −1500 ms |

The resulting per-participant time–frequency files (`timefreq_morlet_*_ersd_time.mat`) are the input to this code.

## Measures

| Measure | Definition |
|---|---|
| Gamma power | mean baseline-normalised power, 70–90 Hz, 0–100 ms after the button press (An et al., 2018) |
| Laterality index | (R<sub>contra</sub> − R<sub>ipsi</sub>) / (R<sub>contra</sub> + R<sub>ipsi</sub>), where R = 1 + power/100 is gamma power as a ratio to baseline |
| Peak gamma frequency | frequency of maximal power within 60–100 Hz and −100 to 200 ms (An et al., 2018); peaks above 95 Hz excluded |
| Beta power | mean power, 15–25 Hz, −200 to 0 ms (pre-movement) and 0 to 500 ms (movement-related) (An et al., 2021) |

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Configuration and input files

Copy `config_example.py` to `config.py` (or set the environment variable `PRETERM_MEG_CONFIG` to the path of a configuration file kept elsewhere) and edit the paths. Keep the output folder outside the repository.

| Setting | Content |
|---|---|
| `TFR_FILES` | CSV with columns `ID`, `Group` (`FT`/`PT`), `side` (`contra`/`ipsi`), `file` (time–frequency file relative to `BRAINSTORM_ROOT`) |
| `BRAINSTORM_ROOT` | Brainstorm database folder (time–frequency files, epoch folders with `brainstormstudy.mat`, channel files of the continuous recordings) |
| `BEHAVIOUR_XLSX` | one row per child: `ID`, `Groups` (0 = full-term, 1 = preterm), `Age_month`, `Gest_Age_Week`, `Gest_Weight`, `Birth_Height`, `Birth_HeadCircumference`, `Head_controll_ca`, `Sitdown_ca`, `Walk_alone_ca` (milestone ages in months, corrected age for preterm children), `Responsetime_Mean`, `Responsetime_SD` (s), `KABC_HandMovement`, `KABC_Achievement`, `VABS_Motor_Gross`, `VABS_Motor_Fine`, `SRS_total` |
| `BACKGROUND_XLSX` | header in the second row: `ID`, `sex`, `hand`, `index` (Edinburgh laterality quotient), `ASD_judge` (`A` = autism spectrum disorder), `Rt_eye`, `Lt_eye` (visual acuity; values starting with `<` = below 0.8), `5y_HS`, `5y_Ht_SD`, `5y_BW_SD`, `VABS_cd_total`, `VABS_dls_total`, `VABS_msd_total`, K-ABC scale scores |
| `CLINICAL_XLSX` (optional) | data from the third row, first ten columns: group, ID, retinopathy of prematurity (1/0), note (e.g. strabismus), glasses worn during MEG (1/0), ADHD Rating Scale total, sex, right-eye and left-eye acuity, correction (`eyeglass`) |
| `FREESURFER_DIR`, `FREESURFER_SUBJECT_PATTERN` | FreeSurfer subjects folder and the folder-name pattern of each group (`{id}` = participant ID) |
| `OUTPUT_DIR` | output folder for tables (Excel), intermediate files and figures |

## Running the analysis

```bash
bash run_all.sh            # all steps in order
cd code && python3 04_gamma_analyses.py   # or a single step (after steps 1 and 2)
```

| Script | Results |
|---|---|
| `01_extract_meg_measures.py` | gamma, peak frequency and beta measures; epochs averaged and rejected; ICA components removed; sensor-to-scalp distances |
| `02_build_dataset.py` | participant-level analysis dataset, including the laterality index |
| `03_participants_behaviour.py` | Table 1; behavioural group comparisons (Fig. 1B); Table 2 (response time × perinatal and motor measures) |
| `04_gamma_analyses.py` | Table 3; Group × Hemisphere mixed ANOVA with Holm-corrected post hoc comparisons (gamma power, peak frequency); laterality index; model checks and Mann–Whitney tests; ICA components, rejected and averaged epochs by group |
| `05_correlations.py` | laterality index × motor milestones and motor scores (whole sample, preterm group; zero-order and partial); exploratory gamma × response-time correlations |
| `06_sensitivity_analyses.py` | covariate adjustment (age, sex, K-ABC Achievement, K-ABC Mental Processing Composite, number of averaged epochs, ADHD Rating Scale); exclusion of children with autism spectrum disorder or visual concerns; preterm children with versus without retinopathy of prematurity; Group × Sex; head position; alternative laterality measures; hemispheric asymmetry versus overall gamma level as predictors of age at independent walking |
| `07_beta.py` | beta-band activity (An et al., 2021 windows) |
| `08_freesurfer.py` | intracranial and corpus callosum volumes; precentral cortical thickness |
| `09_figures.py` | Figures 3 and 4 |
| `10_power.py` | sensitivity power analysis and power for the effect reported by An et al. (2018); needs no participant data |

Statistical conventions: two-tailed tests; independent-samples t-tests with Welch's correction when Levene's test is significant; Cohen's d from the pooled SD; Holm correction within each family of post hoc comparisons; Spearman correlations with Benjamini–Hochberg FDR correction within each family; α = 0.05.

## Version history

- **1.0.1**: `03_participants_behaviour.py` skips a variable when a group has fewer than two values; `06_sensitivity_analyses.py` adds the adjustment for K-ABC Achievement scores and the comparison of preterm children with and without retinopathy of prematurity. The results in the revised article were produced with this version, using participant data re-checked against the original clinical records.
- **1.0.0**: first release.

## References

- An K, et al. Altered gamma oscillations during motor control in children with autism spectrum disorder. *J Neurosci* 2018;38:7878–7886.
- An K, et al. Aberrant brain oscillatory coupling from the primary motor cortex in children with autism spectrum disorders. *NeuroImage Clin* 2021;29:102560.
- Tadel F, et al. Brainstorm: a user-friendly application for MEG/EEG analysis. *Comput Intell Neurosci* 2011;2011:879716.

## Licence

MIT (see `LICENSE`).
