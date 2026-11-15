"""Curated list of open-access clinical guidelines that ``app.fetch_data`` downloads.

Each entry is a dict::

    title      shown in citations and the UI
    publisher  grouping label
    grading    which evidence-grading scheme the document uses (informational)
    filename   canonical file name inside data/
    urls       candidate locations, tried IN ORDER until one yields a valid PDF. Plain ``https://`` entries
               are direct PDF links. Three special prefixes let the downloader *find* the PDF itself, which
               is far more robust than a hard-coded file URL that the publisher may move:
                   page:<url>   landing page; its HTML is scanned for "Download"/".pdf" links
                   pmc:<PMCID>  PubMed Central / Europe PMC rendered PDF (open-access author manuscripts)
                   doi:<DOI>    looked up on Europe PMC to find an open-access PMC copy
    aliases    other file names the same document is commonly saved under (e.g. the original name when you
               download it with a browser). A PDF already in data/ under an alias is recognised, gets its
               proper title and is not downloaded a second time.
    landing    human-friendly page shown as "Open original document" in the UI
    topic/year informational

Everything here is published openly by WHO (CC BY-NC-SA 3.0 IGO), the CDC (US public domain), the American
Heart Association / American College of Cardiology and KDIGO (free access), or deposited in PubMed Central.
Check each publisher's terms before redistributing the PDFs; this project only downloads them for your own
local indexing.

How reliable are the links?  Every download is validated (PDF header, size, readable by pypdf), so a stale link
can never put a broken file in your index - it is simply reported as failed. Run
``python -m app.fetch_data --check`` to test all links, and ``--list`` to print a table with landing pages if you
prefer to download some files by hand (save them into data/ under the listed file name or any listed alias).
"""
from __future__ import annotations

from typing import Any, Optional

GuidelineSource = dict[str, Any]

GRADE_WHO = "GRADE (strong/conditional recommendation; certainty of evidence)"
GRADE_ACC = "ACC/AHA Class of Recommendation (COR) and Level of Evidence (LOE)"
GRADE_CDC = "CDC recommendation category (A/B) and evidence type (1-4), or narrative evidence review"
GRADE_KDIGO = "KDIGO grade (1 strong / 2 weak) and certainty of evidence (A-D)"
WHO_NAME = "World Health Organization"


def _who(
    isbn: str,
    title: str,
    filename: str,
    topic: str,
    year: int,
    handle: Optional[str] = None,
    grading: str = GRADE_WHO,
    aliases: Optional[list[str]] = None,
) -> GuidelineSource:
    """A WHO publication. The item page (stable, keyed by ISBN) is scanned for the PDF 'Download' link first."""
    landing = f"https://www.who.int/publications/i/item/{isbn}"
    urls = [f"page:{landing}"]
    if handle:  # IRIS direct links as a fallback (older WHO files use '_eng', newer '-eng')
        base = f"https://iris.who.int/bitstream/handle/10665/{handle}/{isbn}"
        urls += [f"{base}-eng.pdf", f"{base}_eng.pdf"]
    return {
        "title": title,
        "publisher": WHO_NAME,
        "grading": grading,
        "filename": filename,
        "urls": urls,
        "aliases": [f"{isbn}-eng.pdf", f"{isbn}_eng.pdf", *(aliases or [])],
        "landing": landing,
        "topic": topic,
        "year": year,
    }


def _mmwr(vol: int, issue: int, title: str, filename: str, topic: str, year: int, grading: str = GRADE_CDC) -> GuidelineSource:
    """A CDC MMWR Recommendations & Reports article (pattern: .../volumes/<vol>/rr/pdfs/rr<vol><issue>a1-H.pdf)."""
    code = f"rr{vol}{issue:02d}a1"
    landing = f"https://www.cdc.gov/mmwr/volumes/{vol}/rr/{code}.htm"
    return {
        "title": title,
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": grading,
        "filename": filename,
        "urls": [f"https://www.cdc.gov/mmwr/volumes/{vol}/rr/pdfs/{code}-H.pdf", f"page:{landing}"],
        "aliases": [f"{code}-H.pdf", f"{code}.pdf"],
        "landing": landing,
        "topic": topic,
        "year": year,
    }


def _aha(
    doi: str,
    title: str,
    filename: str,
    topic: str,
    year: int,
    pmcid: Optional[str] = None,
    grading: str = GRADE_ACC,
    publisher: str = "American Heart Association / American College of Cardiology",
    aliases: Optional[list[str]] = None,
) -> GuidelineSource:
    """A free-access AHA/ACC (or similar) guideline: direct journal PDF, then PubMed Central, then a DOI lookup."""
    urls = [f"https://www.ahajournals.org/doi/pdf/{doi}"]
    if pmcid:
        urls.append(f"pmc:{pmcid}")
    urls.append(f"doi:{doi}")
    return {
        "title": title,
        "publisher": publisher,
        "grading": grading,
        "filename": filename,
        "urls": urls,
        "aliases": aliases or [],
        "landing": f"https://doi.org/{doi}",
        "topic": topic,
        "year": year,
    }


def _ada(section: str, name: str, filename: str, topic: str) -> GuidelineSource:
    """One section of the ADA Standards of Care in Diabetes-2026 (open access on PubMed Central; DOI 10.2337/dc26-Sxxx)."""
    doi = f"10.2337/dc26-{section}"
    return {
        "title": f"ADA Standards of Care in Diabetes-2026: {name}",
        "publisher": "American Diabetes Association",
        "grading": "ADA evidence levels A-E (A = clear evidence from well-conducted RCTs)",
        "filename": filename,
        "urls": [f"doi:{doi}"],
        "aliases": [f"dc26{section}.pdf"],
        "landing": f"https://doi.org/{doi}",
        "topic": topic,
        "year": 2026,
    }


SOURCES: list[GuidelineSource] = [
    # ------------------------------------------------------------------ the original five
    {
        "title": "WHO Guideline for the Pharmacological Treatment of Hypertension in Adults (2021)",
        "publisher": WHO_NAME,
        "grading": GRADE_WHO,
        "filename": "who_hypertension_2021.pdf",
        "urls": [
            "https://iris.who.int/bitstream/handle/10665/344424/9789240033986-eng.pdf",
            "https://apps.who.int/iris/bitstream/handle/10665/344424/9789240033986-eng.pdf",
            "https://www.ncbi.nlm.nih.gov/books/n/who344424/pdf/",
            "page:https://www.who.int/publications/i/item/9789240033986",
        ],
        "aliases": ["9789240033986-eng.pdf"],
        "landing": "https://www.who.int/publications/i/item/9789240033986",
        "topic": "hypertension",
        "year": 2021,
    },
    {
        "title": "WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience (2016)",
        "publisher": WHO_NAME,
        "grading": "GRADE (recommended / context-specific / not recommended)",
        "filename": "who_antenatal_care_2016.pdf",
        "urls": [
            "https://iris.who.int/bitstream/handle/10665/250796/9789241549912-eng.pdf",
            "https://apps.who.int/iris/bitstream/handle/10665/250796/9789241549912-eng.pdf",
            "https://www.ncbi.nlm.nih.gov/books/NBK409108/pdf/Bookshelf_NBK409108.pdf",
        ],
        "aliases": ["Bookshelf_NBK409108.pdf", "9789241549912-eng.pdf"],
        "landing": "https://www.who.int/publications/i/item/9789241549912",
        "topic": "pregnancy / antenatal care",
        "year": 2016,
    },
    {
        "title": "CDC Clinical Practice Guideline for Prescribing Opioids for Pain - United States (2022)",
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": "GRADE (recommendation category A/B; evidence type 1-4)",
        "filename": "cdc_opioid_prescribing_2022.pdf",
        "urls": ["https://www.cdc.gov/mmwr/volumes/71/rr/pdfs/rr7103a1-H.pdf"],
        "aliases": ["rr7103a1-H.pdf"],
        "landing": "https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm",
        "topic": "pain / opioids",
        "year": 2022,
    },
    {
        "title": "CDC Sexually Transmitted Infections Treatment Guidelines (2021)",
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": "Narrative evidence review",
        "filename": "cdc_sti_treatment_2021.pdf",
        "urls": ["https://www.cdc.gov/mmwr/volumes/70/rr/pdfs/rr7004a1-H.pdf"],
        "aliases": ["rr7004a1-H.pdf"],
        "landing": "https://www.cdc.gov/mmwr/volumes/70/rr/rr7004a1.htm",
        "topic": "sexually transmitted infections",
        "year": 2021,
    },
    {
        "title": "2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease",
        "publisher": "American Heart Association / American College of Cardiology",
        "grading": GRADE_ACC,
        "filename": "acc_aha_primary_prevention_2019.pdf",
        "urls": [
            "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7685565&blobtype=pdf",
            "https://europepmc.org/articles/PMC7685565?pdf=render",
        ],
        "aliases": ["nihms-1643235.pdf"],
        "landing": "https://doi.org/10.1016/j.jacc.2019.03.010",
        "topic": "cardiovascular prevention (aspirin, statins, blood pressure, diet, tobacco)",
        "year": 2019,
    },
    # ------------------------------------------------------------------ WHO
    _who("9789240107243", "WHO Consolidated Guidelines on Tuberculosis, Module 4: Treatment and Care (2025)",
         "who_tb_treatment_care_2025.pdf", "tuberculosis", 2025),
    {
        "title": "WHO Guidelines for Malaria (living guideline, latest PDF)",
        "publisher": WHO_NAME,
        "grading": GRADE_WHO,
        "filename": "who_malaria_guidelines.pdf",
        "urls": ["page:https://www.who.int/publications/i/item/guidelines-for-malaria"],
        "aliases": [],
        "landing": "https://www.who.int/publications/i/item/guidelines-for-malaria",
        "topic": "malaria",
        "year": 2025,
    },
    _who("9789240015128", "WHO Guidelines on Physical Activity and Sedentary Behaviour (2020)",
         "who_physical_activity_2020.pdf", "physical activity", 2020, handle="336656"),
    _who("9789240090903", "WHO Guidelines for the Prevention, Diagnosis, Care and Treatment of Chronic Hepatitis B (2024)",
         "who_hepatitis_b_2024.pdf", "hepatitis B", 2024, handle="376353", aliases=["B09012-eng.pdf"]),
    _who("9789240030824", "WHO Guideline for Screening and Treatment of Cervical Pre-cancer Lesions, 2nd ed. (2021)",
         "who_cervical_precancer_2021.pdf", "cervical cancer prevention", 2021, handle="342365"),
    _who("9789240045989", "WHO Recommendations on Maternal and Newborn Care for a Positive Postnatal Experience (2022)",
         "who_postnatal_care_2022.pdf", "postnatal care", 2022, handle="352658"),
    _who("9789240084278", "WHO mhGAP Guideline for Mental, Neurological and Substance Use Disorders (2023)",
         "who_mhgap_2023.pdf", "mental health", 2023, handle="373106"),
    _who("9789241548502", "WHO Recommendations for the Prevention and Treatment of Postpartum Haemorrhage (2012)",
         "who_postpartum_haemorrhage_2012.pdf", "postpartum haemorrhage", 2012, handle="75411"),
    _who("9789241548335", "WHO Recommendations for Prevention and Treatment of Pre-eclampsia and Eclampsia (2011)",
         "who_preeclampsia_2011.pdf", "pre-eclampsia / eclampsia", 2011, handle="44703"),
    _who("9789290225300", "WHO Guidelines for the Management of Snakebites, 2nd ed. (2016)",
         "who_snakebite_2016.pdf", "snakebite envenoming", 2016, handle="249547", grading="Expert consensus / narrative evidence review"),
    _who("9789241504836", "WHO Guideline: Sodium Intake for Adults and Children (2012)",
         "who_sodium_intake_2012.pdf", "salt / sodium intake", 2012, handle="77985"),
    _who("9789241506328", "WHO Guideline: Updates on the Management of Severe Acute Malnutrition in Infants and Children (2013)",
         "who_severe_acute_malnutrition_2013.pdf", "child malnutrition", 2013, handle="95584"),
    _who("9789241550390", "WHO Guidelines for the Pharmacological and Radiotherapeutic Management of Cancer Pain in Adults and Adolescents (2018)",
         "who_cancer_pain_2018.pdf", "cancer pain", 2018, handle="279700"),
    _who("9789240024168", "WHO Guidelines for the Management of Symptomatic Sexually Transmitted Infections (2021)",
         "who_symptomatic_sti_2021.pdf", "sexually transmitted infections", 2021),
    _who("9789240062382", "The WHO AWaRe (Access, Watch, Reserve) Antibiotic Book (2022)",
         "who_aware_antibiotic_book_2022.pdf", "antibiotic selection", 2022, grading="Expert guidance based on evidence reviews"),
    # ------------------------------------------------------------------ CDC (MMWR Recommendations & Reports)
    _mmwr(73, 4, "CDC U.S. Medical Eligibility Criteria for Contraceptive Use (2024)",
          "cdc_contraceptive_mec_2024.pdf", "contraception", 2024, grading="US MEC categories 1-4 (evidence-based)"),
    _mmwr(73, 3, "CDC U.S. Selected Practice Recommendations for Contraceptive Use (2024)",
          "cdc_contraceptive_spr_2024.pdf", "contraception", 2024),
    _mmwr(72, 1, "CDC Universal Hepatitis B Screening and Testing of Adults (2023)",
          "cdc_hepatitis_b_screening_2023.pdf", "hepatitis B", 2023),
    _mmwr(69, 2, "CDC Recommendations for Hepatitis C Screening Among Adults (2020)",
          "cdc_hepatitis_c_screening_2020.pdf", "hepatitis C", 2020),
    _mmwr(73, 5, "CDC Prevention and Control of Seasonal Influenza with Vaccines, ACIP Recommendations 2024-25",
          "cdc_influenza_vaccines_2024_25.pdf", "influenza vaccination", 2024),
    {
        "title": "CDC Diagnosis and Management of Tickborne Rickettsial Diseases (2016)",
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": GRADE_CDC,
        "filename": "cdc_tickborne_rickettsial_2016.pdf",
        "urls": [
            "https://www.cdc.gov/mmwr/volumes/65/rr/pdfs/rr6502.pdf",
            "page:https://www.cdc.gov/mmwr/volumes/65/rr/rr6502a1.htm",
        ],
        "aliases": ["rr6502.pdf", "rr6502a1-H.pdf"],
        "landing": "https://www.cdc.gov/mmwr/volumes/65/rr/rr6502a1.htm",
        "topic": "rickettsial / tick-borne diseases",
        "year": 2016,
    },
    {
        "title": "CDC Guideline for Hand Hygiene in Health-Care Settings (2002)",
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": "CDC/HICPAC categories IA, IB, IC, II",
        "filename": "cdc_hand_hygiene_2002.pdf",
        "urls": [
            "https://www.cdc.gov/mmwr/PDF/rr/rr5116.pdf",
            "page:https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5116a1.htm",
        ],
        "aliases": ["rr5116.pdf", "guideline-hand-hygiene-p.pdf"],
        "landing": "https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5116a1.htm",
        "topic": "infection prevention",
        "year": 2002,
    },
    {
        "title": "CDC Guidelines for Preventing the Transmission of Mycobacterium tuberculosis in Health-Care Settings (2005)",
        "publisher": "Centers for Disease Control and Prevention (MMWR)",
        "grading": "CDC/HICPAC categories IA, IB, IC, II",
        "filename": "cdc_tb_infection_control_2005.pdf",
        "urls": [
            "https://www.cdc.gov/mmwr/PDF/rr/rr5417.pdf",
            "page:https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5417a1.htm",
        ],
        "aliases": ["rr5417.pdf"],
        "landing": "https://www.cdc.gov/mmwr/preview/mmwrhtml/rr5417a1.htm",
        "topic": "tuberculosis infection control",
        "year": 2005,
    },
    # ------------------------------------------------------------------ AHA / ACC / ASA, KDIGO, Surviving Sepsis
    _aha("10.1161/CIR.0000000000000625",
         "2018 AHA/ACC Guideline on the Management of Blood Cholesterol (Executive Summary)",
         "acc_aha_cholesterol_2018.pdf", "cholesterol / statins", 2018, pmcid="PMC7403606"),
    _aha("10.1161/CIR.0000000000001063",
         "2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure",
         "acc_aha_heart_failure_2022.pdf", "heart failure", 2022, aliases=["heidenreich-et-al-2022*"]),
    _aha("10.1161/HYP.0000000000000065",
         "2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation and Management of High Blood Pressure in Adults",
         "acc_aha_hypertension_2017.pdf", "hypertension", 2017, aliases=["whelton-et-al-2017*"]),
    _aha("10.1161/CIR.0000000000001193",
         "2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation",
         "acc_aha_atrial_fibrillation_2023.pdf", "atrial fibrillation", 2023, aliases=["joglar-et-al-2023*"]),
    _aha("10.1161/CIR.0000000000001168",
         "2023 AHA/ACC Guideline for the Management of Patients With Chronic Coronary Disease",
         "acc_aha_chronic_coronary_disease_2023.pdf", "coronary artery disease", 2023),
    _aha("10.1161/STR.0000000000000375",
         "2021 AHA/ASA Guideline for the Prevention of Stroke in Patients With Stroke and TIA",
         "aha_asa_stroke_prevention_2021.pdf", "stroke prevention", 2021,
         publisher="American Heart Association / American Stroke Association", aliases=["kleindorfer-et-al-2021*"]),
    {
        "title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "publisher": "KDIGO (Kidney Disease: Improving Global Outcomes)",
        "grading": GRADE_KDIGO,
        "filename": "kdigo_ckd_2024.pdf",
        "urls": [
            "https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf",
            "page:https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        ],
        "aliases": ["KDIGO-2024-CKD-Guideline.pdf"],
        "landing": "https://kdigo.org/guidelines/ckd-evaluation-and-management/",
        "topic": "chronic kidney disease",
        "year": 2024,
    },
    {
        "title": "KDIGO 2022 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease",
        "publisher": "KDIGO (Kidney Disease: Improving Global Outcomes)",
        "grading": GRADE_KDIGO,
        "filename": "kdigo_diabetes_ckd_2022.pdf",
        "urls": [
            "https://kdigo.org/wp-content/uploads/2022/10/KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf",
            "page:https://kdigo.org/guidelines/diabetes-in-ckd/",
        ],
        "aliases": ["KDIGO-2022-Clinical-Practice-Guideline-for-Diabetes-Management-in-CKD.pdf"],
        "landing": "https://kdigo.org/guidelines/diabetes-in-ckd/",
        "topic": "diabetes and kidney disease",
        "year": 2022,
    },
    {
        "title": "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021",
        "publisher": "Society of Critical Care Medicine / ESICM",
        "grading": "GRADE (strong/weak recommendation; quality of evidence)",
        "filename": "surviving_sepsis_2021.pdf",
        "urls": ["pmc:PMC8486643", "doi:10.1007/s00134-021-06506-y"],
        "aliases": [],
        "landing": "https://doi.org/10.1007/s00134-021-06506-y",
        "topic": "sepsis / critical care",
        "year": 2021,
    },
    # ------------------------------------------------------------------ everyday conditions: asthma, COPD, diabetes
    # (GINA and GOLD are free for personal use; the links below were confirmed against the publishers' sites.)
    {
        "title": "GINA 2026 Global Strategy for Asthma Management and Prevention",
        "publisher": "Global Initiative for Asthma (GINA)",
        "grading": "GINA evidence levels A-D",
        "filename": "gina_asthma_2026.pdf",
        "urls": [
            "https://ginasthma.org/wp-content/uploads/2026/05/GINA-2026-Strategy-Report-WMS.pdf",
            "https://ginasthma.org/wp-content/uploads/2025/11/GINA-2025-Update-25_11_08-WMS.pdf",
            "page:https://ginasthma.org/2026-gina-strategy-report/",
        ],
        "aliases": ["GINA-2026-Strategy-Report-WMS.pdf", "GINA-2025-Update-25_11_08-WMS.pdf"],
        "landing": "https://ginasthma.org/2026-gina-strategy-report/",
        "topic": "asthma",
        "year": 2026,
    },
    {
        "title": "GOLD 2026 Global Strategy for the Diagnosis, Management and Prevention of COPD",
        "publisher": "Global Initiative for Chronic Obstructive Lung Disease (GOLD)",
        "grading": "GOLD evidence levels A-D",
        "filename": "gold_copd_2026.pdf",
        "urls": [
            "https://goldcopd.org/wp-content/uploads/2025/12/GOLD-REPORT-2026-v1.3-8Dec2025_WMV.pdf",
            "https://goldcopd.org/wp-content/uploads/2026/01/GOLD-REPORT-2026-v1.3-8Dec2025_WMV2.pdf",
            "page:https://goldcopd.org/2026-gold-report-and-pocket-guide/",
        ],
        "aliases": ["GOLD-REPORT-2026-v1.3-8Dec2025_WMV.pdf", "GOLD-REPORT-2026-v1.3-8Dec2025_WMV2.pdf"],
        "landing": "https://goldcopd.org/2026-gold-report-and-pocket-guide/",
        "topic": "COPD",
        "year": 2026,
    },
    _ada("S002", "Diagnosis and Classification of Diabetes", "ada_2026_s02_diagnosis.pdf", "diabetes diagnosis"),
    _ada("S006", "Glycemic Goals, Hypoglycemia, and Hyperglycemic Crises", "ada_2026_s06_glycemic_goals.pdf", "diabetes glycemic targets"),
    _ada("S009", "Pharmacologic Approaches to Glycemic Treatment", "ada_2026_s09_pharmacologic.pdf", "diabetes medications"),
    _ada("S010", "Cardiovascular Disease and Risk Management", "ada_2026_s10_cardiovascular.pdf", "diabetes and cardiovascular risk"),
]
