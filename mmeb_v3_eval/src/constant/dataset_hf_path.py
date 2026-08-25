import os
from src.constant.dataset_hflocal_path import BASE_RAW_DATA_DIR

# (repo, subset, split)
EVAL_DATASET_HF_PATH = {
    # Video-RET
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    # "YouCook2": ("VLM2Vec/YouCook2", None, "val"), # HF version compatibility issue
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),

    # Video-CLS
    "HMDB51": ("VLM2Vec/HMDB51", None, "test"),
    "UCF101": ("VLM2Vec/UCF101", None, "test"),
    "Breakfast": ("VLM2Vec/Breakfast", None, "test"),
    "Kinetics-700": ("VLM2Vec/Kinetics-700", None, "test"),
    "SmthSmthV2": ("VLM2Vec/SmthSmthV2", None, "test"),

    # Video-MRET
    "QVHighlight": ("VLM2Vec/QVHighlight", None, "test"),
    "Charades-STA": ("VLM2Vec/Charades-STA", None, "test"),
    "MomentSeeker": ("VLM2Vec/MomentSeeker", None, "test"),
    "MomentSeeker_1k8": ("VLM2Vec/MomentSeeker_1k8", None, "test"),

    # Video-QA
    "NExTQA": ("VLM2Vec/NExTQA", "MC", "test"),
    "EgoSchema": ("VLM2Vec/EgoSchema", "Subset", "test"),
    "MVBench": ("VLM2Vec/MVBench", None, "train"),
    "Video-MME": ("VLM2Vec/Video-MME", None, "test"),
    "ActivityNetQA": ("VLM2Vec/ActivityNetQA", None, "test"),

    # Visdoc-ViDoRe
    "ViDoRe_arxivqa": ("vidore/arxivqa_test_subsampled_beir", None, "test"),
    "ViDoRe_docvqa": ("vidore/docvqa_test_subsampled_beir", None, "test"),
    "ViDoRe_infovqa": ("vidore/infovqa_test_subsampled_beir", None, "test"),
    "ViDoRe_tabfquad": ("vidore/tabfquad_test_subsampled_beir", None, "test"),
    "ViDoRe_tatdqa": ("vidore/tatdqa_test_beir", None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": (
        "vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy": ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),

    # Visdoc-VisRAG
    "VisRAG_ArxivQA": ("openbmb/VisRAG-Ret-Test-ArxivQA", None, "train"),
    "VisRAG_ChartQA": ("openbmb/VisRAG-Ret-Test-ChartQA", None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA": ("openbmb/VisRAG-Ret-Test-SlideVQA", None, "train"),
    "VisRAG_InfoVQA": ("openbmb/VisRAG-Ret-Test-InfoVQA", None, "train"),
    "VisRAG_PlotQA": ("openbmb/VisRAG-Ret-Test-PlotQA", None, "train"),

    # Visdoc-ViDoSeek
    "ViDoSeek-doc": ("VLM2Vec/ViDoSeek", None, "test"),
    "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page-fixed", None, "test"),
    "MMLongBench-doc": ("VLM2Vec/MMLongBench-doc", None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench-page-fixed", None, "test"),

    # Visdoc-ViDoRe_v2
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2": ("vidore/biomedical_lectures_v2", "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2": ("vidore/economics_reports_v2", "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2": ("vidore/esg_reports_v2", "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": ("vidore/esg_reports_v2", None, "test"),

    # "ViDoRe_esg_reports_v2":("vidore/synthetic_rse_restaurant_filtered_v1.0", None, "test"),
    # "ViDoRe_esg_reports_v2_multilingual":("vidore/synthetic_rse_restaurant_filtered_v1.0_multilingual", None, "test"),
    # "ViDoRe_biomedical_lectures_v2":("vidore/synthetic_mit_biomedical_tissue_interactions_unfiltered", None, "test"),
    # "ViDoRe_biomedical_lectures_v2_multilingual":
    # ("vidore/synthetic_mit_biomedical_tissue_interactions_unfiltered_multilingual", None, "test"),
    # "ViDoRe_economics_reports_v2":("vidore/synthetic_economics_macro_economy_2024_filtered_v1.0", None, "test"),
    # "ViDoRe_economics_reports_v2_multilingual":
    # ("vidore/synthetic_economics_macro_economy_2024_filtered_v1.0_multilingual", None, "test"),
    # "ViDoRe_esg_reports_human_labeled_v2":("vidore/esg_reports_human_labeled_v2", None, "test"),

    # ToolDe
    "ToolDe-Queries-web": (os.path.join(
        BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "web"), "ToolDe-Queries-web", "test"),
    "ToolDe-Queries-code": (os.path.join(
        BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "code"), "ToolDe-Queries-code", "test"),
    "ToolDe-Queries-customized": (os.path.join(
        BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "customized"), "ToolDe-Queries-customized", "test"),

    # Audio-CLS
    "NSynth": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "nsynth-1k"), "NSynth", "test"),
    "ESC-50": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "esc50"), "ESC-50", "test"),
    "UrbanSound8K": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "urbansound8k"), "UrbanSound8K", "test"),
    "SpeechCommands": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "speechcommand-1k"), "SpeechCommands", "test"),
    "CREMA-D": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "creamD"), "CREMA-D", "test"),
    # Audio-RET
    "SoundDescs": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "sounddescs-1k"), "SoundDescs", "test"),
    "Clotho": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "clotho"), "Clotho", "test"),
    "AVE": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "AVE", "AVE_Dataset"), "AVE", "test"),
    "SpeechCOCO": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "speechcoco-1k"), "SpeechCOCO", "validation"),
    # Audio-GND
    "TUTSound": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "tutsound"), "TUTSound", "test"),


    # Memory-RET
    "KnowMeBench": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Episodic", "KnowMeBench"), None, "test"),
    "REALTALK": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Dialogue", "REALTALK"), None, "test"),
    "PeerQA": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Semantic", "PeerQA"), None, "test"),
    "DeepPlanning": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Procedural", "DeepPlanning"), None, "test"),

    # GAE-GUIAct
    "GAE-GUIAct_q2t": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "q2t"),
    "GAE-GUIAct_q2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "q2s"),
    "GAE-GUIAct_s2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "s2s"),
    "GAE-GUIAct_t2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "t2s"),
    # GAE-Mind2Web
    "GAE-Mind2Web_q2t": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "q2t"),
    "GAE-Mind2Web_q2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "q2s"),
    "GAE-Mind2Web_s2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "s2s"),
    "GAE-Mind2Web_t2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "t2s"),


}
