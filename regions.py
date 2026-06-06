import os
import json
import time
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor

# === CONFIGURATION ===
BASE_PATH = "/home/aegis/ether/Research_HEP/Dataset_ver3/MC/reduce_root"
OUTPUT_DIR = "Regions"
LUMI = 36100.0  
REDUCTION = 0.6 
MAX_WORKERS = 2  # Lowered to 2 to ensure RAM stability on large samples
STEP_SIZE = "100MB" # Limits RAM usage per chunk

# Load Metadata
with open("mc_metadata.json", "r") as f:
    METADATA = json.load(f)

BRANCHES = [
    "MET_Core_AnalysisMETAuxDyn_mpx",
    "MET_Core_AnalysisMETAuxDyn_mpy",
    "MET_Core_AnalysisMETAuxDyn_sumet",
    "EventInfoAuxDyn_mcEventWeights",
    "AnalysisJetsAuxDyn_pt",
    "AnalysisJetsAuxDyn_eta",
    "AnalysisJetsAuxDyn_phi",
    "AnalysisJetsAuxDyn_NNJvtPass",
    "AnalysisLargeRJetsAuxDyn_pt",
    "AnalysisLargeRJetsAuxDyn_eta",
    "AnalysisLargeRJetsAuxDyn_phi",
    "AnalysisLargeRJetsAuxDyn_m",
    "AnalysisLargeRJetsAuxDyn_Tau1_wta",
    "AnalysisLargeRJetsAuxDyn_Tau2_wta",
    "AnalysisLargeRJetsAuxDyn_Tau3_wta",
    "AnalysisElectronsAuxDyn_DFCommonElectronsLHTight",
    "AnalysisMuonsAuxDyn_muonType",
    "AnalysisMuonsAuxDyn_quality",
    "AnalysisTauJetsAuxDyn_JetDeepSetTight",
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu",
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"
]

class CutflowTracker:
    def __init__(self):
        self.steps = []
        self.raw_counts = {}
        self.weighted_counts = {}

    def update(self, step_name, events):
        if step_name not in self.steps:
            self.steps.append(step_name)
        n_raw = len(events)
        w_sum = ak.sum(events["weight_phys"]) if n_raw > 0 else 0.0
        self.raw_counts[step_name] = self.raw_counts.get(step_name, 0) + n_raw
        self.weighted_counts[step_name] = self.weighted_counts.get(step_name, 0.0) + w_sum

    def save_csv(self, process_name, output_dir):
        data = []
        initial_w = self.weighted_counts.get(self.steps[0], 1.0)
        prev_w = initial_w
        for step in self.steps:
            raw = self.raw_counts[step]
            weighted = self.weighted_counts[step]
            abs_eff = (weighted / initial_w) * 100 if initial_w > 0 else 0
            rel_eff = (weighted / prev_w) * 100 if prev_w > 0 else 0
            data.append({
                "Step": step, "Raw Events": raw, "Weighted Yield": weighted,
                "Absolute Eff (%)": f"{abs_eff:.2f}", "Relative Eff (%)": f"{rel_eff:.2f}"
            })
            prev_w = weighted
        df = pd.DataFrame(data)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"cutflow_{process_name}.csv"), index=False)

def apply_preselection(events, tracker):
    if len(events) == 0: return events
    tracker.update("Initial", events)

    # 1. Jet Kinematics
    jet_pt = events["AnalysisJetsAuxDyn_pt"] / 1000.0
    jet_eta = events["AnalysisJetsAuxDyn_eta"]
    acc_mask = (abs(jet_eta) < 2.8) & (jet_pt > 30)
    
    # Check leading jet > 250
    leading_pt = ak.pad_none(jet_pt[acc_mask], 1, axis=1)[:, 0]
    pass_kin = (ak.fill_none(leading_pt > 250, False)) & (ak.num(jet_pt[acc_mask]) >= 2)
    events = events[pass_kin]
    tracker.update("Jet Selection", events)
    if len(events) == 0: return events

    # 2. Large-R Jets
    events = events[ak.num(events["AnalysisLargeRJetsAuxDyn_pt"]) >= 2]
    tracker.update("Large-R Jet >= 2", events)
    if len(events) == 0: return events

    # 3. JVT
    jvt_pass = events["AnalysisJetsAuxDyn_NNJvtPass"]
    j_pt = events["AnalysisJetsAuxDyn_pt"] / 1000.0
    j_eta = events["AnalysisJetsAuxDyn_eta"]
    low_pt_mask = (j_pt < 60) & (abs(j_eta) < 2.4)
    pass_jvt = ak.all(ak.where(low_pt_mask, jvt_pass, True), axis=1)
    events = events[pass_jvt]
    tracker.update("JVT Cleaning", events)

    # 4. MET Recalc
    mpx = events["MET_Core_AnalysisMETAuxDyn_mpx"][:, 0] / 1000.0
    mpy = events["MET_Core_AnalysisMETAuxDyn_mpy"][:, 0] / 1000.0
    met = events["MET_Core_AnalysisMETAuxDyn_sumet"][:, 0] / 1000.0
    j_phi = events["AnalysisJetsAuxDyn_phi"]
    j_px_sum = ak.sum((events["AnalysisJetsAuxDyn_pt"] / 1000.0) * np.cos(j_phi), axis=1)
    j_py_sum = ak.sum((events["AnalysisJetsAuxDyn_pt"] / 1000.0) * np.sin(j_phi), axis=1)
    
    re_px, re_py = -(j_px_sum + mpx), -(j_py_sum + mpy)
    events["met_recalc_pt"] = met
    events["met_recalc_phi"] = np.arctan2(re_py, re_px)

    events = events[events["met_recalc_pt"] > 250]
    tracker.update("MET > 250", events)
    if len(events) == 0: return events

    # 5. dPhi & B-tagging Prep
    dphi = np.abs(events["AnalysisJetsAuxDyn_phi"] - events["met_recalc_phi"])
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    events = events[ak.any(dphi < 2.0, axis=1)]
    tracker.update("dPhi(Jet, MET) < 2.0", events)

    pb, pc, pu = events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"], events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc"], events["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu"]
    dl1 = np.log(pb / (0.080 * pc + (1 - 0.080) * pu + 1e-10))
    events["n_bjets"] = ak.sum(dl1 > 1.42, axis=1)
    events = events[events["n_bjets"] <= 1]
    tracker.update("B-jet Veto", events)

    # 6. Vetoes
    n_tau = ak.sum(events["AnalysisTauJetsAuxDyn_JetDeepSetTight"] == 1, axis=1)
    events = events[n_tau == 0]
    tracker.update("Tau Veto", events)

    events["n_ele"] = ak.sum(events["AnalysisElectronsAuxDyn_DFCommonElectronsLHTight"] == 1, axis=1)
    events["n_mu"] = ak.sum((events["AnalysisMuonsAuxDyn_quality"] >= 8) or (events["AnalysisMuonsAuxDyn_muonType"] == 0), axis=1)
    
    return events

def process_full_dataset(process_name):
    print(f"--> Starting {process_name}")
    tracker = CutflowTracker()
    subprocesses = list(METADATA[process_name].keys())
    
    for subp in subprocesses:
        subp_dir = os.path.join(BASE_PATH, process_name, f"mc20_13TeV_MC_{subp}")
        if not os.path.exists(subp_dir): continue
        files = [os.path.join(subp_dir, f) for f in os.listdir(subp_dir) if f.endswith(".root")]
        if not files: continue

        meta = METADATA[process_name][subp]
        norm = (meta['xsec_pb'] * LUMI) / (meta['sum_w'] * REDUCTION)

        # ITERATIVE PROCESSING
        for chunk in uproot.iterate([f + ":CollectionTree" for f in files], BRANCHES, step_size=STEP_SIZE, library="ak"):
            chunk["weight_phys"] = chunk["EventInfoAuxDyn_mcEventWeights"][:, 0] * norm
            cleaned = apply_preselection(chunk, tracker)
            print(cleaned["weight_phys"].sum(), len(cleaned), f"Chunk processed for {process_name} - {subp}")
            if len(cleaned) == 0: continue

            # Final Selection
            jet_pt = cleaned["AnalysisJetsAuxDyn_pt"] / 1000.0
            ht = ak.sum(jet_pt[abs(cleaned["AnalysisJetsAuxDyn_eta"]) < 2.8], axis=1)
            kin_mask = (cleaned["met_recalc_pt"] > 600) & (ht > 600)

            regions = {
                "SR": kin_mask & (cleaned["n_ele"] == 0) & (cleaned["n_mu"] == 0) & (cleaned["n_bjets"] <= 1),
                "CR1L": kin_mask & (cleaned["n_mu"] == 1) & (cleaned["n_bjets"] == 0),
                "CR1Lb": kin_mask & (cleaned["n_mu"] == 1) & (cleaned["n_bjets"] == 1),
                "CR2L": kin_mask & (cleaned["n_mu"] == 2) & (cleaned["n_bjets"] == 0)
            }

            for reg, mask in regions.items():
                reg_data = cleaned[mask]
                if len(reg_data) > 0:
                    out_path = os.path.join(OUTPUT_DIR, reg)
                    os.makedirs(out_path, exist_ok=True)
                    # Unique filename per sub-process to allow chunked writing
                    ak.to_parquet(reg_data, os.path.join(out_path, f"{process_name}_{subp}.parquet"))

    tracker.save_csv(process_name, OUTPUT_DIR)
    print(f"<-- Finished {process_name}")

if __name__ == "__main__":
    processes = ["ttbar", "Diboson", "Single_top", "Multijet", "Wjets", "Zjets"]
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_full_dataset, processes)

    print(f"PIPELINE COMPLETE - Duration: {(time.time() - start_time)/60:.2f} min")