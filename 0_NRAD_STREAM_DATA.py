import os
import ROOT
import json
import numpy as np
import sys
import datetime
import argparse

# ------------------ ARGUMENT PARSER ------------------
parser = argparse.ArgumentParser(description="Process ATLAS data by period")
parser.add_argument("-period", required=True, help="Specify which period to process (e.g., A, B, C, ...)")
args = parser.parse_args()
selected_period = args.period

# Suppress ROOT info/warning messages
ROOT.gErrorIgnoreLevel = ROOT.kError
# List of all nested types you suspect are in your rel_branches
types_to_generate = [
    "vector<float>",
    "vector<int>",
    "vector<double>",
    "vector<unsigned int>"
]

for t in types_to_generate:
    ROOT.gInterpreter.GenerateDictionary(f"ROOT::VecOps::RVec<{t}>", "vector;vector;ROOT/RVec.hxx")
# ------------------ LOGGING ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

today = datetime.date.today().strftime("%Y-%m-%d")
existing_logs = [f for f in os.listdir(log_dir) if f.startswith(today)]
run_number = len(existing_logs) + 1
log_file = os.path.join(log_dir, f"{today}_run{run_number}.txt")

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
print(f"Logging to {log_file}\n")

# ------------------ LOAD METADATA ------------------
with open("ATLAS.json", "r") as f:
    metadata = json.load(f)

# ------------------ HELPERS ------------------
def get_root_links(run):
    links = []
    for meta_run in metadata["metadata"]["_file_indices"]:
        if meta_run["key"].split("_")[3][2:] == run:
            for root_file in meta_run["files"]:
                links.append(root_file["uri"])
    return links

# Define runs grouped by period
period_runs = {
    "B": ["300908","300863","300800","300784","300687","300655","300600","300571","300540","300487","300418","300415","300345"],
    "C": ["302393","302391","302380","302347","302300","302269","302265","302137","302053","301973", "301932","301918","301915","301912"],
    "D": ["303560","303499","303421","303338","303304","303291","303266","303264","303208", "303201","303079","303059","303007", "302956","302925","302919","302872", "302831","302829","302737"],
    "E": ["303892","303846","303832","303819","303817","303811","303726","303638"],
    "F": ["304494","304431","304409","304337","304308","304243","304211","304198","304178","304128","304008","304006","303943"],
    "G": ["306451","306448","306442","306419","306384","306310","306278","306269","305920","305811","305777","305735","305727","305723","305674","305671", "305618","305571","305543", "305380","305293"],
    "I": ["308084","308047","307935","307861","307732","307716","307710","307656","307619","307601","307569", "307539","307514","307454","307394", "307358","307354","307306","307259","307195","307126","307124"],
    "K": ["309759","309674","309640","309516","309440","309390","309375"],
    "A": ["297730","298595","298609","298633","298687","298690","298771","298773","298862","298967","299055","299144","299147","299184","299241","299243","299288","299315","299340","299343","299390","299584","300279","300287"],
    "L": ["310210","310247","310249","310341","310370","310405","310468","310473","310574","310634","310691","310738","310781", "310809", "310863","310872","310969","311071","311170","311244","311287", "311321","311365","311402","311473","311481"]
}

# Convert to dictionary of {period: {run: links}}
all_periods = {
    period: {run: get_root_links(run) for run in runs}
    for period, runs in period_runs.items()
}

# ------------------ REDUCE ROOT ------------------
path_reduce_root = "Dataset_ver4/Data/reduce_root"
os.makedirs(path_reduce_root, exist_ok=True)

rel_branches = [
    # --- Global Event & MET ---
    "MET_Core_AnalysisMETAuxDyn.mpx",
    "MET_Core_AnalysisMETAuxDyn.mpy",
    "MET_Core_AnalysisMETAuxDyn.sumet",
    "EventInfoAuxDyn.mcEventWeights",
    "EventInfoAuxDyn.DFCommonJets_eventClean_LooseBad", # Required for jet cleaning
    "EventInfoAuxDyn.runNumber", 
    
    # --- Small-R Jets (Selection & Cleaning) ---
    "AnalysisJetsAuxDyn.pt",
    "AnalysisJetsAuxDyn.eta",
    "AnalysisJetsAuxDyn.phi",
    "AnalysisJetsAuxDyn.NNJvtPass",
    "AnalysisJetsAuxDyn.SumPtTrkPt500",    # Required for f_ch cleaning
    "AnalysisJetsAuxDyn.EnergyPerSampling", # Required for f_max cleaning
    "AnalysisJetsAuxDyn.NumTrkPt500",       # Required for   Overlap Removal logic
    
    # --- Large-R Jets (AD Features - Expanded) ---
    "AnalysisLargeRJetsAuxDyn.pt",
    "AnalysisLargeRJetsAuxDyn.eta",
    "AnalysisLargeRJetsAuxDyn.phi",
    "AnalysisLargeRJetsAuxDyn.m",
    "AnalysisLargeRJetsAuxDyn.Tau1_wta",
    "AnalysisLargeRJetsAuxDyn.Tau2_wta",
    "AnalysisLargeRJetsAuxDyn.Tau3_wta",
    "AnalysisLargeRJetsAuxDyn.D2", # Highly recommended for AD
    "AnalysisLargeRJetsAuxDyn.C2", # Highly recommended for AD
    
    # --- Lepton & Tau Vetoes ---
    "AnalysisElectronsAuxDyn.pt",
    "AnalysisElectronsAuxDyn.eta",
    "AnalysisElectronsAuxDyn.phi",
    "AnalysisElectronsAuxDyn.DFCommonElectronsLHTight",
    "AnalysisMuonsAuxDyn.pt",
    "AnalysisMuonsAuxDyn.eta",
    "AnalysisMuonsAuxDyn.phi",
    "AnalysisMuonsAuxDyn.muonType",
    "AnalysisMuonsAuxDyn.quality",
    "AnalysisTauJetsAuxDyn.pt",
    "AnalysisTauJetsAuxDyn.eta",
    "AnalysisTauJetsAuxDyn.phi",
    "AnalysisTauJetsAuxDyn.JetDeepSetTight",
    
    # --- Flavor Tagging ---
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc",
    "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb"
]

def reduce_root(period, run, link, c):
    chain = ROOT.TChain("CollectionTree")
    chain.Add(link)
    df = ROOT.RDataFrame(chain)
    outdir = f"{path_reduce_root}/{period}/run{run}"
    os.makedirs(outdir, exist_ok=True)
    df.Snapshot("CollectionTree", f"{outdir}/root_{run}_{c}.root", rel_branches)

# ------------------ MAKE DATASET ------------------
cut_77 = 0.7
fc = 0.3

def make_dataset(period, run, c, reduce_root_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    chain = ROOT.TChain("CollectionTree")
    chain.Add(reduce_root_file)
    df = ROOT.RDataFrame(chain)

    branches = [
        # --- Global Event & MET ---
        "MET_Core_AnalysisMETAuxDyn_mpx",
        "MET_Core_AnalysisMETAuxDyn_mpy",
        "MET_Core_AnalysisMETAuxDyn_sumet",
        # "EventInfoAuxDyn_mcEventWeights",
        "EventInfoAuxDyn_DFCommonJets_eventClean_LooseBad",
        "EventInfoAuxDyn_runNumber", 

        # --- Small-R Jets (Selection, Cleaning, & OR) ---
        "AnalysisJetsAuxDyn_pt",
        "AnalysisJetsAuxDyn_eta",
        "AnalysisJetsAuxDyn_phi",
        "AnalysisJetsAuxDyn_m",
        "AnalysisJetsAuxDyn_NNJvtPass",
        "AnalysisJetsAuxDyn_SumPtTrkPt500",    # For f_ch calculation
        "AnalysisJetsAuxDyn_EnergyPerSampling", # For f_max calculation
        "AnalysisJetsAuxDyn_NumTrkPt500",       # For Overlap Removal

        # --- Large-R Jets (AD Training Features) ---
        "AnalysisLargeRJetsAuxDyn_pt",
        "AnalysisLargeRJetsAuxDyn_eta",
        "AnalysisLargeRJetsAuxDyn_phi",
        "AnalysisLargeRJetsAuxDyn_m",
        "AnalysisLargeRJetsAuxDyn_Tau1_wta",
        "AnalysisLargeRJetsAuxDyn_Tau2_wta",
        "AnalysisLargeRJetsAuxDyn_Tau3_wta",
        "AnalysisLargeRJetsAuxDyn_D2",          # Key substructure discriminator
        "AnalysisLargeRJetsAuxDyn_C2",          # Key substructure discriminator

        # --- Lepton & Tau Vetoes ---
        "AnalysisElectronsAuxDyn_pt",
        "AnalysisElectronsAuxDyn_eta",
        "AnalysisElectronsAuxDyn_phi",
        "AnalysisElectronsAuxDyn_DFCommonElectronsLHTight",
        "AnalysisMuonsAuxDyn_pt",
        "AnalysisMuonsAuxDyn_eta",
        "AnalysisMuonsAuxDyn_phi",
        "AnalysisMuonsAuxDyn_muonType",
        "AnalysisMuonsAuxDyn_quality",
        "AnalysisTauJetsAuxDyn_pt",
        "AnalysisTauJetsAuxDyn_eta",
        "AnalysisTauJetsAuxDyn_phi",
        "AnalysisTauJetsAuxDyn_JetDeepSetTight",

        # --- Flavor Tagging ---
        "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu",
        "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc",
        "BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"
    ]

    data = df.AsNumpy(branches)

    outfile = os.path.join(outdir, f"dataset_{run}_{c}.txt")
    with open(outfile, "w") as fout:
        fout.write("pT_j1 eta_j1 phi_j1 pT_j2 eta_j2 phi_j2 m_jj "
                   "tau21_j1 tau21_j2 tau32_j1 tau32_j2 "
                   "met phi_met min_dPhi ht\n")

        n_events = len(data["AnalysisJetsAuxDyn_pt"])
        for i in range(n_events):
            # --- small-R jets (for cuts) ---
            jet_pt = np.array(data["AnalysisJetsAuxDyn_pt"][i]) / 1000.0
            jet_eta = np.array(data["AnalysisJetsAuxDyn_eta"][i])
            jet_phi = np.array(data["AnalysisJetsAuxDyn_phi"][i])
            rvec_jvt = data["AnalysisJetsAuxDyn_NNJvtPass"][i]
            jvt_pass = np.array([ord(rvec_jvt[j]) for j in range(len(rvec_jvt))], dtype=int)

            pu = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pu"][i])
            pc = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pc"][i])
            pb = np.array(data["BTagging_AntiKt4EMPFlowAuxDyn_DL1dv01_pb"][i])

            rvec_tau = data["AnalysisTauJetsAuxDyn_JetDeepSetTight"][i]
            tau_tight = np.array([ord(rvec_tau[j]) for j in range(len(rvec_tau))], dtype=int)

            rvec_elec = data["AnalysisElectronsAuxDyn_DFCommonElectronsLHTight"][i]
            electron_tight = np.array([ord(rvec_elec[j]) for j in range(len(rvec_elec))], dtype=int)

            rvec_muon = data["AnalysisMuonsAuxDyn_quality"][i]
            muon_Qual = np.array([ord(rvec_muon[j]) for j in range(len(rvec_muon))], dtype=int)
            muon_Type = np.array(data["AnalysisMuonsAuxDyn_muonType"][i])

            mpx = data["MET_Core_AnalysisMETAuxDyn_mpx"][i][0]
            mpy = data["MET_Core_AnalysisMETAuxDyn_mpy"][i][0]
            met = data["MET_Core_AnalysisMETAuxDyn_sumet"][i][0]/1000.0
            phi_met = np.arctan2(mpy, mpx)

            fat_pt = np.array(data["AnalysisLargeRJetsAuxDyn_pt"][i]) / 1000.0
            fat_eta = np.array(data["AnalysisLargeRJetsAuxDyn_eta"][i])
            fat_phi = np.array(data["AnalysisLargeRJetsAuxDyn_phi"][i])
            fat_m   = np.array(data["AnalysisLargeRJetsAuxDyn_m"][i]) / 1000.0
            tau1    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau1_wta"][i])
            tau2    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau2_wta"][i])
            tau3    = np.array(data["AnalysisLargeRJetsAuxDyn_Tau3_wta"][i])

            # --- Cuts ---
            if len(jet_pt) < 2: continue
            if np.sum(np.abs(jet_eta) < 2.8) < 2: continue
            if jet_pt[0] < 250 or jvt_pass[0] != 1: continue
            if jet_pt[1] < 30 or jvt_pass[1] != 1: continue
            dphis = [abs(ROOT.TVector2.Phi_mpi_pi(jphi - phi_met)) for jphi in jet_phi]
            if np.sum(np.array(dphis) < 2.0) <= 1: continue
            if np.sum(np.log(pb/(fc*pc + (1-fc)*pu)) > cut_77) >= 2: continue
            if np.sum(tau_tight == 1) > 0: continue  # Lepton veto
            if np.sum(electron_tight == 1) > 0: continue # Lepton veto
            if np.sum(muon_Type == 0) > 0 and (np.sum((muon_Qual == 8) | (muon_Qual == 9) > 0)): continue # Lepton veto
            if len(fat_pt) < 2: continue

            j1 = ROOT.TLorentzVector(); j1.SetPtEtaPhiM(fat_pt[0], fat_eta[0], fat_phi[0], fat_m[0])
            j2 = ROOT.TLorentzVector(); j2.SetPtEtaPhiM(fat_pt[1], fat_eta[1], fat_phi[1], fat_m[1])
            m_jj = (j1+j2).M()

            tau21_j1 = tau2[0]/tau1[0] if tau1[0] > 0 else -1
            tau21_j2 = tau2[1]/tau1[1] if tau1[1] > 0 else -1
            tau32_j1 = tau3[0]/tau2[0] if tau2[0] > 0 else -1
            tau32_j2 = tau3[1]/tau2[1] if tau2[1] > 0 else -1
            min_dPhi = np.min(dphis)
            ht = np.sum(jet_pt)

            row = [
                fat_pt[0], fat_eta[0], fat_phi[0],
                fat_pt[1], fat_eta[1], fat_phi[1],
                m_jj,
                tau21_j1, tau21_j2,
                tau32_j1, tau32_j2,
                met, phi_met, min_dPhi, ht
            ]
            fout.write(" ".join(map(str, row)) + "\n")

# ------------------ MAIN LOOP ------------------
if selected_period not in all_periods:
    print(f"Error: Period '{selected_period}' not found. Available: {list(all_periods.keys())}")
    sys.exit(1)

print(f"\n=== Processing Period {selected_period} ===")
run_count = 0
reduce_count = 0
dataset_count = 0

for run, links in all_periods[selected_period].items():
    print(f"\n--- Processing run: {run}")
    run_count += 1
    c = 0
    outdir = f"Dataset_ver4/Data/predataset/{selected_period}/run{run}"
    for link in links:
        reduce_file = f"{path_reduce_root}/{selected_period}/run{run}/root_{run}_{c}.root"
        if not os.path.exists(reduce_file):
            reduce_root(selected_period, run, link, c)
            reduce_count += 1
        # make_dataset(selected_period, run, c, reduce_file, outdir)
        dataset_count += 1
        c += 1

print(f"Summary for Period {selected_period}:")
print(f"  Runs processed:   {run_count}")
print(f"  Reduced ROOTs:    {reduce_count}")
# print(f"  Dataset files:    {dataset_count}")
print("=================================")
