import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

TRIAL_SEPARATOR = ['2', '0', '0']
END_OBJECT = (-99, -99, -99)
DIST_THRESHOLD_PX = 0.1
TRIAL_TYPE_BY_INDEX = {
     0: "P",  1: "P", 2: "P", # practice trial
     3: "B", 6: "B",  9: "B", 12: "B", # baseline trial
     4: "NT", 7: "NT", 10: "NT", 13: "NT", # negative transfer trial 
     5: "SimRI", 11: "SimRI", # Similar retrocative interference
     8: "DivRI", 14: "DivRI", # Diverse retroactive interference
    15: "DB", 17: "DB", # Delay baseline
    16: "D",  18: "D", # Delay
    19: "C" # Copy trial
}
SCALE_FACTOR = 190


""" -------- File reading and processing functions --------"""
# Separates elements of a text line 
def parse_csv_line(line):
    parts = [p.strip() for p in line.split(",")]
    return parts

# Iterate through raw data file and extract information
def parse_raw_file(path):

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [line.strip() for line in f if line.strip()]
    trials, current_trial, current_trial_header = {}, [], None
    
    i = 0
    while i < len(raw): # Iterate through file lines
        parts = parse_csv_line(raw[i])

        if len(parts) == 8:  # Trial header
            if current_trial: 
                index = current_trial_header["h0"]
                trials[index] = {"numbr_obj": current_trial_header["h1"],
                                 "encoding_time": current_trial_header["h7"],
                                 "objects":current_trial}
                current_trial = []
                
            current_trial_header = {"h0": parts[0],"h1": parts[1],"h7": parts[7]}
            i+=1
            continue

        if len(parts) == 4:
            if parts[1:] == TRIAL_SEPARATOR: # Trial separator line: 0,2,0,0
                i+=1
                continue
                
            else:  # Object header: index, correct_x, correct_y, "Name"
                try:
                    idx = int(float(parts[0]))
                    cx, cy = float(parts[1]), float(parts[2])
                    obj_name = ",".join(parts[3:]).strip().strip('"')
    
                    obj = {"index": idx,"obj_name": obj_name,
                           "correct_x": cx,"correct_y": cy,"placements": []}
                    i += 1
                    while i < len(raw):  # read placement lines until -99,-99,-99
                        p = parse_csv_line(raw[i])
                        if len(p) == 3:
                            px, py, t = float(p[0]), float(p[1]),float(p[2])
                            if (int(px), int(py), int(t)) == END_OBJECT:
                                break
                            obj["placements"].append({"placed_x": px,"placed_y": py,"time_ms": t})
                        i += 1
                        
                    obj["final_placement"] = obj["placements"][-1] if obj["placements"] else None
                    current_trial.append(obj)
                    
                except:
                    pass # not a valid object header, ignore

        i+=1

    if current_trial:
        index = current_trial_header["h0"]
        trials[index] = {"numbr_obj": current_trial_header["h1"],
                         "encoding_time": current_trial_header["h7"], "objects":current_trial}            
    return trials


# Process participant file into a dataframe of results
def analyze_participant_file(path, run_label, threshold_px=DIST_THRESHOLD_PX):
    trials = parse_raw_file(path)
    object_rows,trial_rows = [],[]

    for trial_idx, trial in trials.items(): # Iterate through each trial and extract information 
        trial_objects = trial["objects"]

        if run_label == "b":
            trial_idx = str(int(trial_idx) + 2)

        obj_df = process_object_results(trial_objects,threshold_px)
        obj_df["trial"] = trial_idx
        obj_df["trial type"] = TRIAL_TYPE_BY_INDEX.get(int(trial_idx), "unknown")
        trial_df = process_trial_results(trial_objects, threshold_px)
        trial_df.insert(0, "trial index", trial_idx)
        trial_df.insert(1, "trial type", TRIAL_TYPE_BY_INDEX.get(int(trial_idx), "unknown"))
        trial_df.insert(2, "number objects", trial["numbr_obj"])
        trial_df.insert(3, "encoding time (ms)", trial["encoding_time"])

        object_rows.append(obj_df)
        trial_rows.append(trial_df)

    obj_level = pd.concat(object_rows, ignore_index=True) if object_rows else pd.DataFrame()
    trial_level = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
    return obj_level, trial_level

# Process a folder of raw data and save the processed information 
def process_folder_raw_data(path, output_path, threshold_px=DIST_THRESHOLD_PX):
    files = sorted(glob.glob(os.path.join(path, "*")))
    patterns = ["c", "controle", "p", "patient"]

    for p in patterns:
        participant_files = dict()
        pattern = re.compile(rf"({p}\d+)([ab])", re.IGNORECASE)
        directory = Path(output_path+f"/{p}")
        directory.mkdir(parents=True, exist_ok=True)  
        
        for f in files:
            name = os.path.basename(f).strip()
            m = pattern.search(name)
            if not m:
                continue
        
            pid, run = m.groups() # extract patient ID and run from file name 
            pid, run = pid.lower(), run.lower()
            participant_files.setdefault(pid, {})[run] = f

        for pid, runs in participant_files.items():
            obj_dfs, trial_dfs = [], []
            for run_label in ("a", "b"):
                file_path = runs[run_label]
                obj_df, trial_df = analyze_participant_file(file_path, run_label, threshold_px)
                if not obj_df.empty:   
                    obj_df["run"], trial_df["run"] = run_label,run_label
                    obj_dfs.append(obj_df)
                    trial_dfs.append(trial_df)
            
            if not obj_dfs:
                print(f"No valid data for {pid}")
                continue
                
            pd.concat(obj_dfs).to_csv(directory / f"{pid}_obj.csv", index=False)
            pd.concat(trial_dfs).to_csv(directory / f"{pid}_trials.csv", index=False)


""" -------- Process task results --------"""
# Process task results at object-level
def process_object_results(trial_objects, threshold_px=DIST_THRESHOLD_PX):
    rows = []
    for obj in trial_objects:
        placements = obj.get("placements", [])
        
        coords_all = [(p["placed_x"], p["placed_y"]) for p in placements]
        times_all = [p["time_ms"] for p in placements]
        px = obj["final_placement"]["placed_x"]
        py = obj["final_placement"]["placed_y"]
        time_ms = obj["final_placement"]["time_ms"]
        dist =  distance.euclidean((px, py), (obj["correct_x"], obj["correct_y"])) * SCALE_FACTOR
        correct = int(dist <= threshold_px)

        rows.append({
            "object index": obj["index"], "object name": obj["obj_name"],
            "correct x": obj["correct_x"], "correct y": obj["correct_y"],
            "final placement x": px, "final placement y": py,
            "final time (ms)": time_ms, "absolute distance": dist, "correct result": correct,
            "number placements": len(placements), "all placements": coords_all, "all times": times_all})

    return pd.DataFrame(rows)
    
# Process task results at trial-level
def process_trial_results(trial_objects, threshold_px=DIST_THRESHOLD_PX):
    targets = [(o["correct_x"], o["correct_y"]) for o in trial_objects]
    placed = []
    placed_indices = []
    abs_errors = []
    final_times = []
    
    # Get absolute distance scores
    for i, o in enumerate(trial_objects):
        px, py = o["final_placement"]["placed_x"], o["final_placement"]["placed_y"]
        abs_errors.append(distance.euclidean((px, py), (o["correct_x"], o["correct_y"]))* SCALE_FACTOR) 
        final_times.append(o["final_placement"]["time_ms"])
        placed.append((px,py))
        placed_indices.append(i)

    # Get best fit scores
    cost = np.zeros((len(targets), len(placed)), dtype=float)
    for i, (tx, ty) in enumerate(targets):
        for j, (px, py) in enumerate(placed):
            cost[i, j] = distance.euclidean((tx, ty), (px, py))*SCALE_FACTOR
    row_ind, col_ind = linear_sum_assignment(cost) # get minimum distance for each object
    matched_distances = cost[row_ind, col_ind]

    # Get permutations
    perm = list(range(len(trial_objects)))
    for r, c in zip(row_ind, col_ind):¡
        perm[r] = placed_indices[c]
        
    # Get swaps and subs 
    visited, swaps, subs = set(), 0, 0
    for i, p in enumerate(perm):
        if i == p:
            continue
        if p < len(trial_objects) and perm[p] == i and i not in visited:
            swaps += 1
            visited.update({i, p})
        elif i not in visited:
            subs += 1
            visited.add(i)

    return  pd.DataFrame([{"absolute error score": float(np.sum(abs_errors)),
            "bestfit score": float(np.sum(matched_distances)), 
            "final time (ms)": max(final_times),
            "permutations": "".join(map(str, perm)),
            "swaps": swaps, "substitutions": subs, "repetitions": np.sum([len(o["placements"]) for o in trial_objects])}])

