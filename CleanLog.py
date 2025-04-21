import os
import json
import pandas as pd
import random
from collections import defaultdict
from queue import PriorityQueue
from itertools import count
from sklearn.metrics import mean_squared_error

# === Config ===
base_dir = "C://VLNLP//Test"
num_samples = 20000

shape_options = ['cube', 'sphere', 'cylinder', 'cone', 'torus']
material_options = ['opaque', 'transparent', 'transparent_blue', 'mirror', 'gold']
cost_map = {"edit": 1.0, "add": 2, "remove": 2.5}

# === Load original objects ===
df_original = pd.read_csv(os.path.join(base_dir, "scene_objects.csv"))
df_original = df_original[df_original["scene_id"].isin(df_original["scene_id"].unique()[:num_samples])]
scenes = df_original.groupby("scene_id")

# === Violation checker ===
def check_violations(scene):
    violations = []
    counter = defaultdict(int)
    for obj in scene:
        key = (obj["shape"], obj["material"])
        counter[key] += 1
    freq = defaultdict(int)
    for count in counter.values():
        if count != 1:
            freq[count] += 1
        if freq[count] > 1:
            violations.append("rule1")
            break
    shape_count = defaultdict(int)
    for obj in scene:
        shape_count[obj["shape"]] += 1
    if any(count >= 4 for count in shape_count.values()):
        violations.append("rule2")
    sphere_count = sum(1 for o in scene if o["shape"] == "sphere")
    torus_count = sum(1 for o in scene if o["shape"] == "torus")
    cube_count = sum(1 for o in scene if o["shape"] == "cube")
    if (sphere_count + torus_count) > cube_count:
        violations.append("rule3")
    return violations

# === A* Solver ===
def apply_action(scene, action):
    new_scene = [obj.copy() for obj in scene]
    if action["type"] == "edit":
        if "new_size" in action:
            new_scene[action["target"]]["size"] = action["new_size"]
        if "new_shape" in action:
            new_scene[action["target"]]["shape"] = action["new_shape"]
        if "new_material" in action:
            new_scene[action["target"]]["material"] = action["new_material"]
    elif action["type"] == "remove":
        del new_scene[action["target"]]
    elif action["type"] == "add":
        new_scene.append(action["object"])
    return new_scene

def get_possible_actions(scene):
    actions = []
    for i, obj in enumerate(scene):
        actions.append({"type": "edit", "target": i, "new_size": obj["size"][:2] + [0.6]})
        actions.append({"type": "edit", "target": i, "new_shape": random.choice(shape_options)})
        actions.append({"type": "edit", "target": i, "new_material": random.choice(material_options)})
        actions.append({"type": "remove", "target": i})
    actions.append({"type": "add", "object": {
        "shape": "cube",
        "material": "gold",
        "size": [1.0, 1.0, 1.0],
        "position": [0, 0],
        "rotation_z": 0
    }})
    return actions

def a_star_solver(initial_scene, max_steps=1000):
    pq = PriorityQueue()
    visited = set()
    unique_counter = count()
    pq.put((len(check_violations(initial_scene)), next(unique_counter), 0, initial_scene, []))
    best_solutions = []
    seen_action_sets = set()

    def actions_to_key(actions):
        return json.dumps(actions, sort_keys=True)

    def scene_to_key(scene):
        return json.dumps(scene, sort_keys=True)

    while not pq.empty() and len(best_solutions) < 1:
        _, _, cost_so_far, current_scene, actions = pq.get()
        scene_key = scene_to_key(current_scene)
        if scene_key in visited:
            continue
        visited.add(scene_key)
        if not check_violations(current_scene):
            best_solutions.append({
                "actions": actions,
                "cost": cost_so_far,
                "success": True,
                "cleaned_scene": current_scene
            })
            break
        if len(actions) >= max_steps:
            continue
        for action in get_possible_actions(current_scene):
            new_scene = apply_action(current_scene, action)
            new_cost = cost_so_far + cost_map[action["type"]]
            heuristic = len(check_violations(new_scene))
            total_score = new_cost + heuristic
            pq.put((total_score, next(unique_counter), new_cost, new_scene, actions + [action]))
    return best_solutions or [{
        "actions": [],
        "cost": None,
        "success": False,
        "cleaned_scene": initial_scene
    }]

# === Containers ===
updated_objects = []
violation_log = []
solution_log = []

# === Loop through grouped objects ===
for scene_id, group in scenes:
    scene = []
    for _, row in group.iterrows():
        scene.append({
            "shape": row["shape"],
            "material": row["material"],
            "size": [1.0, 1.0, row["size_z"]],
            "position": [0.0, 0.0],
            "rotation_z": 0
        })

    violations = check_violations(scene)
    label = "faulty" if violations else "normal"
    rule_cost = len(violations)

    for idx, obj in enumerate(scene):
        updated_objects.append({
            "scene_id": scene_id,
            "object_id": idx,
            "label": label,
            "shape": obj["shape"],
            "material": obj["material"],
            "size_z": obj["size"][2],
            "size_category": (
                "small" if obj["size"][2] < 0.8 else
                "big" if obj["size"][2] > 1.2 else
                "mid"
            )
        })

    violation_log.append({
        "scene_id": scene_id,
        "violations": ", ".join(violations),
        "label": label,
        "rule_cost": rule_cost
    })

    if label == "faulty":
        solutions = a_star_solver(scene)
        for idx, sol in enumerate(solutions):
            solution_log.append({
                "scene_id": scene_id,
                "solution_index": idx + 1,
                "success": sol["success"],
                "cost": sol["cost"],
                "actions": json.dumps(sol["actions"], default=str)
            })
    else:
        solution_log.append({
            "scene_id": scene_id,
            "solution_index": None,
            "success": None,
            "cost": 0,
            "actions": "[]"
        })

# === Save updated logs ===
pd.DataFrame(updated_objects).to_csv(os.path.join(base_dir, "scene_objects_updated.csv"), index=False)
pd.DataFrame(violation_log).to_csv(os.path.join(base_dir, "violation_labels_updated.csv"), index=False)
pd.DataFrame(solution_log).to_csv(os.path.join(base_dir, "solution_log_updated.csv"), index=False)

# === MSE Summary ===
df_base = df_original.copy()
df_test = pd.DataFrame(updated_objects)
merged = df_base.merge(df_test, on=["scene_id", "object_id"], suffixes=('_old', '_new'))

print("\n📊 MSE:")
print("Shape MSE:    ", mean_squared_error(
    merged["shape_old"].apply(lambda s: shape_options.index(s)),
    merged["shape_new"].apply(lambda s: shape_options.index(s))
))
print("Material MSE: ", mean_squared_error(
    merged["material_old"].apply(lambda s: material_options.index(s)),
    merged["material_new"].apply(lambda s: material_options.index(s))
))
print("Size Z MSE:   ", mean_squared_error(
    merged["size_z_old"],
    merged["size_z_new"]
))
print("Label MSE:    ", mean_squared_error(
    merged["label_old"].map(lambda x: 1 if x == "faulty" else 0),
    merged["label_new"].map(lambda x: 1 if x == "faulty" else 0)
))
