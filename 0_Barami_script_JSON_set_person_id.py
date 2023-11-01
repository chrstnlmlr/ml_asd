import numpy as np
import json
from json.decoder import JSONDecodeError
from os import listdir
from os.path import join, isfile

def read_json(file):
    try:
        with open(file, 'r') as j:
            return json.loads(j.read())
    except (OSError, UnicodeDecodeError, JSONDecodeError) as e:
        print(f'Error while reading json file: {file}')
        print(e)
        return


def write_json(j, dst):
    with open(dst, 'w') as f:
        json.dump(j, f)


def distance(p1, p2):
    x1, y1 = center_of_mass(p1)
    x2, y2 = center_of_mass(p2)

    if not any([x for x in [x1, x2, y1, y2] if x is np.NaN]):
        return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
    else:
        return np.inf


def center_of_mass(p):
    k = np.array([float(c) for c in p['pose_keypoints_2d']])
    coords = [c for c in zip(k[::3], k[1::3], k[2::3])]
    threshold = 0.1

    x = np.array([c[0] for c in coords if c[2] > threshold]).mean()
    y = np.array([c[1] for c in coords if c[2] > threshold]).mean()

    return x, y


def find_closest(p, prevs):
    ids = set([pi['person_id'] for pi in prevs])
    id_weights = dict([(i, []) for i in ids])
    for pi in prevs:
        d = distance(p, pi)
        id_weights[pi['person_id']].append(d)

    for i in ids:
        id_weights[i] = np.mean(id_weights[i])

    if len(id_weights) > 0:
        selected_id = min(id_weights, key=id_weights.get)
        selected_weight = id_weights[selected_id]
    else:
        selected_id = 0
        selected_weight = 0
    return selected_id, selected_weight


def match_frames(ps, prevs):
    def gen_id(ps):
        ids = [p['person_id'] for p in ps]
        i = 0
        while i in ids:
            i += 1
        return i

    found = {}
    for p in ps:
        new_id, d_new = find_closest(p, prevs)
        if new_id in found:
            p_old, d_old = found[new_id]
            generated_id = gen_id(prevs + [v[0] for v in found.values()])
            if d_old < d_new:
                new_id = generated_id
            else:
                p_old['person_id'] = generated_id
                found[generated_id] = (p_old, d_old)
                p['person_id'] = new_id
        p['person_id'] = new_id
        found[new_id] = (p, d_new)


def set_person_id(json_src, n=5, print_interval=10000):
    file_names = [f for f in listdir(json_src) if isfile(join(json_src, f)) and f.endswith('json')]

    def src_path(i):
        return join(json_src, file_names[i])

    def dst_path(i):
        return join(json_src, file_names[i])

    jsons = [read_json(src_path(0))]
    for idx, p in enumerate(jsons[0]['people']):
        p['person_id'] = idx
    write_json(jsons[0], dst_path(0))

    for i in range(1, len(file_names)):
        jsons.append(read_json(src_path(i)))
        people = jsons[i]['people']

        prevs = []
        for j in range(min(n, i)):
            for p in jsons[j]['people']:
                prevs.append(p)

        match_frames(people, prevs)

        write_json(jsons[i], dst_path(i))

        if i % print_interval == 0:
            print(f'frame: {i}')

json_src = 'path_to_JSON_files'

set_person_id(json_src,3,1000)
