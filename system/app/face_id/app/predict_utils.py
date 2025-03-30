import numpy as np

def xywh2xyxy(x):
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y

def auto_conf_threshold(pred, target_box_count=3, min_conf=0.1, max_conf=0.9, step=0.01):
    thresholds = np.arange(max_conf, min_conf - step, -step)
    for threshold in thresholds:
        count = np.sum(pred[:, 4] >= threshold)
        if count >= target_box_count:
            return float(threshold)
    return min_conf

def estimate_target_box_count(pred, min_conf=0.1):
    return int(np.sum(pred[:, 4] >= min_conf))

def is_new_face(embedding, db_embeddings, threshold=0.5):
    for known_embed in db_embeddings:
        sim = np.dot(embedding, known_embed) / (np.linalg.norm(embedding) * np.linalg.norm(known_embed))
        if sim > (1 - threshold):
            return False
    return True
