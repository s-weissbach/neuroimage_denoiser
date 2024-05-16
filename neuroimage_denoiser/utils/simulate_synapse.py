import numpy as np


def generate_synapse(
    height: int, width: int, synapse_diameter: int, peak: np.ndarray
) -> np.ndarray:
    canvas = np.zeros((len(peak), height, width), dtype=np.float64)
    # circular synapse
    random_pos_y = np.random.choice(height)
    random_pos_x = np.random.choice(width)
    # initialize normal distribution
    norm_dist = norm.pdf(np.arange(0, synapse_diameter + 1, 0.01), 0, 8)
    norm_dist = [n / np.max(norm_dist) for n in norm_dist]
    print(len(norm_dist))
    for y in range(height):
        for x in range(width):
            distance = np.sqrt(pow(y - random_pos_y, 2) + pow(x - random_pos_x, 2))
            if distance > synapse_diameter:
                continue
            for frame in range(len(peak)):
                canvas[frame, y, x] = peak[frame] * norm_dist[int(distance * 100)]
    return canvas


x = generate_synapse(20, 20, 7, np.array(peak_db.get("1")["0"]))
