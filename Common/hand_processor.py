import numpy as np


class HandProcessor:
    def calculate_distances(self, hand_data):
        pairs_to_calculate = [['a1', 'a5'], ['a1', 'a9'], ['a1', 'a13'], ['a1', 'a17'], ['a1', 'a21'],
                              ['a5', 'a9'], ['a5', 'a13'], ['a5', 'a17'], ['a5', 'a21'],
                              ['a9', 'a13'], ['a9', 'a17'], ['a9', 'a21'],
                              ['a13', 'a17'], ['a13', 'a21'],
                              ['a17', 'a21']]
        distances = {}
        for pair in pairs_to_calculate:
            point1, point2 = pair
            x1, y1, z1 = hand_data[point1]
            x2, y2, z2 = hand_data[point2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            distances[f"{point1}:{point2}"] = distance
        return distances

    def sum_distances_between_points(self, hand_data):
        pairs_to_calculate = [
            ['a1', 'a2'], ['a2', 'a3'], ['a3', 'a4'], ['a4', 'a5'],
            ['a1', 'a6'], ['a6', 'a7'], ['a7', 'a8'], ['a8', 'a9'],
            ['a6', 'a10'], ['a10', 'a11'], ['a11', 'a12'], ['a12', 'a13'],
            ['a10', 'a14'], ['a14', 'a15'], ['a15', 'a16'], ['a16', 'a17'],
            ['a14', 'a18'], ['a18', 'a19'], ['a19', 'a20'], ['a20', 'a21'],
            ['a1', 'a18']
        ]
        total_distance = 0
        for pair in pairs_to_calculate:
            point1, point2 = pair
            x1, y1, z1 = hand_data[point1]
            x2, y2, z2 = hand_data[point2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            total_distance += distance
        return total_distance

    def get_point_data(self, hand_positions, label):
        output = []
        for idx, hand_data in enumerate(hand_positions):
            distances = self.calculate_distances(hand_data)
            sum_of_hand_distances = self.sum_distances_between_points(hand_data)
            distances_normalized = {k: v / sum_of_hand_distances for k, v in distances.items()}

            all_x = [v[0] for v in hand_data.values()]
            all_y = [v[1] for v in hand_data.values()]
            all_z = [v[2] for v in hand_data.values()]

            min_x, max_x = np.min(all_x), np.max(all_x)
            min_y, max_y = np.min(all_y), np.max(all_y)
            min_z, max_z = np.min(all_z), np.max(all_z)

            max_diff_x = (max_x - min_x) / sum_of_hand_distances
            max_diff_y = (max_y - min_y) / sum_of_hand_distances
            max_diff_z = (max_z - min_z) / sum_of_hand_distances

            distances_normalized['max_diff_x'] = max_diff_x
            distances_normalized['max_diff_y'] = max_diff_y
            distances_normalized['max_diff_z'] = max_diff_z

            distances_normalized['label'] = label

            output.append(distances_normalized)

        return output