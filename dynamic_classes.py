import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class Node:
    def __init__(self, id, x, y, z, known_h=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

        self.h = 0.0
        self.qx = 0.0
        self.qy = 0.0

        self.known_h = known_h


class Element:
    def __init__(self, id, nodes, manning_n):
        self.id = id
        self.nodes = nodes
        self.manning_n = manning_n
        self.area = 0.0

    # Матриця градієнтів(2.27 - 2.28)
    def get_B_matrix(self):
        x_i, y_i = self.nodes[0].x, self.nodes[0].y
        x_j, y_j = self.nodes[1].x, self.nodes[1].y
        x_k, y_k = self.nodes[2].x, self.nodes[2].y

        b_i = y_j - y_k
        b_j = y_k - y_i
        b_k = y_i - y_j

        c_i = x_k - x_j
        c_j = x_i - x_k
        c_k = x_j - x_i

        double_area = abs(x_i * b_i + x_j * b_j + x_k * b_k)
        self.area = double_area / 2.0

        B_matrix = np.array([
            [b_i, b_j, b_k],
            [c_i, c_j, c_k]
        ]) / double_area

        return B_matrix

    def get_D_matrix(self): # Матриця фізичних властивостей(гідравлічна провідність)
        conductivity = 1.0 / self.manning_n

        D = np.array([
            [conductivity, 0.0],
            [0.0, conductivity]
        ])

        return D

    # Локальна матриця провідності (або матриця жорсткості) скінченного елемента(2.29) # коефіцієнт
    def get_stiffness(self, h_avg=1.0):  # h_avg глибина води в трикутнику
        B = self.get_B_matrix() # під яким кутом лежить трикутник
        conductivity = 1.0 / self.manning_n
        D = np.array([
            [conductivity, 0.0],
            [0.0, conductivity]
        ])

        k_matrix = (B.T @ D @ B) * self.area * h_avg
        return k_matrix

    def get_mass_matrix(self):
        # Бере площу трикутника і ділить її порівну на 3 вузли.
        c_val = self.area / 3.0

        C = np.array([
            [c_val, 0.0, 0.0],
            [0.0, c_val, 0.0],
            [0.0, 0.0, c_val]
        ])

        return C



    def get_velocity(self, H_nodes):  # фізичну швидкість і напрямок течії

        B = self.get_B_matrix()

        D = self.get_D_matrix()

        velocity = -np.dot(D, np.dot(B, H_nodes))

        return velocity


class Mesh:

    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements

    def assemble_global_system(self, H_current=None):
        num_nodes = len(self.nodes)

        # Глобальна матриця провідності [K].
        K_global = lil_matrix((num_nodes, num_nodes))

        # 3. Глобальна матриця мас/ємності [C]
        C_global = lil_matrix((num_nodes, num_nodes))

        # 4. Глобальний вектор навантажень {F} (гравітація).
        F_global = np.zeros(num_nodes)

        for elem in self.elements:

            # Оновлюємо матрицю градієнтів [B]
            _ = elem.get_B_matrix()

            # Отримуємо локальну матрицю мас [c] (3x3)
            c_local = elem.get_mass_matrix()

            if H_current is not None:
                h_avg = sum([H_current[n.id] for n in elem.nodes]) / 3.0
            else:
                h_avg = 0.0

            if h_avg > 1e-6:
                k_local = elem.get_stiffness(h_avg)
                z_local = np.array([n.z for n in elem.nodes])
                f_gravity = -k_local @ z_local
            else:
                k_local = np.zeros((3, 3))
                f_gravity = np.zeros(3)

            g_idx = [n.id for n in elem.nodes]

            for i in range(3):
                F_global[g_idx[i]] += f_gravity[i]

                for j in range(3):
                    K_global[g_idx[i], g_idx[j]] += k_local[i, j]
                    C_global[g_idx[i], g_idx[j]] += c_local[i, j]


        return K_global.tocsr(), C_global.tocsr(), F_global


    def apply_dirichlet_condition(self, K, F, node_indices, values=None):
        if K is not None:
            for node_id in node_indices:
                K[node_id, :] = 0
                # K[:, node_id] = 0
                K[node_id, node_id] = 1.0

        if F is not None and values is not None:
            if np.isscalar(values):
                values = np.full(len(node_indices), values)
            for i, node_id in enumerate(node_indices):
                F[node_id] = values[i]

        return K, F

    def calculate_node_velocities(self, H_results):
        num_nodes = len(self.nodes)
        sum_vx = np.zeros(num_nodes)
        sum_vy = np.zeros(num_nodes)
        count = np.zeros(num_nodes)

        # 2.46: Обчислення локальної швидкості в трикутнику
        for elem in self.elements:

            h_elem = [H_results[node.id] for node in elem.nodes]

            vx, vy = elem.get_velocity(h_elem)

            # 2.47, 2.48: Накопичення даних для усереднення
            for node in elem.nodes:
                sum_vx[node.id] += vx
                sum_vy[node.id] += vy
                count[node.id] += 1


        node_vx = sum_vx / count
        node_vy = sum_vy / count

        node_v_total = np.sqrt(node_vx ** 2 + node_vy ** 2)

        return node_vx, node_vy, node_v_total

    def solve_system(self, K, F):
        return spsolve(K, F)