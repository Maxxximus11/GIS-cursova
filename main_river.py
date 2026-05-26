import sys
import tkinter
from tkinter import simpledialog, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.animation as animation

from dynamic_classes import Node, Element, Mesh
from dem_loader import DEMLoader

root = tkinter.Tk()
root.withdraw()

# 1. Рельєф (DEM)
filepath = filedialog.askopenfilename(
    title="Оберіть файл цифрової моделі рельєфу (DEM)",
    filetypes=[("ASCII Grid files", "*.asc"), ("All files", "*.*")]
)

if not filepath:
    print("Вибір файлу скасовано. Програма завершує роботу.")
    sys.exit(0)

loader = DEMLoader(filepath)
dem, cell_size = loader.load()

rows, cols = dem.shape
print(f"Матриця рельєфу: {rows}x{cols}")

# 2. Вузли розрахункової сітки
nodes = []
for r in range(rows):
    for c in range(cols):
        node_id = r * cols + c
        nodes.append(Node(node_id, x=c, y=r, z=dem[r, c]))

x_coords = np.array([n.x for n in nodes])
y_coords = np.array([n.y for n in nodes])
tri = Triangulation(x_coords, y_coords)

# 3. Скінченні елементи (Трикутники)
elements = []
elem_id = 0
for r in range(rows - 1):
    for c in range(cols - 1):
        n1 = r * cols + c
        n2 = r * cols + (c + 1)
        n3 = (r + 1) * cols + c
        n4 = (r + 1) * cols + (c + 1)

        elements.append(Element(elem_id, [nodes[n1], nodes[n2], nodes[n3]], 0.005))
        elem_id += 1
        elements.append(Element(elem_id, [nodes[n2], nodes[n4], nodes[n3]], 0.005))
        elem_id += 1

mesh = Mesh(nodes, elements)

# 4. Граничні умови (Визначення меж розрахункової області)
border_top_nodes = [0 * cols + c for c in range(cols)]
border_bottom_nodes = [(rows - 1) * cols + c for c in range(cols)]
border_left_nodes = [r * cols + 0 for r in range(rows)]
border_right_nodes = [r * cols + (cols - 1) for r in range(rows)]

fig_select, ax_select = plt.subplots(figsize=(10, 6))
z_vals = [n.z for n in nodes]
contour_select = ax_select.tricontourf(tri, z_vals, levels=50, cmap='viridis')
fig_select.colorbar(contour_select, ax=ax_select, label='Висота рельєфу (м)')
ax_select.set_title("Клікніть біля будь-якої межі для вибору центру затоплення")

pts = plt.ginput(1, timeout=0)

if not pts:
    print("Вибір точки скасовано користувачем. Програма завершує роботу.")
    sys.exit(0)

click_x, click_y = pts[0]
plt.close(fig_select)

dist_left = click_x
dist_right = (cols - 1) - click_x
dist_top = click_y
dist_bottom = (rows - 1) - click_y

min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
source_width = 10

custom_source_nodes = []
outflow_nodes = []
default_node_id = 0
edge_name = ""

if min_dist == dist_left:
    edge_name = "Західна (Ліва)"
    custom_source_nodes = [n for n in border_left_nodes if abs(nodes[n].y - click_y) <= source_width / 2]
    outflow_nodes = border_right_nodes + border_top_nodes + border_bottom_nodes
    default_node_id = int(round(click_y)) * cols
elif min_dist == dist_right:
    edge_name = "Східна (Права)"
    custom_source_nodes = [n for n in border_right_nodes if abs(nodes[n].y - click_y) <= source_width / 2]
    outflow_nodes = border_left_nodes + border_top_nodes + border_bottom_nodes
    default_node_id = int(round(click_y)) * cols + (cols - 1)
elif min_dist == dist_top:
    edge_name = "Південна (Нижня)"
    custom_source_nodes = [n for n in border_top_nodes if abs(nodes[n].x - click_x) <= source_width / 2]
    outflow_nodes = border_bottom_nodes + border_left_nodes + border_right_nodes
    default_node_id = int(round(click_x))
else:
    edge_name = "Північна (Верхня)"
    custom_source_nodes = [n for n in border_bottom_nodes if abs(nodes[n].x - click_x) <= source_width / 2]
    outflow_nodes = border_top_nodes + border_left_nodes + border_right_nodes
    default_node_id = (rows - 1) * cols + int(round(click_x))

outflow_nodes = list(set(outflow_nodes) - set(custom_source_nodes))

default_node_id = max(0, min(default_node_id, len(nodes) - 1))
default_z = nodes[default_node_id].z

# Графічне вікно параметризації
user_lake_level = simpledialog.askfloat(
    "Рівень води",
    f"ВВедіть абсолютну висоту води(Рельєф тут: ~{default_z:.1f} м)",
    initialvalue=default_z + 1.0,
    minvalue=0.0,
    parent=root
)

if user_lake_level is None:
    print("Введення параметрів скасовано. Програма завершує роботу.")
    sys.exit(0)

# 5. Динамічна симуляція
def calculate_hydraulic_head(source_nodes, absolute_level):
    fixed_depths = {}
    active_nodes = []
    for n_id in source_nodes:
        h = absolute_level - nodes[n_id].z
        if h > 0:
            fixed_depths[n_id] = h
            active_nodes.append(n_id)
    return fixed_depths, active_nodes

source_fixed_H, active_source_nodes = calculate_hydraulic_head(custom_source_nodes, user_lake_level)

block_wall_nodes = []
for r in range(35, rows):
    for c in range(0, 15):
        block_wall_nodes.append(r * cols + c)

H_old = np.zeros(rows * cols)
for node_id in active_source_nodes:
    H_old[node_id] = source_fixed_H[node_id]

history_H = [H_old.copy()]
video_frames, sub_steps, dt = 90, 5, 1.0


for frame in range(video_frames):
    H_sub_old = H_old.copy()

    for _ in range(sub_steps):
        K, C, F_base = mesh.assemble_global_system(H_sub_old)
        K_eff_base = C + dt * K
        K_eff_base = K_eff_base.tolil()

        dummy_F = np.zeros(len(nodes))

        # Динамічне обнулення коефіцієнтів у глобальній матриці
        K_eff_base, _ = mesh.apply_dirichlet_condition(
            K_eff_base, dummy_F, outflow_nodes + block_wall_nodes + active_source_nodes, 0.0
        )

        K_eff_base.setdiag(K_eff_base.diagonal() + 1e-9)
        K_eff_csr = K_eff_base.tocsr()

        F_eff = C @ H_sub_old + dt * F_base

        # Застосування граничних умов Діріхле
        _, F_eff = mesh.apply_dirichlet_condition(None, F_eff, outflow_nodes + block_wall_nodes, 0.0)

        # Моделювання активного припливу
        source_values = [source_fixed_H[n_id] for n_id in active_source_nodes]
        _, F_eff = mesh.apply_dirichlet_condition(None, F_eff, active_source_nodes, source_values)

        H_new = mesh.solve_system(K_eff_csr, F_eff)
        H_new = np.maximum(H_new, 0.0)

        # Жорстка фіксація
        for node_id in outflow_nodes:
            H_new[node_id] = 0.0
        for node_id in block_wall_nodes:
            H_new[node_id] = 0.0
        for node_id in active_source_nodes:
            H_new[node_id] = source_fixed_H[node_id]

        H_sub_old = H_new

    H_old = H_new
    history_H.append(H_new.copy())

    if (frame + 1) % 10 == 0:
        print(f"Розраховано кадр: {frame + 1}/{video_frames}")

# 6. Модуль динамічної візуалізації
z_vals = np.array([n.z for n in nodes])

triangles = []
for elem in elements:
    triangles.append([n.id for n in elem.nodes])
triangles = np.array(triangles)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

H_total_history = [h + z_vals for h in history_H]
min_val = np.min(H_total_history)
max_val = np.max(H_total_history)
step_quiver = 10

def update(frame_number):
    ax1.clear()
    ax2.clear()

    real_time = frame_number * (sub_steps * dt)
    H_current = history_H[frame_number]
    H_total_current = H_total_history[frame_number]

    vx, vy, _ = mesh.calculate_node_velocities(H_total_current)
    water_mask = H_current > 0.01
    vx = vx * water_mask
    vy = vy * water_mask
    v_mag_safe = np.where((vx ** 2 + vy ** 2) == 0, 1e-10, np.sqrt(vx ** 2 + vy ** 2))

    ax1.set_title(f"ГЛИБИНА ВОДИ: {real_time:.1f} сек")
    ax1.set_xlabel("X (клітинки)")
    ax1.set_ylabel("Y (клітинки)")
    ax1.axis('equal')
    contour1 = ax1.tricontourf(tri, H_current, levels=20, cmap='Blues', vmin=0.0, vmax=2.5)
    ax1.quiver(x_coords[::step_quiver], y_coords[::step_quiver],
               (vx / v_mag_safe)[::step_quiver], (vy / v_mag_safe)[::step_quiver],
               color='black', scale=30, width=0.003)

    ax2.set_title(f"РЕЛЬЄФ ТА РІВЕНЬ ЗАТОПЛЕННЯ: {real_time:.1f} сек")
    ax2.set_xlabel("X (клітинки)")
    ax2.set_ylabel("Y (клітинки)")
    ax2.axis('equal')
    contour2 = ax2.tricontourf(tri, H_total_current, levels=20, cmap='viridis', vmin=min_val, vmax=max_val)
    ax2.quiver(x_coords[::step_quiver], y_coords[::step_quiver],
               (vx / v_mag_safe)[::step_quiver], (vy / v_mag_safe)[::step_quiver],
               color='red', scale=30, width=0.003)

    return contour1, contour2

cbar1 = fig.colorbar(ax1.tricontourf(tri, history_H[0], levels=20, cmap='Blues', vmin=0.0, vmax=2.5), ax=ax1)
cbar1.set_label("Глибина водної поверхні (метри)")

cbar2 = fig.colorbar(ax2.tricontourf(tri, H_total_history[0], levels=20, cmap='viridis', vmin=min_val, vmax=max_val), ax=ax2)
cbar2.set_label("Абсолютний рівень (метри)")

plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=len(history_H), interval=50, repeat=True)
plt.show()