# find_mesh_center.py
import mujoco
import numpy as np

xml = """
<mujoco>
  <asset>
    <mesh name="switch_mesh"
          file="/Users/vijeyavelan/AME547_Keyboard_Assembly/models/meshes/Switch_Decimated.stl"
          scale="0.001 0.001 0.001"
          refpos="0 0 0"
          refquat="0.7071 0.7071 0 0"/>
  </asset>
  <worldbody>
    <body name="test">
      <geom type="mesh" mesh="switch_mesh" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(xml)

# Get mesh vertex data
mesh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "switch_mesh")
vert_start = m.mesh_vertadr[mesh_id]
vert_count = m.mesh_vertnum[mesh_id]
verts = m.mesh_vert[vert_start:vert_start + vert_count]

print(f"Vertex count: {vert_count}")
print(f"X range: {verts[:,0].min()*1000:.3f} to {verts[:,0].max()*1000:.3f} mm")
print(f"Y range: {verts[:,1].min()*1000:.3f} to {verts[:,1].max()*1000:.3f} mm")
print(f"Z range: {verts[:,2].min()*1000:.3f} to {verts[:,2].max()*1000:.3f} mm")

# Geometric center
cx = (verts[:,0].min() + verts[:,0].max()) / 2
cy = (verts[:,1].min() + verts[:,1].max()) / 2
cz = (verts[:,2].min() + verts[:,2].max()) / 2
z_min = verts[:,2].min()

print(f"\nGeometric center (m): ({cx:.6f}, {cy:.6f}, {cz:.6f})")
print(f"Z min (bottom face): {z_min:.6f} m")

# To put bottom face at z=0: refpos should shift so z_min becomes 0
# refpos X,Y = center (to center horizontally)
# refpos Z = z_min (to put bottom at origin)
print(f"\nUse this refpos to center XY and put bottom at z=0:")
print(f'  refpos="{cx:.4f} {cy:.4f} {z_min:.4f}"')