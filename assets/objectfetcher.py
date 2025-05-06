import sapien
import os

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im5iMzIyN0Bjb2x1bWJpYS5lZHUiLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE3NDYxNTM4NDcsImV4cCI6MTc3NzY4OTg0N30.goA4mqWI6oWmw3TxBtinClOvnOYiY6LToPe_SkmEOUQ"
model_id = 44817

# Download returns a SINGLE directory path
download_dir = sapien.asset.download_partnet_mobility(
    model_id, token=TOKEN, directory="./objects/"
)

# Rename folder
new_folder_name = f"./objects/cabinet_{model_id}"
os.rename(download_dir, new_folder_name)

# Manually construct URDF and mesh paths
urdf_path = os.path.join(new_folder_name, "mobility.urdf")  # Actual URDF filename
mesh_dir = os.path.join(new_folder_name, "meshes")

print("URDF saved at:", urdf_path)
print("Meshes saved at:", mesh_dir)
