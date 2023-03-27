from scipy.io import loadmat
import pandas as pd

mat_path = "./data/dataOut4.mat"

mat_data = loadmat(mat_path)
df = pd.DataFrame(mat_data["temp"][1:])
label = pd.DataFrame(mat_data["label"])

print(df.shape, label.shape)
print(df)
name = mat_path.split(".mat")[0] + ".xlsx"
print(name)
df.to_excel(f"{name}", index=False)

# import pdb; pdb.set_trace()