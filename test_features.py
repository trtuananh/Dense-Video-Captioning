import numpy as np

# Đường dẫn đến file .npy
file_path = '/Users/trantuananh/Documents/HMCUT/Thesis/DVC/PDVC/v_3meb_5kcPFg.npy'

# Đọc file .npy
loaded_array = np.load(file_path)

# In mảng đã đọc
print(loaded_array.shape)