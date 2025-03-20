import pickle
from pathlib import Path

file_path="only_grouping_two_images-pos_ver.pkl"
file_path = Path(file_path)  # Pathオブジェクトに変換

with file_path.open("rb") as f:
    train_data = pickle.load(f)
print("既存のデータを読み込みました。")


print(len(train_data["observations"]["state"]))


min=list()
max=list()
count=1
for value in train_data["actions"]:
    if count==1:
        min=value.copy()
        max=value.copy()
        count+=1
        continue
    for i in range(0,len(train_data["observations"]["state"][0])):
        if value[i]>max[i]:
            max[i]=value[i]
            # print(f"max:{max}")
        if value[i]<min[i]:
            min[i]=value[i]
            # print(f"min:{min}")


    count+=1

print(f"count:{count} min:{min} max:{max}")