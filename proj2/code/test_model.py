import torch
from net_class import Net
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# 50x50 pixels
img_size = 50

net = Net()
net.load_state_dict(torch.load('melanoma_model.pth'))
net.eval()

testing_data = torch.load("testing_data.pt")

# putting all the image arrays into this tensor
test_X = torch.stack( [item[0] for item in testing_data]  )

# this tensor has the images labels
test_y = torch.tensor( [item[1] for item in testing_data], dtype=torch.long  )

correct = 0
total = 0

with torch.no_grad():
    outputs = net(test_X)
    _, predicted = torch.max(outputs, 1)

accuracy = accuracy_score(test_y, predicted)
roc_auc = roc_auc_score(test_y, predicted)
precision = precision_score(test_y, predicted)
recall = recall_score(test_y, predicted)
f1 = f1_score(test_y, predicted)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
