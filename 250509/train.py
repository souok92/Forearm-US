import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, VGG16_Weights, VGG19_Weights

# 경로 설정
DATA_ROOT = "./cat_gest"
CLASSES = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
NUM_CLASSES = len(CLASSES)

# 데이터셋 클래스
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
        
        # 데이터 로딩
        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 그레이스케일로 이미지 로드
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 데이터 변환 설정 (그레이스케일)
train_transform  = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG 모델은 224x224 입력을 기대함
    transforms.RandomRotation(10),  # 랜덤 회전 (± 10도)
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 랜덤 이동 (최대 10%)
    transforms.RandomAffine(0, scale=(0.9, 1.1)),  # 랜덤 확대/축소 (90~110%)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 그레이스케일 정규화
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG 모델은 224x224 입력을 기대함
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 전체 데이터셋 로드
train_full_dataset = HandGestureDataset(DATA_ROOT, transform=train_transform)
val_test_full_dataset = HandGestureDataset(DATA_ROOT, transform=val_test_transform)

# 데이터셋 분할 (훈련:검증:평가 = 7:2:1)
total_size = len(train_full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_indices, val_indices, test_indices = random_split(
    range(total_size), [train_size, val_size, test_size], 
    generator=generator
)

# 데이터 로더
train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices.indices)
val_dataset = torch.utils.data.Subset(val_test_full_dataset, val_indices.indices)
test_dataset = torch.utils.data.Subset(val_test_full_dataset, test_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의 (그레이스케일 입력에 맞게 수정)
class HandGestureModel(nn.Module):
    def __init__(self, model_type='resnet18', num_classes=NUM_CLASSES):
        super(HandGestureModel, self).__init__()
        
        if model_type == 'resnet18':
            # ResNet18 기반 모델, 첫 레이어를 그레이스케일 입력에 맞게 수정
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'mobilenet':
            # MobileNetV2 기반 모델
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            # 그레이스케일 입력에 맞게 첫 번째 레이어 수정
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # 마지막 분류기 레이어 수정
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        elif model_type == 'vgg16':
            # VGG16 기반 모델
            self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
            # 그레이스케일 입력에 맞게 첫 번째 레이어 수정
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            # 마지막 분류기 레이어 수정
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_type == 'vgg19':
            # VGG19 기반 모델
            self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
            # 그레이스케일 입력에 맞게 첫 번째 레이어 수정
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            # 마지막 분류기 레이어 수정
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)

# 모델, 손실 함수, 옵티마이저 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

# 모델 타입 선택 ('resnet18', 'mobilenet', 'vgg16', 'vgg19')
model_type = 'mobilenet'  # 여기서 모델 타입 선택 (vgg16 또는 vgg19로 변경)
model = HandGestureModel(model_type=model_type).to(device)
print(f"Using {model_type} model")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

# 검증 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc

# 모델 평가
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 정확도 및 혼동 행렬
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('Confusion Matrix (Count):')
    print(conf_matrix)
    
    # 클래스별 정확도
    print('\nClass-wise Accuracy:')
    for i, class_name in enumerate(CLASSES):
        class_acc = conf_matrix[i, i] / conf_matrix[i].sum() if conf_matrix[i].sum() > 0 else 0
        print(f'Class {class_name}: {class_acc:.4f}')
    
    return accuracy, conf_matrix

def get_next_version_path(base_path):
    """기존 파일이 있으면 _1, _2 등의 번호를 자동으로 증가시켜 반환합니다."""
    if not os.path.exists(base_path):
        return base_path
    
    # 파일 이름과 확장자 분리
    base_dir = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while True:
        new_path = os.path.join(base_dir, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# 학습 루프
num_epochs = 10
best_val_acc = 0
fixed_model_path = f'best_hand_gesture_{model_type}_model.pth'  # 학습 중 덮어쓸 고정 경로

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 학습 중에는 같은 파일 이름으로 덮어쓰기
        torch.save(model.state_dict(), fixed_model_path)
        print(f'Model saved to {fixed_model_path} with Val Acc: {val_acc:.4f}')

# 최종 모델을 버전 증가된 파일 이름으로 저장
final_model_path = get_next_version_path(f'best_hand_gesture_{model_type}_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f'Final model saved to {final_model_path} with Val Acc: {best_val_acc:.4f}')

# 최고 모델 로드 및 평가
model.load_state_dict(torch.load(fixed_model_path))  # 또는 final_model_path
accuracy, conf_matrix = evaluate(model, test_loader, device)

# 확률 기반 혼동 행렬 계산 및 시각화
# 클래스별 행 단위로 정규화하여 확률로 변환
norm_conf_matrix = np.zeros(conf_matrix.shape)
for i in range(len(CLASSES)):
    if conf_matrix[i].sum() > 0:
        norm_conf_matrix[i] = conf_matrix[i] / conf_matrix[i].sum()
    
plt.figure(figsize=(10, 8))
plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.title('Normalized Confusion Matrix (Probability)')
plt.colorbar()
tick_marks = np.arange(len(CLASSES))
plt.xticks(tick_marks, CLASSES, rotation=45)
plt.yticks(tick_marks, CLASSES)

# 각 셀에 확률값 표시
thresh = norm_conf_matrix.max() / 2.
for i in range(norm_conf_matrix.shape[0]):
    for j in range(norm_conf_matrix.shape[1]):
        plt.text(j, i, f'{norm_conf_matrix[i, j]:.2f}',
                 horizontalalignment='center',
                 color='white' if norm_conf_matrix[i, j] > thresh else 'black')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
img_path = get_next_version_path(f'confusion_matrix_{model_type}_normalized.png')
plt.savefig(img_path)
print(f'Confusion matrix saved to {img_path}')
plt.show()

# 예측 함수 (새로운 이미지에 대한 예측)
def predict_gesture(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return CLASSES[predicted.item()], probability.cpu().numpy()

# 예측 예시
# gesture, probs = predict_gesture(model, 'test_image.jpg', val_test_transform, device)
# print(f'Predicted gesture: {gesture}')
# for i, class_name in enumerate(CLASSES):
#     print(f'{class_name}: {probs[i]*100:.2f}%')