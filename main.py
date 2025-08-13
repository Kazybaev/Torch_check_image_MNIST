from fastapi import FastAPI, File, UploadFile    # FastAPI: создание API, загрузка файлов
from fastapi.responses import JSONResponse       # Возврат ответа в формате JSON
import torch                                     # PyTorch: основной модуль
import torch.nn as nn                            # Модули нейросети (слои, модели)
import torchvision.transforms as transforms      # Преобразования изображений (resize, toTensor и т.д.)
from PIL import Image                            # Работа с изображениями (открытие, преобразование)
import io                                        # Работа с байтами (нужно для `UploadFile.read()`)

# .
#   dataset ________ normalize ____________


# class CheckImage(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.covn = nn.Sequential(
#         nn.Conv2d(1, 32, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Dropout(0.25),
#         nn.Conv2d(32, 64, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Dropout(0.25),
#
#     )
#     self.fc = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(64 * 7 * 7, 128),
#         nn.ReLU(),
#         nn.Dropout(0.25),
#         nn.Linear(128, 10),
#     )
#   def forward(self, x):
#     x = self.covn(x)
#     x = self.fc(x)
#     return x


class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.covn = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),


    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
  def forward(self, x):
    x = self.covn(x)
    x = self.fc(x)
    return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckImage()
model.load_state_dict(torch.load("model_v4.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


image_dd = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
# uvicorn main:app --reload
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1).item()
            label = image_dd[predicted]
        return {"class": label, "index": predicted}
        # if predicted == 0:
        #     return {"T-Shirt": predicted}
        # elif predicted == 1:
        #     return {"Trouser": predicted}
        # elif predicted == 2:
        #     return {"Pullover": predicted}
        # elif predicted == 3:
        #     return {"Dress": predicted}
        # elif predicted == 4:
        #     return {"Coat": predicted}
        # elif predicted == 5:
        #     return {"Sandal": predicted}
        # elif predicted == 6:
        #     return {"Shirt": predicted}
        # elif predicted == 7:
        #     return {"Sneaker": predicted}
        # elif predicted == 8:
        #     return {"Bag": predicted}
        # elif predicted == 9:
        #     return {"Ankle boot": predicted}



    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
