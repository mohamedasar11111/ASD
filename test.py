import torch
from transformers import pipeline , TrOCRProcessor , VisionEncoderDecoderModel
from datasets import load_dataset
import soundfile as sf
from PIL import Image
from IPython.display import display
import pytesseract

from google.colab import drive
drive.mount('/content/drive')
     
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

def show_image(pathStr):
  img = Image.open(pathStr).convert("RGB")
  display(img)
  return img

def ocr_image(src_img):
  return pytesseract.image_to_string(src_img)

pictureText = show_image('/content/handwriting.png')
exportedText = ocr_image(pictureText)
print(exportedText)
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser(exportedText, forward_params={"speaker_embeddings": speaker_embedding})
sf.write("exportedText.wav", speech["audio"], samplerate=speech["sampling_rate"])
