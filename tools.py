from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = """USE WHEN: You need visual context about an image. 
    INPUT: File path (e.g., "image.jpg"). OUTPUT: Text description."""
    
    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)
        return processor.decode(output[0], skip_special_tokens=True)
    
    async def _arun(self, img_path: str) -> str:
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = """USE WHEN: Identifying objects/coordinates in images. 
    INPUT: File path. OUTPUT: List of [x1,y1,x2,y2] boxes with labels."""
    
    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += f'[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] '
            detections += f'{model.config.id2label[int(label)]} {float(score):.2f}\n'
        return detections
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")
