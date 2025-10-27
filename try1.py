from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

class ImageCaptioner:
	def __init__(self):
		"""Initialize the model and processor once."""
		model_id = "Salesforce/blip-image-captioning-base"
		
		# Auto-select device: MPS or CPU
		self.device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
		
		print(f"Loading model on {self.device}...")
		
		self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
		self.model = AutoModelForImageTextToText.from_pretrained(model_id)

		if self.device == "cpu":
			try:
				from torch.ao.quantization import quantize_dynamic
				self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
			except (ImportError, Exception):
				pass

		self.model = self.model.to(self.device)
		print("Model loaded and ready!\n")

	def process_image(self, img_path: str):
		"""Process a single image and return its caption."""
		image = Image.open(img_path).convert("RGB")
		image = image.resize((384, 384), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)
		
		inputs = self.processor(images=image, return_tensors="pt").to(self.device)
		
		with torch.inference_mode():
			output = self.model.generate(**inputs, max_new_tokens=30)
		
		caption = self.processor.decode(output[0], skip_special_tokens=True)
		print(f"[{img_path}] Caption: {caption}\n")
		
		return caption

if __name__ == "__main__":
	print("=" * 60)
	print("Image Captioner - Interactive Mode")
	print("=" * 60)
	
	captioner = ImageCaptioner()
	
	print("Commands:")
	print("  - Enter image path (e.g., 'pic1.jpg')")
	print("  - Type 'quit' or 'exit' to close")
	print("=" * 60)
	
	while True:
		try:
			user_input = input("\nEnter image path: ").strip()
			
			if user_input.lower() in ['quit', 'exit', 'q']:
				print("Goodbye!")
				break
			
			if not user_input:
				continue
			
			try:
				captioner.process_image(user_input)
			except FileNotFoundError:
				print(f"❌ Error: File '{user_input}' not found.")
			except Exception as e:
				print(f"❌ Error: {e}")
		
		except (KeyboardInterrupt, EOFError):
			print("\nGoodbye!")
			break