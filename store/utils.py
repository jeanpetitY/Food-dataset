import base64
from sentence_transformers import SentenceTransformer
from .config.utils import PINECONE_API_KEY
from .config.utils import OPENAI_API_KEY
from openai import OpenAI
from pydantic import BaseModel
from transformers import  AutoTokenizer, pipeline, CLIPProcessor, CLIPModel
from typing import List, Optional, Union
from starlette.datastructures import UploadFile
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
import requests
from io import BytesIO
import torch
from dotenv import load_dotenv 
from transformers import AutoModelForCausalLM, TextIteratorStreamer
import threading


load_dotenv()


device = "cuda" if torch.cuda.is_available() else "cpu"

    

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


class StoreEmbeddings:
    def __init__(self, index_name, cloud='aws', region="us-east-1", metric='cosine'):
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self.text_model = None
        self.model_emb = None
        self.processor = None
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

    # ------------------ Pinecone setup ------------------

    def create_index(self, dimension=512):
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region)
        )
        return self.pc.Index(self.index_name)

    def use_index(self, name):
        self.index_name = name
        return self.pc.Index(self.index_name)

    # ------------------ Text embedding ------------------

    def set_text_model(self):  # e.g. SentenceTransformer(...)
        self.text_model = SentenceTransformer("intfloat/multilingual-e5-large")

    def embed_text(self, text: str):
        return self.text_model.encode(text, convert_to_numpy=True)
        

    def load_clip(self):
        if self.model_emb is None:
            self.model_emb = model
            self.processor = processor
            
    def embed_image(self, image_source: Union[str, UploadFile, BytesIO]):
        # Case 1 : Image URL (str)
        if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # Case 2 : Local path (str)
        elif isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")

        # Cas3 3 : Uploaded File (UploadFile ou BytesIO)
        elif isinstance(image_source, UploadFile):
            image_bytes = image_source.file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_source, BytesIO):
            image = Image.open(image_source).convert("RGB")
        else:
            raise ValueError("Unsupported image source type.")

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model_emb.get_image_features(**inputs)
        return features[0].cpu().numpy()

    # ------------------ Store embeddings ------------------
        
    def store_embedding(self, is_new_index: bool, id: str, embedding, metadata: dict, namespace="ns1"):
        if namespace == "text":
            index = self.use_index(name="tsotsatext")
        else:
            index = self.create_index() if is_new_index else self.use_index()

        index.upsert([{"id": id, 'values': embedding, "metadata": metadata}], namespace=namespace, batch_size=12)

    def store_text_embedding(self, id: str, text: str, metadata: dict, namespace='text'):
        emb = self.embed_text(text)
        self.store_embedding(id, emb, metadata, namespace)

    def store_image_embedding(self, id: str, image_path: any, metadata: dict, namespace='image'):
        emb = self.embed_image(image_path)
        self.store_embedding(id, emb, metadata, namespace)

    # ------------------ Semantic search ------------------

    def search_by_text(self, query: str, top_k=5, namespace='text'):
        emb = self.embed_text(query)
        return self._search(emb, top_k, namespace)

    def search_by_image(self, image_path: Union[str, UploadFile, BytesIO], top_k=5, namespace='image'):
        emb = self.embed_image(image_path)
        return self._search(emb, top_k, namespace)

    def _search(self, embedding, top_k, namespace):
        if namespace == "text":
            index = self.use_index("tsotsatext")
        else:
            index = self.use_index(self.index_name)
        results = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        # print(results)
        return [match['metadata'] for match in results['matches']]

    # ------------------ Unified query ------------------

    def smart_search(self, input_data, top_k=5):
        """
        input_data: str (text) or str (image_path ending with .jpg/.png/.jpeg)
        """
        if isinstance(input_data, str) and ('.jpg' or'.png' or'.jpeg' in input_data.lower()):
            return self.search_by_image(input_data, top_k=top_k, namespace='image')
        else:
            return self.search_by_text(input_data, top_k=top_k, namespace='text')

