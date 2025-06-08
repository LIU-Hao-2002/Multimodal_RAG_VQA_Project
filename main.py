from transformers import CLIPProcessor, CLIPModel
import os
import pickle
import random
from PIL import Image
from torch.nn.functional import cosine_similarity,normalize
import torch
from loguru import logger
# Load the model and processor
class clip_model():
    def __init__(self,device='cpu',model_path="/mlx_devbox/users/liuhao.200207/playground/clip",batch_size=32):
        model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = model.to(device)
        self.batch_size=batch_size

    def get_image_embedding(self,image):
        image_embeddings=[]
        for i in range(0,len(image),self.batch_size):
            logger.info(f'processing {i} to {i+self.batch_size}')
            batch_image=image[i:i+self.batch_size]
            inputs = self.processor(images=batch_image, return_tensors="pt")
            del batch_image
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            del inputs
            outputs = outputs.cpu()
            image_embeddings.append(outputs)
            del outputs
        image_embeddings=torch.cat(image_embeddings,dim=0)
        return image_embeddings
    def get_text_embedding(self,text):
        text_embeddings=[]
        for i in range(0,len(text),self.batch_size):
            logger.info(f'processing {i} to {i+self.batch_size}')
            batch_text=text[i:i+self.batch_size]
            inputs = self.processor(text=batch_text, return_tensors="pt", padding=True)
            del batch_text
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            del inputs
            outputs = outputs.cpu()
            text_embeddings.append(outputs)
            del outputs
        text_embeddings=torch.cat(text_embeddings,dim=0)
        return text_embeddings
    def get_probs(self,image,text):
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        return probs

    
def get_caption_image():
    image2caption={}
    with open('/mlx_devbox/users/liuhao.200207/playground/dataset/captions.txt', 'r') as f:
        for line in f:
            if '.jpg,' not in line:continue
            image_id, caption = line.strip().split('.jpg,')
            image_id=image_id+'.jpg'
            image2caption[image_id] = caption
    caption2image={}
    for image_id, caption in image2caption.items():
        caption2image[caption] = image_id
    return image2caption,caption2image

def load_images(path='/mlx_devbox/users/liuhao.200207/playground/dataset/Images',test_num=500):
    images={}
    for image in os.listdir(path):
        images[image]=Image.open(path+'/'+image).convert('RGB')
    # sample train and test
    print(f"all images {len(images)}")
    test_num=len(images)//4
    test_ids=list(images.keys())
    random.shuffle(test_ids)
    train_ids=test_ids[test_num:]
    test_ids=test_ids[:test_num]
    train_images={}
    test_images={}
    for image_id in train_ids:
        train_images[image_id]=images[image_id]
    for image_id in test_ids:
        test_images[image_id]=images[image_id]
    with open('train_ids.pkl','wb') as f:
        pickle.dump(train_ids,f)
    with open('test_ids.pkl','wb') as f:
        pickle.dump(test_ids,f)
    return train_images,test_images # 1000 and 500 instances

def main(images,name="test",device='cuda:0'):
    id=list(images.keys())
    sample_images=[images[i] for i in id]
    sample_captions=[image2caption[i] for i in id]
    image_embeddings=model.get_image_embedding(sample_images)
    image_embeddings=image_embeddings.numpy()
    with open(f'image_embeddings_{name}.pkl','wb') as f:
        pickle.dump(image_embeddings,f)
    text_embeddings=model.get_text_embedding(sample_captions)
    text_embeddings=text_embeddings.numpy()
    with open(f'text_embeddings_{name}.pkl','wb') as f:
        pickle.dump(text_embeddings,f)
    # cosine
    image_embeddings = torch.from_numpy(image_embeddings)
    text_embeddings = torch.from_numpy(text_embeddings)
    image_embeddings = image_embeddings.to(device)
    image_embeddings = normalize(image_embeddings, dim=1)
    text_embeddings = normalize(text_embeddings, dim=1)

    # 使用 torch.einsum 计算 10x10 余弦相似度矩阵
    # results = torch.einsum('id,jd->ij', image_embeddings, text_embeddings)
    batch_size=50
    results=[]
    cnt=0
    for text_embedding in text_embeddings.split(batch_size):
        if cnt%10==0:
            logger.info(f"processed {cnt} batches")
        cnt+=1
        text_embedding = text_embedding.to(device)
        temp = torch.einsum('id,jd->ij', text_embedding,image_embeddings)
        temp = temp.cpu()
        del text_embedding
        results.append(temp)
        del temp
    results=torch.cat(results,dim=0)
    results=results.numpy()
    with open(f'results_{name}.pkl','wb') as f:
        pickle.dump(results,f)
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    print(results.shape)
if __name__ == '__main__':
    image2caption,caption2image=get_caption_image()
    train_images,test_images=load_images()
    model=clip_model(device='cpu')
    main(train_images,name="train",device='cpu')
    main(test_images,name="test",device='cpu')






# nohup /root/anaconda3/envs/statenv/bin/python3 main.py > nohup.out &
# /root/anaconda3/envs/statenv/bin/python3
# /root/anaconda3/envs/statenv/bin/pip