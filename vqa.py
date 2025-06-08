
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from cca import *
import torch
from main import *
from torch.nn.functional import normalize
processor = BlipProcessor.from_pretrained("../blip")
model = BlipForQuestionAnswering.from_pretrained("../blip").to("cuda")
image_emb=[load_pickle('/mlx_devbox/users/liuhao.200207/playground/stat/full/image_embeddings_test.pkl'),
           load_pickle('/mlx_devbox/users/liuhao.200207/playground/stat/full/image_embeddings_train.pkl')]
image_ids=load_pickle('/mlx_devbox/users/liuhao.200207/playground/stat/full/test_ids.pkl')+load_pickle('/mlx_devbox/users/liuhao.200207/playground/stat/full/train_ids.pkl')
image_emb=np.concatenate(image_emb,axis=0)
image_emb=torch.tensor(image_emb).to("cuda")
clip=clip_model(device="cuda")
def get_answer(img_path,question,model=model,processor=processor):
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda:0")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def search(query,image_emb,clip,image_ids):
    '''用户对于图像的模糊描述,基于clip实现文搜图并返回图片的路径'''
    text_emb=clip.get_text_embedding([query]) # cuda torch
    image_emb=normalize(image_emb,dim=1)
    text_emb=normalize(text_emb,dim=1)
    image_emb=image_emb.to("cuda")
    text_emb=text_emb.to("cuda")
    pred=torch.einsum('id,jd->ij', text_emb,image_emb)
    pred=pred.cpu().numpy()
    pred=pred[0]
    pred=np.argsort(pred)[::-1]
    return image_ids[pred[0]]

id=search("a picture of a cat",image_emb,clip,image_ids)#'xxxx.jpg'
print(id)
answer=get_answer('../dataset/Images/'+id,'what is the color of the cat?')
print(answer)
answer=get_answer('../dataset/Images/'+id,'How many cats are there in the picture?')
print(answer)
answer=get_answer('../dataset/Images/'+id,'Where is the cat on?')
print(answer)

# nohup /root/anaconda3/envs/statenv/bin/python vqa.py > vqa.txt &