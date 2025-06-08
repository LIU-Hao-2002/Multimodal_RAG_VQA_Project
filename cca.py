import pickle
from sklearn.cross_decomposition import CCA
import numpy as np
from loguru import logger
def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)
def dump_pickle(object,path):
    with open(path,'wb') as f:
        return pickle.dump(object,f)
def normalize(x, dim=1):
    norm = np.linalg.norm(x, axis=dim, keepdims=True)
    return x / norm
def eval_(results,mode='text2image'):
    labels=np.array([range(len(results))])
    if mode=='text2image':
        preds=results.argmax(axis=1) # dim=1是文本的acc,text2image是1个prompt文本去搜索全体图片的精度
    else:
        preds=results.argmax(axis=0) # dim=0是图片的损失/acc，即某个图片搜索全体文本的损失
    acc=(labels==preds).sum()
    acc/=len(results)
    return acc
def eval_final(results):
    txt2image=eval_(results=results,mode='text2image')
    image2txt=eval_(results=results,mode='image2text')
    final=0.5*txt2image+0.5*image2txt
    logger.info(f'txt2image acc: {txt2image}')
    logger.info(f'image2txt acc: {image2txt}')
    logger.info(f'final acc: {final}')
def main(image_embeddings, text_embeddings, name="pca"):
    image_embeddings = normalize(image_embeddings, dim=1)
    text_embeddings = normalize(text_embeddings, dim=1)
    batch_size = 50
    results = []
    cnt = 0
    for i in range(0, len(text_embeddings), batch_size):
        if cnt % 10 == 0:
            logger.info(f"processed {cnt} batches")
        cnt += 1
        text_embedding = text_embeddings[i:i + batch_size]
        # 使用 np.einsum 替代 torch.einsum
        temp = np.einsum('id,jd->ij', text_embedding, image_embeddings)
        results.append(temp)
    # 使用 np.concatenate 替代 torch.cat
    results = np.concatenate(results, axis=0)
    with open(f'evaluation/results_test_{name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    print(results.shape)
    return results

def main2(image_embeddings,text_embeddings,name="pca"):
    image_embeddings = normalize(image_embeddings, dim=1)
    text_embeddings = normalize(text_embeddings, dim=1)
    batch_size = 50
    results = []
    cnt = 0
    for i in range(0, len(text_embeddings), batch_size):
        if cnt % 10 == 0:
            logger.info(f"processed {cnt} batches")
        cnt += 1
        text_embedding = text_embeddings[i:i + batch_size]
        # 使用 np.einsum 替代 torch.einsum
        temp = np.einsum('id,jd->ij', text_embedding, image_embeddings)
        results.append(temp)
    # 使用 np.concatenate 替代 torch.cat
    results = np.concatenate(results, axis=0)
    with open(f'evaluation/results_test_{name}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    print(results.shape)
    logger.info(f"forward {name}")
    eval_final(results)

    text_embeddings,image_embeddings=image_embeddings,text_embeddings
    results = []
    cnt = 0
    for i in range(0, len(text_embeddings), batch_size):
        if cnt % 10 == 0:
            logger.info(f"processed {cnt} batches")
        cnt += 1
        text_embedding = text_embeddings[i:i + batch_size]
        # 使用 np.einsum 替代 torch.einsum
        temp = np.einsum('id,jd->ij', text_embedding, image_embeddings)
        results.append(temp)
    # 使用 np.concatenate 替代 torch.cat
    results = np.concatenate(results, axis=0)
    with open(f'evaluation/results_test_{name}_reverse.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(image_embeddings.shape)
    print(text_embeddings.shape)
    print(results.shape)
    logger.info(f"reversed {name}")
    eval_final(results)
    return results




class CustomCCA():
    def __init__(self, train_image_emb, train_text_emb, test_image_emb, test_text_emb, n_components=2):
        self.train_image_emb = load_pickle(train_image_emb) if isinstance(train_image_emb, str) else train_image_emb
        self.train_text_emb = load_pickle(train_text_emb) if isinstance(train_text_emb, str) else train_text_emb
        self.test_image_emb = load_pickle(test_image_emb) if isinstance(test_image_emb, str) else test_image_emb
        self.test_text_emb = load_pickle(test_text_emb) if isinstance(test_text_emb, str) else test_text_emb
        self.n_components = n_components
        self.cca = CCA(n_components=self.n_components)

    def cca_reduce(self):
        # 对训练数据进行典型相关分析
        self.cca.fit(self.train_image_emb, self.train_text_emb)
        # 对训练数据进行降维
        reduced_train_image_emb, reduced_train_text_emb = self.cca.transform(self.train_image_emb, self.train_text_emb)
        # 对测试数据进行降维
        reduced_test_image_emb, reduced_test_text_emb = self.cca.transform(self.test_image_emb, self.test_text_emb)
        # 获取投影矩阵
        projection_matrix_image = self.cca.x_weights_
        projection_matrix_text = self.cca.y_weights_
        logger.info(f"projection_matrix_image: {projection_matrix_image.shape}")
        logger.info(f"projection_matrix_text: {projection_matrix_text.shape}")
        return reduced_train_image_emb, reduced_train_text_emb, reduced_test_image_emb, reduced_test_text_emb, projection_matrix_image, projection_matrix_text

def main_cca():
    train_image=load_pickle('full/image_embeddings_train.pkl')
    train_text=load_pickle('full/text_embeddings_train.pkl')
    test_image=load_pickle('full/image_embeddings_test.pkl')
    test_text=load_pickle('full/text_embeddings_test.pkl')

    method=CustomCCA(train_image, train_text, test_image, test_text, n_components=256)
    reduced_train_image_emb, reduced_train_text_emb, reduced_test_image_emb, reduced_test_text_emb, projection_matrix_image, projection_matrix_text = method.cca_reduce()

    dump_pickle(reduced_train_image_emb,'cca/reduced_train_images.pkl')
    dump_pickle(reduced_train_text_emb,'cca/reduced_train_texts.pkl')
    dump_pickle(reduced_test_image_emb,'cca/reduced_test_images.pkl')
    dump_pickle(reduced_test_text_emb,'cca/reduced_test_texts.pkl')

    # test full outcome
    logger.info('*'*20)
    logger.info('test full outcome')
    full_results=main(test_image, test_text, name="full_test")
    eval_final(full_results)
    logger.info('*'*20)

    # test cca outcome
    logger.info('*'*20)
    logger.info('test reduced outcome')
    reduced_results=main(reduced_test_image_emb, reduced_test_text_emb, name="cca_test")
    eval_final(reduced_results)
    logger.info('*'*20)

    # test original trainset
    logger.info('*'*20)
    logger.info('test original trainset')
    original_results=main(train_image, train_text, name="full_train")
    eval_final(original_results)
    logger.info('*'*20)

    # test cca outcome on trainset
    logger.info('*'*20)
    logger.info('test reduced outcome on trainset')
    reduced_results=main(reduced_train_image_emb, reduced_train_text_emb, name="cca_train")
    eval_final(reduced_results)
    logger.info('*'*20)

if __name__=='__main__':
    main_cca()


# nohup /root/anaconda3/envs/statenv/bin/python3 cca.py > cca.txt &