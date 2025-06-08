
from sklearn.decomposition import PCA
from cca import *
class CustomPCA():
    def __init__(self, train_emb_path, test_emb_path, variance_threshold=0.8):
        self.train_embeds = load_pickle(train_emb_path) if type(train_emb_path)==str else train_emb_path
        self.test_embeds = load_pickle(test_emb_path) if type(test_emb_path)==str else test_emb_path
        self.variance_threshold = variance_threshold
        self.pca = None
        self.n_components = None

    def determine_n_components(self):
        # 初始化 PCA 以保留所有主成分
        pca_all = PCA()
        pca_all.fit(self.train_embeds)
        # 计算累计方差贡献率
        cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
        # 找到满足方差阈值的最小主成分数量
        self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        logger.info(f"Determined n_components: {self.n_components}")
        # 重新初始化 PCA 并使用确定的 n_components
        self.pca = PCA(n_components=self.n_components)

    def pca_reduce(self):
        if self.pca is None:
            self.determine_n_components()
        # 对训练嵌入进行主成分分析
        self.pca.fit(self.train_embeds)
        # 对训练嵌入进行降维
        reduced_train_embeds = self.pca.transform(self.train_embeds)
        # 对测试嵌入进行降维
        reduced_test_embeds = self.pca.transform(self.test_embeds)
        # 获取投影矩阵
        projection_matrix = self.pca.components_.T
        logger.info(f"projection_matrix: {projection_matrix.shape}")
        return reduced_train_embeds, reduced_test_embeds, projection_matrix

def main_pca():
    train_image=load_pickle('full/image_embeddings_train.pkl')
    train_text=load_pickle('full/text_embeddings_train.pkl')
    test_image=load_pickle('full/image_embeddings_test.pkl')
    test_text=load_pickle('full/text_embeddings_test.pkl')
    train_emb=np.concatenate([train_image,train_text],axis=0)
    test_emb=np.concatenate([test_image,test_text],axis=0)
    method=CustomPCA(train_emb,test_emb,variance_threshold=0.999)
    method.determine_n_components()
    reduced_train_embeds, reduced_test_embeds, projection_matrix=method.pca_reduce()
    reduced_train_images=reduced_train_embeds[:len(train_image)]
    reduced_train_texts=reduced_train_embeds[len(train_image):]
    reduced_test_images=reduced_test_embeds[:len(test_image)]
    reduced_test_texts=reduced_test_embeds[len(test_image):]
    dump_pickle(reduced_train_images,'pca/reduced_train_images.pkl')
    dump_pickle(reduced_train_texts,'pca/reduced_train_texts.pkl')
    dump_pickle(reduced_test_images,'pca/reduced_test_images.pkl')
    dump_pickle(reduced_test_texts,'pca/reduced_test_texts.pkl')

    # test full outcome
    logger.info('*'*20)
    logger.info('test full outcome')
    full_results=main(test_image, test_text, name="full_test")
    eval_final(full_results)
    logger.info('*'*20)


    # test pca outcome
    logger.info('*'*20)
    logger.info('test reduced outcome')
    reduced_results=main(reduced_test_images, reduced_test_texts, name="pca_test")
    eval_final(reduced_results)
    logger.info('*'*20)


    # test original trainset
    logger.info('*'*20)
    logger.info('test original trainset')
    original_results=main(train_image, train_text, name="full_train")
    eval_final(original_results)
    logger.info('*'*20)

    # test pca outcome on trainset
    logger.info('*'*20)
    logger.info('test reduced outcome on trainset')
    reduced_results=main(reduced_train_images, reduced_train_texts, name="pca_train")
    eval_final(reduced_results)
    logger.info('*'*20)

if __name__=='__main__':
    main_pca()


    
# nohup /root/anaconda3/envs/statenv/bin/python3 pca.py > pca.txt &