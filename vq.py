
from cca import *
def quantize_embedding(embedding, bit=8):
    """
    对 embedding 进行量化，支持 int8 和 int4
    :param embedding: 输入的 embedding 矩阵
    :param bit: 量化位数，8 或 4
    :return: 量化后的 embedding 矩阵、反量化所需的缩放因子和零点
    """
    if bit == 8:
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        scale = (max_val - min_val) / 255
        zero_point = np.round(-min_val / scale)
        # 确保零点在 int8 范围内
        zero_point = np.clip(zero_point, -128, 127).astype(np.int8)
        quantized = np.round(embedding / scale + zero_point)
        # 确保量化值在 int8 范围内
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        return quantized, scale, zero_point
    elif bit == 4:
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        scale = (max_val - min_val) / 15
        zero_point = np.round(-min_val / scale).astype(np.uint8)
        quantized = np.round(embedding / scale + zero_point).clip(0, 15).astype(np.uint8)
        # 检查第一个维度的元素数量是否为奇数
        if quantized.shape[0] % 2 != 0:
            # 舍弃最后一个元素
            quantized = quantized[:-1]
        # 将两个 4 位整数打包到一个 8 位整数中
        quantized_packed = (quantized[::2] << 4) | quantized[1::2]
        return quantized_packed, scale, zero_point
    else:
        raise ValueError("Unsupported bit depth. Only 8 and 4 are supported.")

def dequantize_embedding(quantized, scale, zero_point, bit=8, original_shape=None):
    """
    对量化后的 embedding 进行反量化
    :param quantized: 量化后的 embedding 矩阵
    :param scale: 缩放因子
    :param zero_point: 零点
    :param bit: 量化位数，8 或 4
    :param original_shape: 原始 embedding 的形状
    :return: 反量化后的 embedding 矩阵
    """
    if bit == 8:
        return (quantized.astype(np.float32) - zero_point) * scale
    elif bit == 4:
        # 先将 quantized 展平为一维数组
        quantized_flat = quantized.flatten()
        # 解包 8 位整数为两个 4 位整数
        unpacked = np.zeros(quantized_flat.size * 2, dtype=np.uint8)
        unpacked[::2] = (quantized_flat >> 4) & 0x0F
        unpacked[1::2] = quantized_flat & 0x0F
        unpacked = unpacked.reshape((-1,) + original_shape[1:])
        # 如果原始形状第一个维度是奇数，去除填充的最后一行
        if original_shape[0] % 2 != 0:
            unpacked = unpacked[:original_shape[0]]
        return (unpacked.astype(np.float32) - zero_point) * scale
    else:
        raise ValueError("Unsupported bit depth. Only 8 and 4 are supported.")


def main_quantized(image_embeddings, text_embeddings, name="pca", bit=8):
    """
    对 embedding 进行量化并测试
    :param image_embeddings: 图像 embedding
    :param text_embeddings: 文本 embedding
    :param name: 保存结果的文件名
    :param bit: 量化位数，8 或 4
    """
    # 量化图像和文本 embedding
    quantized_image_emb, image_scale, image_zero_point = quantize_embedding(image_embeddings, bit)
    quantized_text_emb, text_scale, text_zero_point = quantize_embedding(text_embeddings, bit)
    dump_pickle(quantized_image_emb, f"quantized/image_embeddings_{name}_quantized_{bit}bit.pkl")
    dump_pickle(quantized_text_emb, f"quantized/text_embeddings_{name}_quantized_{bit}bit.pkl")

    # 压缩存储，计算时再反量化，时间换空间
    # 反量化
    dequantized_image_emb = dequantize_embedding(quantized_image_emb, image_scale, image_zero_point, bit, image_embeddings.shape)
    dequantized_text_emb = dequantize_embedding(quantized_text_emb, text_scale, text_zero_point, bit, text_embeddings.shape)

    # 测试反量化后的 embedding
    results = main(dequantized_image_emb, dequantized_text_emb, name=f"{name}_quantized_{bit}bit")
    eval_final(results)

def main_quantization_test():
    train_image = load_pickle('full/image_embeddings_train.pkl')
    train_text = load_pickle('full/text_embeddings_train.pkl')
    test_image = load_pickle('full/image_embeddings_test.pkl')
    test_text = load_pickle('full/text_embeddings_test.pkl')

    # 测试 8 位量化
    logger.info('*'*20)
    logger.info('Test 8-bit quantization')
    main_quantized(train_image, train_text, name="train", bit=8)
    logger.info('*'*20)

    # 测试 8 位量化
    logger.info('*'*20)
    logger.info('Test 4-bit quantization')
    main_quantized(train_image, train_text, name="train", bit=4)
    logger.info('*'*20)

    # 测试 8 位量化
    logger.info('*'*20)
    logger.info('Test 8-bit quantization')
    main_quantized(test_image, test_text, name="test", bit=8)
    logger.info('*'*20)

    # 测试 4 位量化
    logger.info('*'*20)
    logger.info('Test 4-bit quantization')
    main_quantized(test_image, test_text, name="test", bit=4)
    logger.info('*'*20)




if __name__ == '__main__':
    main_quantization_test()