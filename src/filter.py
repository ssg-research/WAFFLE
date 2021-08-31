import hashlib
import itertools
import math
import hmac
# import bitstring
import torch


def default_key(length: int):
    import string
    import random
    random.seed(42)
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


class WatermarkFilter:
    def __init__(self, key: str, shape: (int, int, int), precision: int = 16, probability: float = 5 / 1000, hash_f: callable = hashlib.sha256) -> None:
        self.key = key.encode("utf-8")
        self.shape = shape
        self.hash_f = hash_f
        self.precision = precision
        self.probability = probability
        self.bound = math.floor((2 ** self.precision) * self.probability)
        self.strip = 25  # None in the final codebase; used for efficiency

        self.__check_len(torch.rand(self.shape).numpy().tobytes(), self.key)

    @staticmethod
    # def __bytes_to_bits(msg: str) -> str:
    #     return bitstring.BitArray(hex=msg).bin

    @staticmethod
    def __check_len(sample: bytes, key: bytes) -> None:
        len_s = len(sample)
        len_k = len(key)
        if len_s > len_k:
            raise ValueError("Secret key shorter than the message: {} < {}".format(len_k, len_s))

    def __hmac(self, key: bytes, msg: bytes, strip: int = None) -> str:
        if strip is not None:
            key = itertools.islice(key, 0, strip)
            msg = itertools.islice(msg, 0, strip)

        return hmac.new(key, msg, self.hash_f).hexdigest()

    # def is_watermark(self, image: torch.FloatTensor) -> bool:
    #     if image.shape != self.shape:
    #         raise AssertionError("Image shape {} different from expected {}.".format(image.shape, self.shape))
    #
    #     hashed = self.__hmac(key=self.key, msg=image.numpy().tobytes())
    #     bits = self.__bytes_to_bits(hashed)
    #     return int(bits[:self.precision], 2) <= self.bound

    @staticmethod
    def flip_label(original_prediction: torch.FloatTensor, n=4) -> torch.FloatTensor:
        _, idx = torch.topk(original_prediction, n)
        idx = idx.squeeze()
        flipped = original_prediction.clone()
        flipped[0, idx[0]] = original_prediction[0, idx[n - 1]]
        flipped[0, idx[n - 1]] = original_prediction[0, idx[0]]
        return flipped

    @staticmethod
    def shuffle_label(original_prediction: torch.FloatTensor, n=5) -> torch.FloatTensor:
        values, idx = torch.topk(original_prediction, n)
        idx, values = idx.squeeze(), values.squeeze()
        shuffled = original_prediction.clone()
        values = values[torch.randperm(values.size()[0])]
        for idx, val in zip(idx, values):
            shuffled[0, idx] = val

        return shuffled
