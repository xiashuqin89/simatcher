import re
import math
from typing import Text, List

import pandas as pd


class HalfPatten:
    TYPE = "halfPatten"

    def __init__(self, x: List[Text]):
        self.x = x
        self.x_length = [len(i) for i in x]
        self.entropy = self.cal_entropy(self.x_length)
        self.regex_entropy = self.cal_entropy(x)

    @staticmethod
    def cal_entropy(serial: List) -> float:
        entropy = 0.
        for item in set(serial):
            entropy -= (float(serial.count(item)) / len(serial)) \
                       * math.log(float(serial.count(item)) / len(serial))
        return entropy

    def detection(self) -> Text:
        """todo add more pattern"""
        serial = ''.join(self.x)
        if len(serial) == 0:
            return ''
        elif serial.isdigit():
            return '\d'
        elif serial.isalnum():
            return '\w'
        else:
            return '.'

    def run(self, entropy_rule: float) -> Text:
        pattern = self.detection()
        if not pattern:
            return pattern

        if self.entropy > entropy_rule:
            return f'{pattern}+'
        elif self.entropy > 0:
            x_len = self.x_length
            x_len.sort()
            return f'{pattern}' + '{' + f'{x_len[0]},{x_len[-1]}' + '}'
        else:
            return f'{pattern}' + '{' + str(self.x_length[0]) + '}'


class FullPatten:
    TYPE = "fullPatten"

    def __init__(self, x: List[Text]):
        self.x = x
        self.entropy = 0.
        self.regex_entropy = 0.

    def run(self, entropy_rule: float):
        if len(set(self.x)) > 1:
            return f'({"|".join(list(set(self.x)))})'
        else:
            return f'({self.x[0]})'


class StrPattern:
    def __init__(self, y: List[Text], entropy: float):
        self.y = y
        self.entropy = entropy

    def re_split(self, delimiter: Text) -> List:
        sentence = [
            HalfPatten([x[:x.index(delimiter)] for x in self.y]),
            FullPatten([delimiter for x in self.y]),
            HalfPatten([x[x.index(delimiter) + len(delimiter):] for x in self.y])
        ]

        for i, item in enumerate(sentence):
            if item.entropy != 0.:
                if item.regex_entropy < self.entropy:
                    sentence[i] = FullPatten(item.x)
        return sentence


class AutoPattern:
    def __init__(self,
                 x: List[Text],
                 entropy: float = .3,
                 regex_entropy: float = .8):
        self.x = x
        self.entropy = entropy
        self.regex_entropy = regex_entropy
        self.sentence = []

    @staticmethod
    def process(df: pd.DataFrame, col: Text) -> Text:
        t = df[col][0]
        start, max_len = 0, 1
        i, j = 0, 1
        try:
            while (i + j) < len(t):
                if all(df[col].str.contains(t[i: i + j])):
                    j += 1
                else:
                    if max_len < j:
                        start = i
                        max_len = j
                    i += 1
                    j = 1
            return t[start: start + max_len - 1]
        except re.error:
            return ''

    def generate(self, deep: int = 1):
        x = self.x
        for i in range(deep):
            if len(self.sentence) == 0:
                sp = StrPattern(self.x, self.regex_entropy)
                df = pd.DataFrame({'data': x})
                delimiter = self.process(df, 'data')
                self.sentence = sp.re_split(delimiter)
                continue

            chance = None
            chance_entropy = 0.

            for j, item in enumerate(self.sentence):
                if item.regex_entropy != 0.:
                    if chance_entropy < item.regex_entropy:
                        chance = j
                        chance_entropy = item.regex_entropy

            if chance:
                di = {'data': self.sentence[chance].x}
                sp = StrPattern(di['data'], self.regex_entropy)
                df = pd.DataFrame(di)
                try:
                    delimiter = self.process(df, 'data')
                except IndexError as e:
                    break
                sentence = sp.re_split(delimiter)
                self.sentence = self.sentence[:chance] + sentence + self.sentence[chance + 1:]

    def build(self):
        return "".join([item.run(self.entropy) for item in self.sentence])


def main(entropy: float = .7,
         regex_entropy: float = .3,
         deep: int = 1):
    # s = ['www.asb.baids.com', 'www.ww.baidu.com', 'www.www.baidu.com']
    s = ['231017_LIVE_D10_230426_Server_CN.zip.nc', '12321_LIVE_D10_213_Server_CN.zip.nc']
    a = AutoPattern(s, entropy, regex_entropy)
    a.generate(deep)
    reg = a.build()
    print(reg)


if __name__ == "__main__":
    main()
    main(0, 0)
    main(0, 1.5)
    main(0, 1.5, deep=2)
    main(deep=2)
